# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Lightweight TTNN helper modules for DPT-Large.

These intentionally mirror the Hugging Face ViT pieces but use TTNN /
fallback_ops for execution when a TT device is available. They keep
dependencies minimal so we can iterate on sharding and fused-op choices
in `tt_configs.py`.
"""

from __future__ import annotations

from typing import Optional

import math
import time
import torch

from .tt_cnn_ops import TTConv2dCached

try:
    import ttnn  # type: ignore
except Exception:  # pragma: no cover
    ttnn = None  # type: ignore

try:
    from tt_lib.fallback_ops import fallback_ops  # type: ignore
except Exception:  # pragma: no cover
    fallback_ops = None  # type: ignore

try:
    from models.common.utility_functions import torch_to_tt_tensor_rm, torch2tt_tensor  # type: ignore
except Exception:  # pragma: no cover
    torch_to_tt_tensor_rm = None  # type: ignore
    torch2tt_tensor = None  # type: ignore


def _tt_from_torch_rm(t: torch.Tensor, device):
    if callable(torch_to_tt_tensor_rm):
        return torch_to_tt_tensor_rm(t, device, put_on_device=True)
    if ttnn is None:
        raise RuntimeError("TT runtime is unavailable; cannot convert torch tensor to TT tensor")
    return ttnn.from_torch(
        t.to(dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )


def _tt_from_torch_tile(t: torch.Tensor, device):
    if callable(torch2tt_tensor):
        return torch2tt_tensor(t, device)
    if ttnn is None:
        raise RuntimeError("TT runtime is unavailable; cannot convert torch tensor to TT tensor")
    return ttnn.from_torch(
        t.to(dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def pad_tokens_3d(x_3d: torch.Tensor, pad_multiple: int = 32):
    """
    Pad a 3D [B, N, C] tensor along the sequence dimension to a multiple of
    `pad_multiple`. Returns the padded tensor and the original N.
    """
    if x_3d.dim() != 3:
        raise ValueError(f"pad_tokens_3d expects [B, N, C], got {tuple(x_3d.shape)}")

    B, N, C = x_3d.shape
    if N == 0:
        return x_3d, N

    N_pad = math.ceil(N / pad_multiple) * pad_multiple
    if N_pad == N:
        return x_3d, N

    pad_len = N_pad - N
    pad = torch.zeros(B, pad_len, C, dtype=x_3d.dtype, device=x_3d.device)
    x_padded = torch.cat([x_3d, pad], dim=1)
    return x_padded, N


def unpad_tokens_3d(x_3d_padded: torch.Tensor, original_N: int):
    """
    Remove padding added by `pad_tokens_3d`. Expects a 3D [B, N_pad, C] tensor.
    """
    if x_3d_padded.dim() != 3:
        raise ValueError(f"unpad_tokens_3d expects [B, N_pad, C], got {tuple(x_3d_padded.shape)}")
    return x_3d_padded[:, :original_N, :]


def build_attn_padding_mask_4d(
    padded_seq_len: int, valid_seq_len: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Build an additive attention mask that blocks padded key/value tokens.

    Returns a tensor shaped [1, 1, S, S] with zeros for valid keys and -inf for
    padded keys (columns >= valid_seq_len). This is meant to be *added* to the
    attention score matrix before softmax (PyTorch/TTNN SDPA convention).
    """
    s = int(padded_seq_len)
    v = int(valid_seq_len)
    if v < 0 or s < 0:
        raise ValueError(f"Invalid lengths: padded_seq_len={s}, valid_seq_len={v}")
    if v >= s or s == 0:
        return torch.zeros((1, 1, s, s), dtype=dtype)
    mask = torch.zeros((1, 1, s, s), dtype=dtype)
    mask[..., :, v:] = float("-inf")
    return mask


def _to_torch_attn_mask(attn_mask, dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
    """Convert TT/torch attention masks to torch [B|1, H|1, S, S] form."""
    if attn_mask is None:
        return None

    if ttnn is not None and isinstance(attn_mask, ttnn.Tensor):
        attn_mask_host = attn_mask.cpu()
        if hasattr(attn_mask_host, "layout") and attn_mask_host.layout == ttnn.TILE_LAYOUT:
            attn_mask_host = attn_mask_host.to(ttnn.ROW_MAJOR_LAYOUT)
        mask_torch = attn_mask_host.to_torch()
    elif torch.is_tensor(attn_mask):
        mask_torch = attn_mask
    else:
        mask_torch = torch.as_tensor(attn_mask)

    if mask_torch.dim() == 2:
        mask_torch = mask_torch.unsqueeze(0).unsqueeze(0)
    elif mask_torch.dim() == 3:
        mask_torch = mask_torch.unsqueeze(1)
    elif mask_torch.dim() != 4:
        raise ValueError(f"Unsupported attention mask shape: {tuple(mask_torch.shape)}")

    return mask_torch.to(dtype=dtype, device=device)


def _apply_attn_mask(attn_logits: torch.Tensor, attn_mask) -> torch.Tensor:
    """Apply additive attention mask before softmax."""
    if attn_mask is None:
        return attn_logits
    mask_torch = _to_torch_attn_mask(attn_mask, dtype=attn_logits.dtype, device=attn_logits.device)
    return attn_logits + mask_torch


def _ttnn_linear_with_optional_program_config(*, x, w, bias, dtype, memory_config, program_config, op_name: str = "unknown"):
    """Call ttnn.linear with best-effort program_config plumbing across runtime versions.

    In perf runs we want to surface when a program_config cannot be used, since
    silently falling back to the default kernel changes both performance and
    determinism.
    """
    try:
        from .perf_counters import inc_program_config_fallback, strict_program_config_enabled
    except Exception:  # pragma: no cover
        inc_program_config_fallback = None
        strict_program_config_enabled = lambda: False  # type: ignore

    if program_config is None:
        return ttnn.linear(x, w, bias=bias, dtype=dtype, memory_config=memory_config)
    try:
        return ttnn.linear(x, w, bias=bias, dtype=dtype, memory_config=memory_config, program_config=program_config)
    except TypeError:
        # Older ttnn builds may not expose program_config for this op.
        if callable(inc_program_config_fallback):
            inc_program_config_fallback(op=op_name, reason="kwarg_unsupported")
        if strict_program_config_enabled():
            raise
        return ttnn.linear(x, w, bias=bias, dtype=dtype, memory_config=memory_config)
    except Exception:
        # Some runtimes accept the kwarg but can reject specific program configs at runtime.
        if callable(inc_program_config_fallback):
            inc_program_config_fallback(op=op_name, reason="runtime_rejected")
        if strict_program_config_enabled():
            raise
        return ttnn.linear(x, w, bias=bias, dtype=dtype, memory_config=memory_config)


def _program_config_fuses_activation(program_config) -> bool:
    if program_config is None:
        return False
    try:
        return getattr(program_config, "fused_activation", None) is not None
    except Exception:
        # If we can't introspect, assume the provided program config is intentional.
        return True


def _ttnn_matmul_with_optional_program_config(
    *,
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    dtype,
    memory_config,
    program_config,
    op_name: str = "unknown",
):
    # Mirror the linear retry logic so perf runs can be strict and observable.
    try:
        from .perf_counters import inc_program_config_fallback, strict_program_config_enabled
    except Exception:  # pragma: no cover
        inc_program_config_fallback = None
        strict_program_config_enabled = lambda: False  # type: ignore

    if program_config is None:
        return ttnn.matmul(a, b, dtype=dtype, memory_config=memory_config)

    try:
        return ttnn.matmul(a, b, dtype=dtype, memory_config=memory_config, program_config=program_config)
    except TypeError:
        if callable(inc_program_config_fallback):
            inc_program_config_fallback(op=op_name, reason="kwarg_unsupported")
        if strict_program_config_enabled():
            raise
        return ttnn.matmul(a, b, dtype=dtype, memory_config=memory_config)
    except Exception:
        if callable(inc_program_config_fallback):
            inc_program_config_fallback(op=op_name, reason="runtime_rejected")
        if strict_program_config_enabled():
            raise
        return ttnn.matmul(a, b, dtype=dtype, memory_config=memory_config)


def _ttnn_attention_softmax_with_optional_program_config(
    *, x: ttnn.Tensor, attention_mask, head_size: int, program_config, op_name: str = "unknown", memory_config=None
):
    try:
        from .perf_counters import inc_program_config_fallback, strict_program_config_enabled
    except Exception:  # pragma: no cover
        inc_program_config_fallback = None
        strict_program_config_enabled = lambda: False  # type: ignore

    kwargs = dict(attention_mask=attention_mask, head_size=int(head_size))
    if memory_config is not None:
        kwargs["memory_config"] = memory_config

    if program_config is None:
        return ttnn.transformer.attention_softmax_(x, **kwargs)

    try:
        return ttnn.transformer.attention_softmax_(x, program_config=program_config, **kwargs)
    except TypeError:
        if callable(inc_program_config_fallback):
            inc_program_config_fallback(op=op_name, reason="kwarg_unsupported")
        if strict_program_config_enabled():
            raise
        return ttnn.transformer.attention_softmax_(x, **kwargs)
    except Exception:
        if callable(inc_program_config_fallback):
            inc_program_config_fallback(op=op_name, reason="runtime_rejected")
        if strict_program_config_enabled():
            raise
        return ttnn.transformer.attention_softmax_(x, **kwargs)


def _ttnn_is_sharded(x) -> bool:
    try:
        if hasattr(ttnn, "is_sharded"):
            return bool(ttnn.is_sharded(x))
    except Exception:
        pass
    try:
        return bool(x.is_sharded())
    except Exception:
        return False


class TTLinear:
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        device,
        output_mem: Optional[ttnn.MemoryConfig] = None,
        fast_gelu: bool = False,
        program_config=None,
    ):
        if ttnn is None:
            raise RuntimeError("TT runtime is unavailable; cannot construct TTLinear")
        # weight: [out, in]
        self.weight_torch = weight
        self.bias_torch = bias
        self.device = device
        # Upload weights in the format expected by ttnn.linear: [in, out] in TILE.
        w_t = torch.transpose(weight.detach(), -1, -2).contiguous()
        self.weight = ttnn.from_torch(w_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        # Bias can remain row-major.
        self.bias = (
            ttnn.from_torch(bias.detach(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            if bias is not None
            else None
        )
        self.output_mem = output_mem
        # When fast_gelu is True, enable fused GELU on the matmul where possible.
        # Use "gelu_approx" string so kernels can pick the approximate fast path when supported.
        self.activation = "gelu_approx" if fast_gelu else None
        self.program_config = program_config

    def __call__(self, x: ttnn.Tensor, **mm_opts) -> ttnn.Tensor:
        shape = tuple(getattr(x, "shape", ()))
        if len(shape) == 4:
            b, one, n, c = shape
            if int(one) != 1:
                raise ValueError(f"TTLinear expected [B,1,N,C] or [B,N,C], got {shape}")
            x4 = x
            wants_4d = True
        elif len(shape) == 3:
            b, n, c = shape
            x4 = ttnn.reshape(x, (int(b), 1, int(n), int(c)))
            wants_4d = False
        else:
            raise ValueError(f"TTLinear expected 3D/4D TT tensor, got {shape}")

        pc_obj = self.program_config
        memcfg = getattr(pc_obj, "mlp_memcfg", None) if pc_obj is not None else None
        if memcfg is None:
            memcfg = self.output_mem or ttnn.DRAM_MEMORY_CONFIG

        prog = None
        if pc_obj is not None:
            prog = getattr(pc_obj, "ff1_program_config", None) if self.activation else getattr(pc_obj, "ff2_program_config", None)

        out4 = _ttnn_linear_with_optional_program_config(
            x=x4,
            w=self.weight,
            bias=self.bias,
            dtype=ttnn.bfloat16,
            memory_config=memcfg,
            program_config=prog,
            op_name="ttlinear_ff1" if self.activation else "ttlinear_ff2",
        )
        if self.activation and not _program_config_fuses_activation(prog):
            out4 = ttnn.gelu(out4)

        if wants_4d:
            return out4
        out_shape = tuple(getattr(out4, "shape", (int(b), 1, int(n), 0)))
        out_c = int(out_shape[-1])
        return ttnn.reshape(out4, (int(b), int(n), out_c))


class TTPatchEmbedding:
    def __init__(
        self,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        device,
        stride: int,
        padding: int,
        output_mem: Optional[ttnn.MemoryConfig] = None,
        allow_cpu_fallback: bool = True,
    ):
        self.device = device
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        self.conv = None
        self.tt_conv = TTConv2dCached.from_tensors(
            weight_torch=conv_weight,
            bias_torch=conv_bias,
            stride=(int(stride), int(stride)),
            padding=(int(padding), int(padding)),
            dilation=(1, 1),
            groups=1,
        )
        if self.allow_cpu_fallback:
            if fallback_ops is None:
                raise RuntimeError(
                    "TTPatchEmbedding fallback requires tt_lib.fallback_ops. "
                    "Install TT runtime dependencies or disable CPU fallback."
                )
            wt = _tt_from_torch_rm(conv_weight, device)
            bs = _tt_from_torch_rm(conv_bias, device)
            self.conv = fallback_ops.Conv2d(
                in_channels=conv_weight.shape[1],
                out_channels=conv_weight.shape[0],
                kernel_size=conv_weight.shape[2],
                weights=wt,
                biases=bs,
                stride=stride,
                padding=padding,
            )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        try:
            return self.tt_conv(x, device=self.device)
        except Exception as exc:
            if not self.allow_cpu_fallback or self.conv is None:
                raise RuntimeError("TTPatchEmbedding TT conv path failed and CPU fallback is disabled") from exc
            return self.conv(x)


class TTLayerNorm:
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        device,
        output_mem: Optional[ttnn.MemoryConfig] = None,
        program_config=None,
        allow_cpu_fallback: bool = True,
    ):
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.device = device
        self.output_mem = output_mem
        self.program_config = program_config
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        try:
            # ttnn.layer_norm consumes device-side affine params.
            self.weight_tt = ttnn.from_torch(
                weight.detach().unsqueeze(0).to(dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self.bias_tt = ttnn.from_torch(
                bias.detach().unsqueeze(0).to(dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        except Exception:
            self.weight_tt = None
            self.bias_tt = None

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        try:
            if self.weight_tt is None or self.bias_tt is None:
                raise RuntimeError("LayerNorm affine params not available on device")

            try:
                from .perf_counters import inc_program_config_fallback, strict_program_config_enabled
            except Exception:  # pragma: no cover
                inc_program_config_fallback = None
                strict_program_config_enabled = lambda: False  # type: ignore

            kwargs = {
                "weight": self.weight_tt,
                "bias": self.bias_tt,
                "epsilon": self.eps,
            }
            pc = self.program_config
            ln_pc = getattr(pc, "ln_program_config", None) if pc is not None else None
            # Perf LN program configs typically assume a sharded input tensor.
            x_is_sharded = _ttnn_is_sharded(x)
            if ln_pc is not None and x_is_sharded:
                kwargs["program_config"] = ln_pc

            cc = getattr(pc, "ln_compute_config", None) if pc is not None else None
            if cc is not None:
                kwargs["compute_kernel_config"] = cc
            try:
                return ttnn.layer_norm(x, **kwargs)
            except TypeError:
                # Backward compat for older runtimes without program_config / compute_kernel_config kwargs.
                if ("program_config" in kwargs or "compute_kernel_config" in kwargs) and callable(inc_program_config_fallback):
                    inc_program_config_fallback(op="layer_norm", reason="kwarg_unsupported")
                if strict_program_config_enabled() and ("program_config" in kwargs or "compute_kernel_config" in kwargs):
                    raise
                kwargs.pop("program_config", None)
                kwargs.pop("compute_kernel_config", None)
                return ttnn.layer_norm(x, **kwargs)
            except Exception:
                # Some runtimes accept these kwargs but can fail for particular inputs/configs.
                # Retry once without the perf configs to avoid hard-failing the entire pipeline.
                if ("program_config" in kwargs or "compute_kernel_config" in kwargs) and callable(inc_program_config_fallback):
                    inc_program_config_fallback(op="layer_norm", reason="runtime_rejected")
                if strict_program_config_enabled() and ("program_config" in kwargs or "compute_kernel_config" in kwargs):
                    raise
                kwargs.pop("program_config", None)
                kwargs.pop("compute_kernel_config", None)
                try:
                    return ttnn.layer_norm(x, **kwargs)
                except Exception:
                    # Sharded LayerNorm support is runtime-dependent. When it fails, explicitly
                    # run an interleaved "island" and reshard the output back, rather than
                    # forcing a CPU fallback in perf/trace runs.
                    if x_is_sharded:
                        # In perf mode, do not silently interleave/reshard; fail so the issue is observable.
                        if strict_program_config_enabled():
                            raise
                        try:
                            from .perf_counters import inc_ln_island_interleave, inc_ln_island_reshard
                        except Exception:  # pragma: no cover
                            inc_ln_island_interleave = None
                            inc_ln_island_reshard = None

                        mc_in = None
                        try:
                            mc_in = ttnn.get_memory_config(x)
                        except Exception:
                            mc_in = None

                        t0 = time.perf_counter()
                        try:
                            x_int = ttnn.to_memory_config(
                                x, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16
                            )
                        except TypeError:
                            x_int = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        if callable(inc_ln_island_interleave):
                            inc_ln_island_interleave((time.perf_counter() - t0) * 1000.0)

                        y_int = ttnn.layer_norm(x_int, **kwargs)

                        if mc_in is not None:
                            try:
                                if mc_in.is_sharded():
                                    t1 = time.perf_counter()
                                    try:
                                        y = ttnn.to_memory_config(
                                            y_int, memory_config=mc_in, dtype=ttnn.bfloat16
                                        )
                                    except TypeError:
                                        y = ttnn.to_memory_config(y_int, memory_config=mc_in)
                                    if callable(inc_ln_island_reshard):
                                        inc_ln_island_reshard((time.perf_counter() - t1) * 1000.0)
                                    return y
                            except Exception:
                                pass
                        return y_int
                    raise
        except Exception as exc:
            if not self.allow_cpu_fallback:
                raise RuntimeError("TTLayerNorm device path failed and CPU fallback is disabled") from exc
            # Host fallback path for strict parity/debugging.
            x_host = x.cpu()
            if hasattr(x_host, "layout") and x_host.layout == ttnn.TILE_LAYOUT:
                x_host = x_host.to(ttnn.ROW_MAJOR_LAYOUT)
            x_torch = x_host.to_torch()
            y_torch = torch.nn.functional.layer_norm(
                x_torch,
                normalized_shape=[x_torch.shape[-1]],
                weight=self.weight,
                bias=self.bias,
                eps=self.eps,
            )
            y = ttnn.from_torch(
                y_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            return y


class TTAttention:
    def __init__(
        self,
        q_w,
        q_b,
        k_w,
        k_b,
        v_w,
        v_b,
        proj_w,
        proj_b,
        num_heads: int,
        head_dim: int,
        device,
        output_mem: Optional[ttnn.MemoryConfig] = None,
        program_config=None,
        allow_cpu_fallback: bool = True,
    ):
        # Fused QKV weights: pack as [Q_head0, K_head0, V_head0, Q_head1, K_head1, V_head1, ...]
        # to match TTNN sharded split-heads expectations (see vit.md custom_preprocessor).
        # Keep the raw torch weights around for a host-based attention path.
        self.q_w = q_w
        self.q_b = q_b
        self.k_w = k_w
        self.k_b = k_b
        self.v_w = v_w
        self.v_b = v_b
        self.proj_w = proj_w
        self.proj_b = proj_b

        # Prepare fused QKV weights for device-side linear.
        # Torch linear weights are [out, in]; ttnn.linear expects [in, out] in TILE.
        H = int(num_heads)
        D = int(head_dim)
        # Reshape to [heads, head_dim, in] then interleave along head_dim axis.
        q_w_hdi = q_w.detach().reshape(H, D, -1)
        k_w_hdi = k_w.detach().reshape(H, D, -1)
        v_w_hdi = v_w.detach().reshape(H, D, -1)
        qkv_w = torch.cat([q_w_hdi, k_w_hdi, v_w_hdi], dim=1).reshape(H * 3 * D, -1)  # [3*out, in]
        wqkv_t = torch.transpose(qkv_w, -1, -2).contiguous()  # [in, 3*out]

        q_b_hd = q_b.detach().reshape(H, D)
        k_b_hd = k_b.detach().reshape(H, D)
        v_b_hd = v_b.detach().reshape(H, D)
        bqkv = torch.cat([q_b_hd, k_b_hd, v_b_hd], dim=1).reshape(H * 3 * D)  # [3*out]
        try:
            self._wqkv_tt = ttnn.from_torch(
                wqkv_t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self._bqkv_tt = ttnn.from_torch(
                bqkv.unsqueeze(0),
                dtype=(ttnn.bfloat8_b if hasattr(ttnn, "bfloat8_b") else ttnn.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            # Per-Q/K/V biases for the explicit sharded attention path. We add these
            # after split-heads to avoid large broadcast buffers in the fused QKV linear.
            bias_dtype = ttnn.bfloat8_b if hasattr(ttnn, "bfloat8_b") else ttnn.bfloat16
            self._q_bias_tt = ttnn.from_torch(
                q_b_hd.unsqueeze(0).unsqueeze(2),  # [1, H, 1, D]
                dtype=bias_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self._k_bias_tt = ttnn.from_torch(
                k_b_hd.unsqueeze(0).unsqueeze(3),  # [1, H, D, 1] (K is transposed)
                dtype=bias_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self._v_bias_tt = ttnn.from_torch(
                v_b_hd.unsqueeze(0).unsqueeze(2),  # [1, H, 1, D]
                dtype=bias_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        except Exception:
            self._wqkv_tt = None
            self._bqkv_tt = None
            self._q_bias_tt = None
            self._k_bias_tt = None
            self._v_bias_tt = None
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        # Persist knobs so callers can tune memory/program behavior.
        self.output_mem = output_mem
        self.program_config = program_config
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        # Cache height-sharded memory configs for [B, N] attention shapes.
        self._height_shard_mc_cache: dict[tuple[int, int, int, int], dict[str, object]] = {}
        # Cache block-sharded QKV output specs keyed by (B, N, C, grid_x, grid_y).
        self._qkv_block_shard_mc_cache: dict[tuple[int, int, int, int, int], object] = {}
        # Pre-create TTNN projection weights for device-side output matmul
        try:
            # Use transposed weights to avoid transpose_b flag during linear
            w_t = torch.transpose(self.proj_w.detach(), -1, -2).contiguous()
            self._proj_w_tt = ttnn.from_torch(
                w_t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            self._proj_b_tt = ttnn.from_torch(
                self.proj_b.detach(),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
        except Exception:
            self._proj_w_tt = None
            self._proj_b_tt = None

    def __call__(self, x: ttnn.Tensor, **mm_opts) -> ttnn.Tensor:
        """
        Attention with optional TTNN SDPA acceleration.

        Contract:
          - Input:  TT tensor [B, N, C] (tokens)
          - Output: TT tensor [B, N, C]

        Path A (fast): Q/K/V on host, SDPA on TT via ttnn.transformer.scaled_dot_product_attention,
                        output projection on host; returns TT tensor.
        Path B (fallback): full host attention as before.
        """
        # Stay on device: expect TT tensor [B, N, C] (TILE or ROW_MAJOR)
        if not hasattr(x, "shape"):
            raise ValueError("TTAttention expects a TT tensor input")
        shape = tuple(getattr(x, "shape", ()))
        if len(shape) == 4:
            B, _, N, C = shape
            x3 = ttnn.reshape(x, (int(B), int(N), int(C)))
        elif len(shape) == 3:
            B, N, C = shape
            x3 = x
        else:
            raise ValueError(f"TTAttention expects TT tensor with 3D/4D shape, got {shape}")
        input_is_4d = len(shape) == 4
        H = self.num_heads
        D = self.head_dim
        if C != H * D:
            raise ValueError(f"TTAttention hidden dim mismatch: C={C}, H*D={H * D}")
        # Stage-2/3: tokens may be block-sharded across the encoder.
        tokens_shard_mc = mm_opts.get("tokens_shard_mc", None)
        cfg = getattr(self, "program_config", None)
        input_is_sharded = tokens_shard_mc is not None and (_ttnn_is_sharded(x) or _ttnn_is_sharded(x3))
        use_default_attn_programs = bool(getattr(cfg, "use_default_attention_programs", False)) if cfg is not None else False
        # Prefer explicit attention on sharded operands; fall back to the SDPA island
        # only when not sharded.
        explicit_sharded_attn = input_is_sharded
        try:
            if self._wqkv_tt is None or self._bqkv_tt is None:
                raise RuntimeError("QKV fused weights not available on device")
            split_memcfg = getattr(cfg, "split_heads_memcfg", None) if cfg is not None else None
            if split_memcfg is None:
                split_memcfg = getattr(ttnn, "L1_HEIGHT_SHARDED_MEMORY_CONFIG", None)

            # QKV fused linear must be block-sharded for TTNN split-heads to take the
            # fully sharded path (vit.md pattern). Prefer the canonical TTNN constant
            # so the op can pick the right sharded kernel from program_config.
            memcfg = getattr(cfg, "qkv_memcfg", None) if cfg is not None else None
            if explicit_sharded_attn:
                memcfg = getattr(ttnn, "L1_BLOCK_SHARDED_MEMORY_CONFIG", None) or memcfg
            if memcfg is None:
                memcfg = getattr(self, "output_mem", None) or ttnn.DRAM_MEMORY_CONFIG
            qkv_pc = getattr(cfg, "qkv_program_config", None) if cfg is not None else None
            if qkv_pc is not None and not (_ttnn_is_sharded(x) or _ttnn_is_sharded(x3)):
                qkv_pc = None
            # On some TTNN builds, the explicit QKV program_config can over-allocate static
            # circular buffers on small grids (e.g., 8x1) and clash with L1 buffers.
            # Prefer the default kernel here; attention/softmax program configs remain enabled.
            if explicit_sharded_attn:
                qkv_pc = None
            qkv_dtype = ttnn.bfloat16
            if explicit_sharded_attn and hasattr(ttnn, "bfloat8_b"):
                # Split-heads sharded path is sensitive to L1 circular-buffer pressure.
                # Match vit.md and use BF8 for the fused QKV projection in perf mode.
                qkv_dtype = ttnn.bfloat8_b
            if explicit_sharded_attn:
                # Reduce L1 fragmentation / static CB overlap risk (vit demo pattern).
                try:
                    if hasattr(ttnn, "reallocate"):
                        x3 = ttnn.reallocate(x3)
                except Exception:
                    pass
            qkv3 = _ttnn_linear_with_optional_program_config(
                x=x3,
                w=self._wqkv_tt,
                bias=(None if explicit_sharded_attn else self._bqkv_tt),
                dtype=qkv_dtype,
                memory_config=memcfg,
                program_config=qkv_pc,
                op_name="attn_qkv",
            )
            if explicit_sharded_attn:
                # `split_query_key_value_and_split_heads` expects the fused QKV tensor to be
                # block-sharded across the encoder grid (vit demo reshard step).
                grid = getattr(cfg, "grid", None) if cfg is not None else None
                if grid is None:
                    raise RuntimeError("Missing encoder core grid in TT config")
                grid_x, grid_y = int(grid[0]), int(grid[1])
                cache_key = ("qkv3", int(B), int(N), int(C), int(grid_x), int(grid_y))
                qkv_shard_mc = self._qkv_block_shard_mc_cache.get(cache_key)
                if qkv_shard_mc is None:
                    if not hasattr(ttnn, "create_sharded_memory_config"):
                        raise RuntimeError("ttnn.create_sharded_memory_config unavailable; cannot shard QKV output")
                    core_grid = ttnn.CoreGrid(y=int(grid_y), x=int(grid_x))
                    qkv_shard_mc = ttnn.create_sharded_memory_config(
                        getattr(qkv3, "padded_shape", (int(B), int(N), int(3 * C))),
                        core_grid=core_grid,
                        strategy=ttnn.ShardStrategy.BLOCK,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    )
                    self._qkv_block_shard_mc_cache[cache_key] = qkv_shard_mc
                qkv3 = ttnn.to_memory_config(qkv3, qkv_shard_mc)
                try:
                    if hasattr(ttnn, "reallocate"):
                        qkv3 = ttnn.reallocate(qkv3)
                except Exception:
                    pass
            # Split to heads (vit.md pattern): returns [B, H, N, D] (Q,V) and [B, H, D, N] (K) by default.
            try:
                q_tt, k_tt, v_tt = ttnn.transformer.split_query_key_value_and_split_heads(
                    qkv3, num_heads=H, memory_config=split_memcfg
                )
            except TypeError:
                # Backward compat: older runtimes may not expose these kwargs.
                q_tt, k_tt, v_tt = ttnn.transformer.split_query_key_value_and_split_heads(qkv3, num_heads=H)

            if explicit_sharded_attn and getattr(self, "_q_bias_tt", None) is not None:
                # Apply Q/K/V biases post-split to avoid large sharded broadcast in QKV linear.
                qb, kb, vb = self._q_bias_tt, self._k_bias_tt, self._v_bias_tt
                # Broadcast+add kernels are sensitive to bias placement; keep biases in L1.
                try:
                    qb = ttnn.to_memory_config(qb, memory_config=ttnn.L1_MEMORY_CONFIG)
                    kb = ttnn.to_memory_config(kb, memory_config=ttnn.L1_MEMORY_CONFIG)
                    vb = ttnn.to_memory_config(vb, memory_config=ttnn.L1_MEMORY_CONFIG)
                except Exception:
                    pass
                q_tt = ttnn.add(q_tt, qb)
                k_tt = ttnn.add(k_tt, kb)
                v_tt = ttnn.add(v_tt, vb)

            if explicit_sharded_attn:
                # Explicit attention path (ViT TTNN pattern) on sharded operands:
                # QK -> fused scale+mask+softmax -> AV, all height-sharded.
                qk_pc = None if use_default_attn_programs else getattr(cfg, "qk_program_config", None)
                softmax_pc = None if use_default_attn_programs else getattr(cfg, "softmax_program_config", None)
                av_pc = None if use_default_attn_programs else getattr(cfg, "av_program_config", None)

                scores_memcfg = split_memcfg
                attn_scores = _ttnn_matmul_with_optional_program_config(
                    a=q_tt,
                    b=k_tt,
                    dtype=ttnn.bfloat16,
                    memory_config=scores_memcfg,
                    program_config=qk_pc,
                    op_name="attn_qk_matmul",
                )
                attn_mask = mm_opts.get("attn_mask", None)
                attn_probs = _ttnn_attention_softmax_with_optional_program_config(
                    x=attn_scores,
                    attention_mask=attn_mask,
                    head_size=int(D),
                    program_config=softmax_pc,
                    op_name="attn_softmax",
                    memory_config=scores_memcfg,
                )
                ctx_tt = _ttnn_matmul_with_optional_program_config(
                    a=attn_probs,
                    b=v_tt,
                    dtype=ttnn.bfloat16,
                    memory_config=scores_memcfg,
                    program_config=av_pc,
                    op_name="attn_av_matmul",
                )
                try:
                    if hasattr(ttnn, "deallocate"):
                        ttnn.deallocate(attn_scores)
                        ttnn.deallocate(attn_probs)
                except Exception:
                    pass

                try:
                    merged = ttnn.transformer.concatenate_heads(
                        ctx_tt,
                        memory_config=(tokens_shard_mc if tokens_shard_mc is not None else ttnn.L1_MEMORY_CONFIG),
                    )
                except TypeError:
                    merged = ttnn.transformer.concatenate_heads(ctx_tt)
                ctx = merged
            else:
                # Legacy SDPA path (interleaved). Keep it for non-sharded / debug.
                scale = 1.0 / math.sqrt(D)
                sdpa_kwargs = dict(is_causal=False, scale=scale)
                if "attn_mask" in mm_opts and mm_opts["attn_mask"] is not None:
                    sdpa_kwargs["attn_mask"] = mm_opts["attn_mask"]
                pc = getattr(self, "program_config", None)
                if pc is not None:
                    try:
                        if hasattr(pc, "sdpa_program_config"):
                            maybe_pc = pc.sdpa_program_config(N)
                        else:
                            maybe_pc = pc
                        if maybe_pc is not None:
                            sdpa_kwargs["program_config"] = maybe_pc
                    except Exception:
                        pass
                ctx_tt = ttnn.transformer.scaled_dot_product_attention(q_tt, k_tt, v_tt, **sdpa_kwargs)
                try:
                    merged = ttnn.transformer.concatenate_heads(ctx_tt, memory_config=memcfg)
                except TypeError:
                    merged = ttnn.transformer.concatenate_heads(ctx_tt)
                ctx = merged
        except Exception as exc:
            if not self.allow_cpu_fallback:
                raise RuntimeError("TTAttention device path failed and CPU fallback is disabled") from exc
            # Fallback to host path (slow), preserving correctness
            x_host = x.cpu()
            if hasattr(x_host, "layout") and x_host.layout == ttnn.TILE_LAYOUT:
                x_host = x_host.to(ttnn.ROW_MAJOR_LAYOUT)
            x_torch = x_host.to_torch()
            x_f32 = x_torch.to(dtype=torch.float32)
            q = torch.nn.functional.linear(x_f32, self.q_w.to(dtype=torch.float32), self.q_b.to(dtype=torch.float32))
            k = torch.nn.functional.linear(x_f32, self.k_w.to(dtype=torch.float32), self.k_b.to(dtype=torch.float32))
            v = torch.nn.functional.linear(x_f32, self.v_w.to(dtype=torch.float32), self.v_b.to(dtype=torch.float32))
            q_ = q.view(B, N, H, D).permute(0, 2, 1, 3)
            k_ = k.view(B, N, H, D).permute(0, 2, 1, 3)
            v_ = v.view(B, N, H, D).permute(0, 2, 1, 3)
            scale = 1.0 / math.sqrt(D)
            attn = torch.matmul(q_, k_.transpose(-2, -1)) * scale
            attn = _apply_attn_mask(attn, mm_opts.get("attn_mask"))
            attn = torch.softmax(attn, dim=-1)
            ctx_ = torch.matmul(attn, v_)
            ctx_host = ctx_.permute(0, 2, 1, 3).contiguous().view(B, N, C)
            ctx = ttnn.from_torch(
                ctx_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            if input_is_4d:
                ctx = ttnn.reshape(ctx, (B, 1, N, C))

        # Output projection on device when possible.
        out: ttnn.Tensor
        if self._proj_w_tt is not None and self._proj_b_tt is not None:
            # Keep attention projection on rank-3 [B, N, C] to match vit.md sharded path.
            ctx3 = ctx if len(getattr(ctx, "shape", ())) == 3 else ttnn.reshape(ctx, (int(B), int(N), int(C)))
            cfg = getattr(self, "program_config", None)
            memcfg = getattr(cfg, "proj_memcfg", None) if cfg is not None else None
            if memcfg is None:
                memcfg = getattr(self, "output_mem", None) or ttnn.DRAM_MEMORY_CONFIG
            if input_is_sharded and tokens_shard_mc is not None:
                memcfg = tokens_shard_mc
            proj_pc = getattr(cfg, "proj_program_config", None) if cfg is not None else None
            if proj_pc is not None and not _ttnn_is_sharded(ctx3):
                proj_pc = None
            out3 = _ttnn_linear_with_optional_program_config(
                x=ctx3,
                w=self._proj_w_tt,
                bias=self._proj_b_tt,
                dtype=ttnn.bfloat16,
                memory_config=memcfg,
                program_config=proj_pc,
                op_name="attn_proj",
            )
            out = out3 if not input_is_4d else ttnn.reshape(out3, (int(B), 1, int(N), int(C)))
        else:
            if not self.allow_cpu_fallback:
                raise RuntimeError("TTAttention projection weights unavailable and CPU fallback is disabled")
            out_host = torch.nn.functional.linear(
                (ctx.cpu().to_torch() if hasattr(ctx, "cpu") else ctx).to(dtype=torch.float32),
                self.proj_w.to(dtype=torch.float32),
                self.proj_b.to(dtype=torch.float32),
            )  # [B, N, C]
            out = ttnn.from_torch(
                out_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            if input_is_4d:
                out = ttnn.reshape(out, (B, 1, N, C))
        return out


class TTMLP:
    def __init__(
        self,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        device,
        output_mem=None,
        program_config=None,
        allow_cpu_fallback: bool = True,
    ):
        self.device = device
        self.output_mem = output_mem
        self.program_config = program_config
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        # Host fallback linears kept for safety
        self._fc1_host = TTLinear(
            fc1_w, fc1_b, device, output_mem=output_mem, fast_gelu=True, program_config=program_config
        )
        self._fc2_host = TTLinear(fc2_w, fc2_b, device, output_mem=output_mem, program_config=program_config)
        # Pre-upload FC1/FC2 weights/biases for device path
        try:
            w1_t = torch.transpose(fc1_w.detach(), -1, -2).contiguous()
            w2_t = torch.transpose(fc2_w.detach(), -1, -2).contiguous()
            self.w1_tt = ttnn.from_torch(w1_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            self.b1_tt = ttnn.from_torch(
                fc1_b.detach(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            self.w2_tt = ttnn.from_torch(w2_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            self.b2_tt = ttnn.from_torch(
                fc2_b.detach(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
        except Exception:
            self.w1_tt = self.b1_tt = self.w2_tt = self.b2_tt = None

    def __call__(self, x: ttnn.Tensor, **mm_opts) -> ttnn.Tensor:
        try:
            if self.w1_tt is None or self.w2_tt is None:
                raise RuntimeError("Device MLP weights unavailable")

            shape = tuple(getattr(x, "shape", ()))
            if len(shape) == 4:
                B, one, N, C = shape
                if int(one) != 1:
                    raise ValueError(f"TTMLP expected [B,1,N,C] or [B,N,C], got {shape}")
                x3 = ttnn.reshape(x, (int(B), int(N), int(C)))
                wants_4d = True
            elif len(shape) == 3:
                B, N, C = shape
                x3 = x
                wants_4d = False
            else:
                raise ValueError(f"TTMLP expects 3D/4D TT tensor, got {shape}")

            tokens_shard_mc = mm_opts.get("tokens_shard_mc", None)
            incoming_sharded = tokens_shard_mc is not None and _ttnn_is_sharded(x3)
            cfg = getattr(self, "program_config", None)

            memcfg = getattr(cfg, "mlp_memcfg", None) if cfg is not None else None
            if incoming_sharded:
                memcfg = getattr(ttnn, "L1_BLOCK_SHARDED_MEMORY_CONFIG", None) or tokens_shard_mc or memcfg
            if memcfg is None:
                memcfg = getattr(self, "output_mem", None) or ttnn.DRAM_MEMORY_CONFIG

            ff1_pc = getattr(cfg, "ff1_program_config", None) if cfg is not None else None
            ff2_pc = getattr(cfg, "ff2_program_config", None) if cfg is not None else None
            if ff1_pc is not None and not _ttnn_is_sharded(x3):
                ff1_pc = None
            if ff2_pc is not None and not _ttnn_is_sharded(x3):
                ff2_pc = None

            y1 = _ttnn_linear_with_optional_program_config(
                x=x3,
                w=self.w1_tt,
                bias=self.b1_tt,
                dtype=ttnn.bfloat16,
                memory_config=memcfg,
                program_config=ff1_pc,
                op_name="mlp_ff1",
            )
            if not _program_config_fuses_activation(ff1_pc):
                y1 = ttnn.gelu(y1)

            y2 = _ttnn_linear_with_optional_program_config(
                x=y1,
                w=self.w2_tt,
                bias=self.b2_tt,
                dtype=ttnn.bfloat16,
                memory_config=memcfg,
                program_config=ff2_pc,
                op_name="mlp_ff2",
            )

            if incoming_sharded and tokens_shard_mc is not None:
                # Restore the encoder-wide token sharding spec for residual adds.
                y2 = ttnn.to_memory_config(y2, memory_config=tokens_shard_mc, dtype=ttnn.bfloat16)

            if wants_4d:
                y2 = ttnn.reshape(y2, (int(B), 1, int(N), int(y2.shape[-1])))
            return y2
        except Exception as exc:
            if not self.allow_cpu_fallback:
                raise RuntimeError("TTMLP device path failed and CPU fallback is disabled") from exc
            # Host fallback path (keeps parity)
            x = self._fc1_host(x, **mm_opts)
            x = self._fc2_host(x, **mm_opts)
            return x


class TTTransformerBlock:
    def __init__(
        self,
        state_dict,
        base: str,
        num_heads: int,
        head_dim: int,
        eps: float,
        device,
        output_mem: Optional[ttnn.MemoryConfig] = None,
        program_config=None,
        allow_cpu_fallback: bool = True,
    ):
        q_w = state_dict[f"{base}.attention.attention.query.weight"]
        q_b = state_dict[f"{base}.attention.attention.query.bias"]
        k_w = state_dict[f"{base}.attention.attention.key.weight"]
        k_b = state_dict[f"{base}.attention.attention.key.bias"]
        v_w = state_dict[f"{base}.attention.attention.value.weight"]
        v_b = state_dict[f"{base}.attention.attention.value.bias"]
        proj_w = state_dict[f"{base}.attention.output.dense.weight"]
        proj_b = state_dict[f"{base}.attention.output.dense.bias"]
        fc1_w = state_dict[f"{base}.intermediate.dense.weight"]
        fc1_b = state_dict[f"{base}.intermediate.dense.bias"]
        fc2_w = state_dict[f"{base}.output.dense.weight"]
        fc2_b = state_dict[f"{base}.output.dense.bias"]
        ln1_w = state_dict[f"{base}.layernorm_before.weight"]
        ln1_b = state_dict[f"{base}.layernorm_before.bias"]
        ln2_w = state_dict[f"{base}.layernorm_after.weight"]
        ln2_b = state_dict[f"{base}.layernorm_after.bias"]

        self.ln1 = TTLayerNorm(
            ln1_w,
            ln1_b,
            eps,
            device,
            output_mem=output_mem,
            program_config=program_config,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        self.ln2 = TTLayerNorm(
            ln2_w,
            ln2_b,
            eps,
            device,
            output_mem=output_mem,
            program_config=program_config,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        self.attn = TTAttention(
            q_w,
            q_b,
            k_w,
            k_b,
            v_w,
            v_b,
            proj_w,
            proj_b,
            num_heads,
            head_dim,
            device,
            output_mem=output_mem,
            program_config=program_config,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        self.mlp = TTMLP(
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
            device,
            output_mem=output_mem,
            program_config=program_config,
            allow_cpu_fallback=allow_cpu_fallback,
        )

    def __call__(self, x: ttnn.Tensor, **mm_opts) -> ttnn.Tensor:
        # Stage-2/3: perf path must remain fully on-device with no DRAM "islands".
        h = self.ln1(x)
        h = self.attn(h, **mm_opts)
        x = ttnn.add(x, h)
        h = self.ln2(x)
        h = self.mlp(h, **mm_opts)
        x = ttnn.add(x, h)
        return x
