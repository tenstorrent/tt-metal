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


def _ttnn_linear_with_optional_program_config(*, x, w, bias, dtype, memory_config, program_config):
    """Call ttnn.linear with best-effort program_config plumbing across runtime versions."""
    if program_config is None:
        return ttnn.linear(x, w, bias=bias, dtype=dtype, memory_config=memory_config)
    try:
        return ttnn.linear(x, w, bias=bias, dtype=dtype, memory_config=memory_config, program_config=program_config)
    except TypeError:
        # Older ttnn builds may not expose program_config for this op.
        return ttnn.linear(x, w, bias=bias, dtype=dtype, memory_config=memory_config)
    except Exception:
        # Some runtimes accept the kwarg but can reject specific program configs at runtime.
        return ttnn.linear(x, w, bias=bias, dtype=dtype, memory_config=memory_config)


def _program_config_fuses_activation(program_config) -> bool:
    if program_config is None:
        return False
    try:
        return getattr(program_config, "fused_activation", None) is not None
    except Exception:
        # If we can't introspect, assume the provided program config is intentional.
        return True


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
        # weight: [out, in]
        self.weight_torch = weight
        self.bias_torch = bias
        self.device = device
        # Keep weights in TILE layout on device to avoid RM<->TILE conversions on every matmul.
        wt = weight.unsqueeze(0).unsqueeze(0)  # 1,1,out,in
        self.weight = _tt_from_torch_tile(wt, device)
        # Bias can remain in row-major layout; keep conversion lightweight.
        self.bias = _tt_from_torch_rm(bias.unsqueeze(0).unsqueeze(0), device) if bias is not None else None
        self.output_mem = output_mem
        # When fast_gelu is True, enable fused GELU on the matmul where possible.
        # Use "gelu_approx" string so kernels can pick the approximate fast path when supported.
        self.activation = "gelu_approx" if fast_gelu else None
        self.program_config = program_config

    def __call__(self, x: ttnn.Tensor, **mm_opts) -> ttnn.Tensor:
        # Phase A: run linear on host to keep shapes simple and rely on
        # torch.nn.functional.linear for correctness. The logical contract
        # remains [B, N, C] in and out, even if some callers still provide
        # a 4D [B, 1, N, C] tensor.
        x_host = x.cpu()
        if hasattr(x_host, "layout") and x_host.layout == ttnn.TILE_LAYOUT:
            x_host = x_host.to(ttnn.ROW_MAJOR_LAYOUT)
        x_torch = x_host.to_torch()

        if x_torch.dim() == 4:
            b, one, n, c = x_torch.shape
            if one != 1:
                raise ValueError(f"TTLinear expected shape [B,1,N,C] or [B,N,C], got {tuple(x_torch.shape)}")
            x_torch = x_torch.view(b, n, c)
        elif x_torch.dim() != 3:
            raise ValueError(f"TTLinear expects 3D [B, N, C], got shape {tuple(x_torch.shape)}")

        x_f32 = x_torch.to(dtype=torch.float32)
        w = self.weight_torch.to(dtype=torch.float32)
        b = self.bias_torch.to(dtype=torch.float32) if self.bias_torch is not None else None
        y = torch.nn.functional.linear(x_f32, w, b)

        if self.activation == "gelu_approx":
            y = torch.nn.functional.gelu(y)

        out = ttnn.from_torch(
            y,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        return out


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

            kwargs = {
                "weight": self.weight_tt,
                "bias": self.bias_tt,
                "epsilon": self.eps,
            }
            pc = self.program_config
            ln_pc = getattr(pc, "ln_program_config", None) if pc is not None else None
            if ln_pc is not None:
                kwargs["program_config"] = ln_pc

            cc = getattr(pc, "ln_compute_config", None) if pc is not None else None
            if cc is not None:
                kwargs["compute_kernel_config"] = cc
            try:
                return ttnn.layer_norm(x, **kwargs)
            except TypeError:
                # Backward compat for older runtimes without program_config / compute_kernel_config kwargs.
                kwargs.pop("program_config", None)
                kwargs.pop("compute_kernel_config", None)
                return ttnn.layer_norm(x, **kwargs)
            except Exception:
                # Some runtimes accept these kwargs but can fail for particular inputs/configs.
                # Retry once without the perf configs to avoid hard-failing the entire pipeline.
                kwargs.pop("program_config", None)
                kwargs.pop("compute_kernel_config", None)
                return ttnn.layer_norm(x, **kwargs)
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
        # Fused QKV weights: stack [Q;K;V] along out_dim
        # Keep the raw torch weights around for a host-based attention path.
        self.q_w = q_w
        self.q_b = q_b
        self.k_w = k_w
        self.k_b = k_b
        self.v_w = v_w
        self.v_b = v_b
        self.proj_w = proj_w
        self.proj_b = proj_b

        # Prepare fused QKV weights for device-side linear: shape [in, 3*out]
        q_w_t = torch.transpose(q_w.detach(), -1, -2).contiguous()
        k_w_t = torch.transpose(k_w.detach(), -1, -2).contiguous()
        v_w_t = torch.transpose(v_w.detach(), -1, -2).contiguous()
        wqkv_t = torch.cat([q_w_t, k_w_t, v_w_t], dim=-1)
        bqkv = torch.cat([q_b.detach(), k_b.detach(), v_b.detach()], dim=0)
        try:
            self._wqkv_tt = ttnn.from_torch(
                wqkv_t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self._bqkv_tt = ttnn.from_torch(
                bqkv,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
        except Exception:
            self._wqkv_tt = None
            self._bqkv_tt = None
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        # Persist knobs so callers can tune memory/program behavior.
        self.output_mem = output_mem
        self.program_config = program_config
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
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
        elif len(shape) == 3:
            B, N, C = shape
        else:
            raise ValueError(f"TTAttention expects TT tensor with 3D/4D shape, got {shape}")
        H = self.num_heads
        D = self.head_dim
        if C != H * D:
            raise ValueError(f"TTAttention hidden dim mismatch: C={C}, H*D={H * D}")
        # Device-side fused QKV linear
        if len(shape) == 3:
            x4 = ttnn.reshape(x, (B, 1, N, C))
        else:
            x4 = x
        try:
            if self._wqkv_tt is None or self._bqkv_tt is None:
                raise RuntimeError("QKV fused weights not available on device")
            cfg = getattr(self, "program_config", None)
            memcfg = getattr(cfg, "qkv_memcfg", None) if cfg is not None else None
            if memcfg is None:
                memcfg = getattr(self, "output_mem", None) or ttnn.DRAM_MEMORY_CONFIG
            qkv_pc = getattr(cfg, "qkv_program_config", None) if cfg is not None else None
            qkv4 = _ttnn_linear_with_optional_program_config(
                x=x4,
                w=self._wqkv_tt,
                bias=self._bqkv_tt,
                dtype=ttnn.bfloat16,
                memory_config=memcfg,
                program_config=qkv_pc,
            )
            # [B, 1, N, 3C] -> [B, N, 3C]
            qkv3 = qkv4 if len(getattr(qkv4, "shape", [])) == 3 else ttnn.reshape(qkv4, (B, N, 3 * C))
            # Split to heads: returns [B, H, N, D] tensors
            q_tt, k_tt, v_tt = ttnn.transformer.split_query_key_value_and_split_heads(
                qkv3, num_heads=H, transpose_key=False
            )
            scale = 1.0 / math.sqrt(D)
            sdpa_kwargs = dict(is_causal=False, scale=scale)
            if "attn_mask" in mm_opts and mm_opts["attn_mask"] is not None:
                sdpa_kwargs["attn_mask"] = mm_opts["attn_mask"]
            pc = getattr(self, "program_config", None)
            if pc is not None:
                try:
                    # Allow passing a TTLayerConfig; compute per-seq program config
                    if hasattr(pc, "sdpa_program_config"):
                        maybe_pc = pc.sdpa_program_config(N)
                    else:
                        maybe_pc = pc
                    if maybe_pc is not None:
                        sdpa_kwargs["program_config"] = maybe_pc
                except Exception:
                    pass
            ctx_tt = ttnn.transformer.scaled_dot_product_attention(q_tt, k_tt, v_tt, **sdpa_kwargs)
            # Merge heads back to [B, N, C]
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

        # Output projection on device when possible.
        out: ttnn.Tensor
        if self._proj_w_tt is not None and self._proj_b_tt is not None:
            # Ensure TT and 4D shape [B, 1, N, C] for ttnn.linear
            ctx_tt4 = ctx if len(getattr(ctx, "shape", [])) == 4 else ttnn.reshape(ctx, (B, 1, N, C))
            cfg = getattr(self, "program_config", None)
            memcfg = getattr(cfg, "proj_memcfg", None) if cfg is not None else None
            if memcfg is None:
                memcfg = getattr(self, "output_mem", None) or ttnn.DRAM_MEMORY_CONFIG
            proj_pc = getattr(cfg, "proj_program_config", None) if cfg is not None else None
            out_tt4 = _ttnn_linear_with_optional_program_config(
                x=ctx_tt4,
                w=self._proj_w_tt,
                bias=self._proj_b_tt,
                dtype=ttnn.bfloat16,
                memory_config=memcfg,
                program_config=proj_pc,
            )
            # Back to [B, N, C]
            out = out_tt4
            if len(getattr(out_tt4, "shape", [])) == 4:
                out = ttnn.reshape(out_tt4, (B, N, C))
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
        # Preferred device path
        try:
            if self.w1_tt is None or self.w2_tt is None:
                raise RuntimeError("Device MLP weights unavailable")
            shape = tuple(getattr(x, "shape", ()))
            if len(shape) == 3:
                B, N, C = shape
                x4 = ttnn.reshape(x, (B, 1, N, C))
            elif len(shape) == 4:
                x4 = x
                B, _, N, C = shape
            else:
                raise ValueError(f"TTMLP expects 3D/4D TT tensor, got {shape}")
            cfg = getattr(self, "program_config", None)
            memcfg = getattr(cfg, "mlp_memcfg", None) if cfg is not None else None
            if memcfg is None:
                memcfg = getattr(self, "output_mem", None) or ttnn.DRAM_MEMORY_CONFIG
            ff1_pc = getattr(cfg, "ff1_program_config", None) if cfg is not None else None
            ff2_pc = getattr(cfg, "ff2_program_config", None) if cfg is not None else None
            y1 = _ttnn_linear_with_optional_program_config(
                x=x4,
                w=self.w1_tt,
                bias=self.b1_tt,
                dtype=ttnn.bfloat16,
                memory_config=memcfg,
                program_config=ff1_pc,
            )
            # If FF1 program config fuses GELU, skip the explicit GELU op.
            if not _program_config_fuses_activation(ff1_pc):
                y1 = ttnn.gelu(y1)
            y2 = _ttnn_linear_with_optional_program_config(
                x=y1,
                w=self.w2_tt,
                bias=self.b2_tt,
                dtype=ttnn.bfloat16,
                memory_config=memcfg,
                program_config=ff2_pc,
            )
            return y2 if len(shape) == 4 else ttnn.reshape(y2, (B, N, self.w2_tt.shape[-1]))
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
        h = self.ln1(x)
        h = self.attn(h, **mm_opts)
        x = ttnn.add(x, h)
        h = self.ln2(x)
        h = self.mlp(h, **mm_opts)
        x = ttnn.add(x, h)
        return x
