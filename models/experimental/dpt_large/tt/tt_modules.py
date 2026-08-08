# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Lightweight TTNN helper modules for DPT-Large.

These intentionally mirror the Hugging Face ViT pieces but use TTNN /
device ops for execution when a TT device is available. They keep
dependencies minimal so we can iterate on sharding and fused-op choices
in `tt_configs.py`.
"""

from __future__ import annotations

from typing import Optional

import math
import time
import torch

from .tt_cnn_ops import TTConv2dCached
from .perf_counters import (
    inc_attn_island_reshard,
    inc_ln_island_interleave,
    inc_ln_island_reshard,
    inc_program_config_fallback,
    strict_program_config_enabled,
)

try:
    import ttnn  # type: ignore
except (ImportError, OSError):  # pragma: no cover
    ttnn = None  # type: ignore

try:
    from models.common.utility_functions import torch_to_tt_tensor_rm, torch2tt_tensor  # type: ignore
except ImportError:  # pragma: no cover
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


def build_attn_padding_key_mask_4d(
    padded_seq_len: int, valid_seq_len: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Build an additive key-padding mask that blocks padded key/value tokens.

    Returns a tensor shaped [1, 1, 1, S] with zeros for valid keys and -inf for
    padded keys (columns >= valid_seq_len). This is broadcastable over the query
    sequence dimension and significantly reduces mask memory pressure compared to
    a full [1, 1, S, S] mask.
    """
    s = int(padded_seq_len)
    v = int(valid_seq_len)
    if v < 0 or s < 0:
        raise ValueError(f"Invalid lengths: padded_seq_len={s}, valid_seq_len={v}")
    if v >= s or s == 0:
        return torch.zeros((1, 1, 1, s), dtype=dtype)
    mask = torch.zeros((1, 1, 1, s), dtype=dtype)
    mask[..., v:] = float("-inf")
    return mask


def _ttnn_linear_with_optional_program_config(
    *, x, w, bias, dtype, memory_config, program_config, op_name: str = "unknown"
):
    """Call ttnn.linear with best-effort program_config plumbing across runtime versions.

    In perf runs we want to surface when a program_config cannot be used, since
    silently falling back to the default kernel changes both performance and
    determinism.
    """
    if program_config is None:
        return ttnn.linear(x, w, bias=bias, dtype=dtype, memory_config=memory_config)
    try:
        return ttnn.linear(x, w, bias=bias, dtype=dtype, memory_config=memory_config, program_config=program_config)
    except TypeError:
        # Older ttnn builds may not expose program_config for this op.
        inc_program_config_fallback(op=op_name, reason="kwarg_unsupported")
        if strict_program_config_enabled():
            raise
        return ttnn.linear(x, w, bias=bias, dtype=dtype, memory_config=memory_config)
    except (RuntimeError, ValueError):
        # Some runtimes accept the kwarg but can reject specific program configs at runtime.
        inc_program_config_fallback(op=op_name, reason="runtime_rejected")
        if strict_program_config_enabled():
            raise
        return ttnn.linear(x, w, bias=bias, dtype=dtype, memory_config=memory_config)


def _program_config_fuses_activation(program_config) -> bool:
    if program_config is None:
        return False
    return getattr(program_config, "fused_activation", None) is not None


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
    if program_config is None:
        return ttnn.matmul(a, b, dtype=dtype, memory_config=memory_config)

    try:
        return ttnn.matmul(a, b, dtype=dtype, memory_config=memory_config, program_config=program_config)
    except TypeError:
        inc_program_config_fallback(op=op_name, reason="kwarg_unsupported")
        if strict_program_config_enabled():
            raise
        return ttnn.matmul(a, b, dtype=dtype, memory_config=memory_config)
    except (RuntimeError, ValueError):
        inc_program_config_fallback(op=op_name, reason="runtime_rejected")
        if strict_program_config_enabled():
            raise
        return ttnn.matmul(a, b, dtype=dtype, memory_config=memory_config)


def _ttnn_attention_softmax_with_optional_program_config(
    *, x: ttnn.Tensor, attention_mask, head_size: int, program_config, op_name: str = "unknown", memory_config=None
):
    kwargs = dict(attention_mask=attention_mask, head_size=int(head_size))
    if memory_config is not None:
        kwargs["memory_config"] = memory_config

    if program_config is None:
        return ttnn.transformer.attention_softmax_(x, **kwargs)

    try:
        return ttnn.transformer.attention_softmax_(x, program_config=program_config, **kwargs)
    except TypeError:
        inc_program_config_fallback(op=op_name, reason="kwarg_unsupported")
        if strict_program_config_enabled():
            raise
        return ttnn.transformer.attention_softmax_(x, **kwargs)
    except (RuntimeError, ValueError):
        inc_program_config_fallback(op=op_name, reason="runtime_rejected")
        if strict_program_config_enabled():
            raise
        return ttnn.transformer.attention_softmax_(x, **kwargs)


def _ttnn_is_sharded(x) -> bool:
    try:
        if hasattr(ttnn, "is_sharded"):
            return bool(ttnn.is_sharded(x))
    except (TypeError, ValueError, RuntimeError):
        pass
    try:
        return bool(x.is_sharded())
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return False


def _ttnn_get_memory_config_or_none(x):
    if ttnn is None or not hasattr(ttnn, "get_memory_config"):
        return None
    try:
        return ttnn.get_memory_config(x)
    except (TypeError, ValueError, RuntimeError):
        return None


def _ttnn_to_memory_config(x, *, memory_config, dtype=None):
    if dtype is None:
        return ttnn.to_memory_config(x, memory_config=memory_config)
    try:
        return ttnn.to_memory_config(x, memory_config=memory_config, dtype=dtype)
    except TypeError:
        return ttnn.to_memory_config(x, memory_config=memory_config)


def _ttnn_deallocate(*tensors) -> None:
    if not hasattr(ttnn, "deallocate"):
        return
    for t in tensors:
        if t is None:
            continue
        try:
            ttnn.deallocate(t)
        except (TypeError, ValueError, RuntimeError):
            pass


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
            prog = (
                getattr(pc_obj, "ff1_program_config", None)
                if self.activation
                else getattr(pc_obj, "ff2_program_config", None)
            )

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
    ):
        self.device = device
        self.tt_conv = TTConv2dCached.from_tensors(
            weight_torch=conv_weight,
            bias_torch=conv_bias,
            stride=(int(stride), int(stride)),
            padding=(int(padding), int(padding)),
            dilation=(1, 1),
            groups=1,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.tt_conv(x, device=self.device)


class TTLayerNorm:
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        device,
        output_mem: Optional[ttnn.MemoryConfig] = None,
        program_config=None,
    ):
        self.eps = float(eps)
        self.device = device
        self.output_mem = output_mem
        self.program_config = program_config
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

    def _try_dram_island_layer_norm(
        self,
        x: ttnn.Tensor,
        *,
        input_memory_config,
    ) -> ttnn.Tensor | None:
        x_ilv = None
        y_ilv = None
        try:
            t0 = time.perf_counter()
            x_ilv = _ttnn_to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
            inc_ln_island_interleave((time.perf_counter() - t0) * 1000.0)

            y_ilv = ttnn.layer_norm(
                x_ilv,
                weight=self.weight_tt,
                bias=self.bias_tt,
                epsilon=self.eps,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if input_memory_config is not None:
                t1 = time.perf_counter()
                y = _ttnn_to_memory_config(y_ilv, memory_config=input_memory_config, dtype=ttnn.bfloat16)
                inc_ln_island_reshard((time.perf_counter() - t1) * 1000.0)
                _ttnn_deallocate(x_ilv, y_ilv)
                return y
            return y_ilv
        except (RuntimeError, TypeError, ValueError):
            _ttnn_deallocate(x_ilv, y_ilv)
            return None

    def _try_chunked_sharded_layer_norm(
        self,
        x: ttnn.Tensor,
        *,
        batch: int,
        seq_len: int,
        hidden: int,
        grid_x: int,
        grid_y: int,
    ) -> ttnn.Tensor | None:
        # Chunk to at most 8 tiles (256 tokens) per LN call to reduce static CB
        # footprint and avoid CB-vs-L1 clashes under trace capture on N300.
        tile = 32
        max_chunk_tokens = 8 * tile
        if seq_len <= max_chunk_tokens or (seq_len % tile) != 0:
            return None

        chunks: list[ttnn.Tensor] = []
        try:
            core_grid = ttnn.CoreGrid(y=int(grid_y), x=int(grid_x))
            out_mc_full = _ttnn_get_memory_config_or_none(x)
            chunk_orient = (
                ttnn.ShardOrientation.COL_MAJOR
                if hasattr(ttnn.ShardOrientation, "COL_MAJOR")
                else ttnn.ShardOrientation.ROW_MAJOR
            )

            start = 0
            while start < seq_len:
                end = min(seq_len, start + max_chunk_tokens)
                # Enforce tile-aligned slicing for TILE layout sharded tensors.
                end = int(((end + tile - 1) // tile) * tile)
                end = min(end, seq_len)
                if end <= start:
                    break

                chunk_len = int(end - start)
                chunk_mc = ttnn.create_sharded_memory_config(
                    (int(batch), int(chunk_len), int(hidden)),
                    core_grid=core_grid,
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=chunk_orient,
                )
                ln_pc_chunk = None
                if hasattr(ttnn, "LayerNormShardedMultiCoreProgramConfig"):
                    block_h_tiles = int(chunk_len) // int(tile)
                    block_w_tiles = max(1, (int(hidden) // int(tile)) // max(1, int(grid_x)))
                    ln_pc_chunk = ttnn.LayerNormShardedMultiCoreProgramConfig(
                        compute_with_storage_grid_size=(int(grid_x), int(grid_y)),
                        subblock_w=int(block_w_tiles),
                        block_h=int(block_h_tiles),
                        block_w=int(block_w_tiles),
                        inplace=False,
                        legacy_reduction=False,
                        legacy_rsqrt=False,
                        use_welford=True,
                    )

                x_chunk = ttnn.slice(
                    x,
                    [0, int(start), 0],
                    [int(batch), int(end), int(hidden)],
                    memory_config=chunk_mc,
                )
                y_chunk = ttnn.layer_norm(
                    x_chunk,
                    weight=self.weight_tt,
                    bias=self.bias_tt,
                    epsilon=self.eps,
                    memory_config=chunk_mc,
                    program_config=ln_pc_chunk,
                )
                chunks.append(y_chunk)
                start = end

            if len(chunks) < 2:
                return None
            return ttnn.concat(chunks, dim=1, memory_config=out_mc_full)
        except (RuntimeError, TypeError, ValueError):
            return None
        finally:
            _ttnn_deallocate(*chunks)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        pc = self.program_config
        ln_pc = getattr(pc, "ln_program_config", None) if pc is not None else None
        cc = getattr(pc, "ln_compute_config", None) if pc is not None else None
        grid = getattr(pc, "grid", None) if pc is not None else None

        x_is_sharded = _ttnn_is_sharded(x)
        x_shape = tuple(getattr(x, "shape", ()))
        batch = 1
        if len(x_shape) >= 1:
            try:
                batch = max(1, int(x_shape[0]))
            except (TypeError, ValueError):
                batch = 1

        # Stage-3 DP batching (dp=2, batch_size=4): when batch == grid_y, block-sharded
        # LayerNorm sees per-core height == seq_len tiles (e.g., 640 tokens -> 20 tiles),
        # which can exceed static-CB limits under trace capture on N300. LayerNorm is
        # token-wise (normalizes over the last dim only), so we can safely chunk the
        # sequence dimension and run multiple smaller LayerNorms without changing results.
        if x_is_sharded and grid is not None and len(x_shape) == 3 and int(batch) > 1 and int(grid[1]) == int(batch):
            input_mc = _ttnn_get_memory_config_or_none(x)
            y = self._try_dram_island_layer_norm(x, input_memory_config=input_mc)
            if y is not None:
                return y
            if all(
                hasattr(ttnn, name)
                for name in (
                    "slice",
                    "concat",
                    "create_sharded_memory_config",
                    "CoreGrid",
                    "ShardStrategy",
                    "ShardOrientation",
                )
            ):
                _, seq_len, hidden = (int(x_shape[0]), int(x_shape[1]), int(x_shape[2]))
                y = self._try_chunked_sharded_layer_norm(
                    x,
                    batch=int(batch),
                    seq_len=int(seq_len),
                    hidden=int(hidden),
                    grid_x=int(grid[0]),
                    grid_y=int(grid[1]),
                )
                if y is not None:
                    return y

        kwargs: dict[str, object] = {
            "weight": self.weight_tt,
            "bias": self.bias_tt,
            "epsilon": self.eps,
        }
        if x_is_sharded:
            input_mc = _ttnn_get_memory_config_or_none(x)
            if input_mc is not None:
                kwargs["memory_config"] = input_mc
            if ln_pc is not None:
                kwargs["program_config"] = ln_pc
        elif self.output_mem is not None:
            kwargs["memory_config"] = self.output_mem
        if cc is not None:
            kwargs["compute_kernel_config"] = cc

        try:
            return ttnn.layer_norm(x, **kwargs)
        except TypeError:
            # Backward compat for older runtimes without program_config / compute_kernel_config kwargs.
            if "program_config" in kwargs or "compute_kernel_config" in kwargs:
                inc_program_config_fallback(op="layer_norm", reason="kwarg_unsupported")
                if strict_program_config_enabled():
                    raise
            kwargs.pop("program_config", None)
            kwargs.pop("compute_kernel_config", None)
            return ttnn.layer_norm(x, **kwargs)
        except (RuntimeError, ValueError):
            # Some runtimes accept these kwargs but can fail for particular inputs/configs.
            # Retry once without the perf configs to avoid hard-failing the entire pipeline.
            if "program_config" in kwargs or "compute_kernel_config" in kwargs:
                inc_program_config_fallback(op="layer_norm", reason="runtime_rejected")
                if strict_program_config_enabled():
                    raise
            kwargs.pop("program_config", None)
            kwargs.pop("compute_kernel_config", None)
            return ttnn.layer_norm(x, **kwargs)


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
    ):
        # Prepare fused QKV weights for device-side linear.
        # Torch linear weights are [out, in]; ttnn.linear expects [in, out] in TILE.
        H = int(num_heads)
        D = int(head_dim)

        qkv_w_dtype = ttnn.bfloat8_b if hasattr(ttnn, "bfloat8_b") else ttnn.bfloat16
        qkv_b_dtype = ttnn.bfloat8_b if hasattr(ttnn, "bfloat8_b") else ttnn.bfloat16

        # Interleaved/SDPA attention path expects QKV packed as [Q, K, V] stacked along the output dimension.
        qkv_w_stacked = torch.cat([q_w.detach(), k_w.detach(), v_w.detach()], dim=0).contiguous()  # [3C, C]
        wqkv_stacked_t = torch.transpose(qkv_w_stacked, -1, -2).contiguous()  # [C, 3C]
        bqkv_stacked = torch.cat([q_b.detach(), k_b.detach(), v_b.detach()], dim=0).contiguous()  # [3C]
        self._wqkv_stacked_tt = ttnn.from_torch(
            wqkv_stacked_t,
            dtype=qkv_w_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self._bqkv_stacked_tt = ttnn.from_torch(
            bqkv_stacked.reshape(1, -1),
            dtype=qkv_b_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Sharded split-heads attention (vit.md) expects head-interleaved packing:
        # [Q_head0, K_head0, V_head0, Q_head1, K_head1, V_head1, ...].
        q_w_hdi = q_w.detach().reshape(H, D, -1)
        k_w_hdi = k_w.detach().reshape(H, D, -1)
        v_w_hdi = v_w.detach().reshape(H, D, -1)
        qkv_w_head_interleaved = torch.cat([q_w_hdi, k_w_hdi, v_w_hdi], dim=1).reshape(H * 3 * D, -1)
        wqkv_head_interleaved_t = torch.transpose(qkv_w_head_interleaved, -1, -2).contiguous()  # [C, 3C]

        q_b_hd = q_b.detach().reshape(H, D)
        k_b_hd = k_b.detach().reshape(H, D)
        v_b_hd = v_b.detach().reshape(H, D)
        bqkv_head_interleaved = torch.cat([q_b_hd, k_b_hd, v_b_hd], dim=1).reshape(H * 3 * D).contiguous()
        self._wqkv_head_interleaved_tt = ttnn.from_torch(
            wqkv_head_interleaved_t,
            dtype=qkv_w_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self._bqkv_head_interleaved_tt = ttnn.from_torch(
            bqkv_head_interleaved.reshape(1, -1),
            dtype=qkv_b_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.device = device
        # Persist knobs so callers can tune memory/program behavior.
        self.output_mem = output_mem
        self.program_config = program_config
        # Cache height-sharded memory configs for [B, N] attention shapes.
        self._height_shard_mc_cache: dict[tuple[int, int, int, int], dict[str, object]] = {}
        # Cache block-sharded QKV output specs keyed by (B, N, C, grid_x, grid_y).
        self._qkv_block_shard_mc_cache: dict[tuple[int, int, int, int, int], object] = {}
        # Pre-create TTNN projection weights for device-side output matmul
        # Use transposed weights to avoid transpose_b flag during linear.
        w_t = torch.transpose(proj_w.detach(), -1, -2).contiguous()
        self._proj_w_tt = ttnn.from_torch(
            w_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self._proj_b_tt = ttnn.from_torch(
            proj_b.detach(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

    def __call__(self, x: ttnn.Tensor, **mm_opts) -> ttnn.Tensor:
        """
        Multi-head self-attention using TTNN transformer primitives.

        Contract:
          - Input:  TT tensor [B, N, C] (tokens)
          - Output: TT tensor [B, N, C]

        Stage-2/3 perf path follows the ViT TTNN reference (vit.md):
          - fused QKV linear on block-sharded tokens
          - split heads (Q,V: [B,H,N,D], K: [B,H,D,N])
          - sharded QK matmul -> attention_softmax_ -> sharded AV matmul
          - concatenate_heads back to block-sharded tokens
        """
        x3, B, N, C, input_is_4d = self._normalize_tokens_input(x)
        H = self.num_heads
        D = self.head_dim
        if C != H * D:
            raise ValueError(f"TTAttention hidden dim mismatch: C={C}, H*D={H * D}")
        # Stage-2/3: tokens may be block-sharded across the encoder.
        tokens_shard_mc = mm_opts.get("tokens_shard_mc", None)
        cfg = getattr(self, "program_config", None)
        input_is_sharded = tokens_shard_mc is not None and (_ttnn_is_sharded(x) or _ttnn_is_sharded(x3))
        use_default_attn_programs = (
            bool(getattr(cfg, "use_default_attention_programs", False)) if cfg is not None else False
        )
        # Prefer explicit attention on sharded operands; fall back to the SDPA island
        # only when not sharded.
        explicit_sharded_attn = input_is_sharded
        try:
            q_tt, k_tt, v_tt, split_memcfg, memcfg = self._compute_qkv_and_split_heads(
                x=x,
                x3=x3,
                B=B,
                N=N,
                C=C,
                H=H,
                explicit_sharded_attn=explicit_sharded_attn,
                cfg=cfg,
            )

            if explicit_sharded_attn:
                ctx = self._run_explicit_attention(
                    q_tt=q_tt,
                    k_tt=k_tt,
                    v_tt=v_tt,
                    B=B,
                    N=N,
                    H=H,
                    D=D,
                    split_memcfg=split_memcfg,
                    tokens_shard_mc=tokens_shard_mc,
                    use_default_attn_programs=use_default_attn_programs,
                    cfg=cfg,
                    mm_opts=mm_opts,
                )
            else:
                ctx = self._run_sdpa_attention(
                    q_tt=q_tt,
                    k_tt=k_tt,
                    v_tt=v_tt,
                    N=N,
                    D=D,
                    memcfg=memcfg,
                    cfg=cfg,
                    mm_opts=mm_opts,
                )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError("TTAttention device path failed") from exc

        out3 = self._run_output_projection(
            ctx,
            B=B,
            N=N,
            C=C,
            cfg=cfg,
            input_is_sharded=input_is_sharded,
            tokens_shard_mc=tokens_shard_mc,
            mm_opts=mm_opts,
        )
        return out3 if not input_is_4d else ttnn.reshape(out3, (int(B), 1, int(N), int(C)))

    def _normalize_tokens_input(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, int, int, int, bool]:
        # Stay on device: accept TT tensor [B, N, C] or [B, 1, N, C] (TILE or ROW_MAJOR).
        if not hasattr(x, "shape"):
            raise ValueError("TTAttention expects a TT tensor input")
        shape = tuple(getattr(x, "shape", ()))
        if len(shape) == 4:
            B, _, N, C = shape
            x3 = ttnn.reshape(x, (int(B), int(N), int(C)))
            return x3, int(B), int(N), int(C), True
        if len(shape) == 3:
            B, N, C = shape
            return x, int(B), int(N), int(C), False
        raise ValueError(f"TTAttention expects TT tensor with 3D/4D shape, got {shape}")

    def _compute_qkv_and_split_heads(
        self,
        *,
        x: ttnn.Tensor,
        x3: ttnn.Tensor,
        B: int,
        N: int,
        C: int,
        H: int,
        explicit_sharded_attn: bool,
        cfg,
    ):
        wqkv_tt = self._wqkv_head_interleaved_tt if explicit_sharded_attn else self._wqkv_stacked_tt
        bqkv_tt = self._bqkv_head_interleaved_tt if explicit_sharded_attn else self._bqkv_stacked_tt
        if wqkv_tt is None or bqkv_tt is None:
            raise RuntimeError("QKV fused weights not available on device")

        split_memcfg = getattr(cfg, "split_heads_memcfg", None) if cfg is not None else None
        if split_memcfg is None:
            split_memcfg = getattr(ttnn, "L1_HEIGHT_SHARDED_MEMORY_CONFIG", None)

        # QKV fused linear must be block-sharded for TTNN split-heads to take the fully sharded path (vit.md pattern).
        memcfg = getattr(cfg, "qkv_memcfg", None) if cfg is not None else None
        if memcfg is None:
            memcfg = getattr(self, "output_mem", None) or ttnn.DRAM_MEMORY_CONFIG

        qkv_block_shard_mc = None
        qkv_block_shard_grid_y = None
        if explicit_sharded_attn and hasattr(ttnn, "create_sharded_memory_config"):
            try:
                grid = getattr(cfg, "grid", None) if cfg is not None else None
                if grid is not None:
                    grid_x, grid_y = int(grid[0]), int(grid[1])
                    qkv_block_shard_grid_y = int(grid_y)
                    # Sharded split-heads requires batch size to equal the sharding grid's y-dimension.
                    # Only build the block-sharded QKV spec when that constraint is satisfied.
                    if int(B) == int(grid_y) and grid_x > 0 and grid_y > 0:
                        cache_key = (int(B), int(N), int(3 * C), int(grid_x), int(grid_y))
                        qkv_block_shard_mc = self._qkv_block_shard_mc_cache.get(cache_key)
                        if qkv_block_shard_mc is None:
                            core_grid = ttnn.CoreGrid(y=int(grid_y), x=int(grid_x))
                            qkv_block_shard_mc = ttnn.create_sharded_memory_config(
                                (int(B), int(N), int(3 * C)),
                                core_grid=core_grid,
                                strategy=ttnn.ShardStrategy.BLOCK,
                                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                            )
                            self._qkv_block_shard_mc_cache[cache_key] = qkv_block_shard_mc
            except (RuntimeError, TypeError, ValueError):
                qkv_block_shard_mc = None

        qkv_pc = getattr(cfg, "qkv_program_config", None) if cfg is not None else None
        if qkv_pc is not None and not (_ttnn_is_sharded(x) or _ttnn_is_sharded(x3)):
            qkv_pc = None

        qkv_dtype = ttnn.bfloat16
        if explicit_sharded_attn and hasattr(ttnn, "bfloat8_b"):
            qkv_dtype = ttnn.bfloat8_b

        qkv3 = _ttnn_linear_with_optional_program_config(
            x=x3,
            w=wqkv_tt,
            bias=bqkv_tt,
            dtype=qkv_dtype,
            memory_config=memcfg,
            program_config=qkv_pc,
            op_name="attn_qkv",
        )

        # DPT-Large on N300: sharded split-heads can TT_FATAL for batch>1 with 16 heads due to core-grid constraints.
        # For dp batched runs, force QKV interleaved before split-heads, then explicitly reshard Q/K/V for attention.
        if explicit_sharded_attn and int(B) > 1 and _ttnn_is_sharded(qkv3):
            qkv3 = _ttnn_to_memory_config(qkv3, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=qkv_dtype)

        # If the runtime cannot emit a sharded QKV tensor directly, reshard once here so split-heads can take the sharded path.
        if (
            explicit_sharded_attn
            and qkv_block_shard_mc is not None
            and qkv_block_shard_grid_y is not None
            and int(qkv_block_shard_grid_y) == int(B)
            and int(B) <= 1
            and not _ttnn_is_sharded(qkv3)
        ):
            qkv3 = _ttnn_to_memory_config(qkv3, memory_config=qkv_block_shard_mc, dtype=qkv_dtype)

        # `split_query_key_value_and_split_heads` sharded output path assumes the input tensor is already sharded.
        # If QKV is interleaved, request interleaved split-heads and reshard explicitly later for attention.
        if split_memcfg is not None and not _ttnn_is_sharded(qkv3):
            split_memcfg = None

        # Prefer the more conservative interleaved split-heads output for dp-batched runs (B>1).
        if explicit_sharded_attn and int(B) > 1:
            split_memcfg = None

        # Split to heads (vit.md pattern): returns [B, H, N, D] (Q,V) and [B, H, D, N] (K) by default.
        try:
            q_tt, k_tt, v_tt = ttnn.transformer.split_query_key_value_and_split_heads(
                qkv3, memory_config=split_memcfg, num_heads=H, transpose_key=explicit_sharded_attn
            )
        except TypeError:
            try:
                q_tt, k_tt, v_tt = ttnn.transformer.split_query_key_value_and_split_heads(
                    qkv3, memory_config=split_memcfg, num_heads=H
                )
            except TypeError:
                try:
                    q_tt, k_tt, v_tt = ttnn.transformer.split_query_key_value_and_split_heads(
                        qkv3, num_heads=H, transpose_key=explicit_sharded_attn
                    )
                except TypeError:
                    # Backward compat: older runtimes may not expose `memory_config`/`transpose_key`.
                    q_tt, k_tt, v_tt = ttnn.transformer.split_query_key_value_and_split_heads(qkv3, num_heads=H)

        return q_tt, k_tt, v_tt, split_memcfg, memcfg

    def _run_explicit_attention(
        self,
        *,
        q_tt: ttnn.Tensor,
        k_tt: ttnn.Tensor,
        v_tt: ttnn.Tensor,
        B: int,
        N: int,
        H: int,
        D: int,
        split_memcfg,
        tokens_shard_mc,
        use_default_attn_programs: bool,
        cfg,
        mm_opts,
    ) -> ttnn.Tensor:
        # Explicit attention path (ViT TTNN reference): QK -> fused scale+mask+softmax -> AV,
        # then concatenate heads back to block-sharded tokens for the output projection.
        qk_pc = None if use_default_attn_programs else getattr(cfg, "qk_program_config", None)
        softmax_pc = None if use_default_attn_programs else getattr(cfg, "softmax_program_config", None)
        av_pc = None if use_default_attn_programs else getattr(cfg, "av_program_config", None)

        attn_mm_dtype = ttnn.bfloat16
        # For DPT-Large padded sequence lengths (e.g., 640), attention score tensors are
        # extremely large in L1 when kept in BF16 (per-core shards can approach ~800KB).
        # Using BF8 reduces the score/probability footprint and avoids static-CB/L1 clashes
        # during trace capture on N300.
        if hasattr(ttnn, "bfloat8_b") and int(N) >= 512:
            attn_mm_dtype = ttnn.bfloat8_b
        scores_memcfg = split_memcfg
        av_memcfg = split_memcfg
        try:
            attn_grid = getattr(cfg, "attn_grid", None) if cfg is not None else None
            # Only reshard when split-heads returned interleaved Q/K/V. When split-heads
            # already emits sharded tensors, keep their layout to preserve compatibility
            # with concatenate_heads and the ViT reference flow.
            if attn_grid is not None and hasattr(ttnn, "create_sharded_memory_config") and not _ttnn_is_sharded(q_tt):
                attn_grid_x, attn_grid_y = int(attn_grid[0]), int(attn_grid[1])
                if attn_grid_x > 0 and attn_grid_y > 0:
                    attn_core_grid = ttnn.CoreGrid(y=int(attn_grid_y), x=int(attn_grid_x))
                    scores_memcfg = ttnn.create_sharded_memory_config(
                        (int(B), int(H), int(N), int(N)),
                        core_grid=attn_core_grid,
                        strategy=ttnn.ShardStrategy.HEIGHT,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    )
                    if int(B) <= 1:
                        q_mc = ttnn.create_sharded_memory_config(
                            getattr(q_tt, "padded_shape", None) or getattr(q_tt, "shape", None),
                            core_grid=attn_core_grid,
                            strategy=ttnn.ShardStrategy.HEIGHT,
                            orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        )
                        k_mc = ttnn.create_sharded_memory_config(
                            getattr(k_tt, "padded_shape", None) or getattr(k_tt, "shape", None),
                            core_grid=attn_core_grid,
                            strategy=ttnn.ShardStrategy.HEIGHT,
                            orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        )
                        v_mc = ttnn.create_sharded_memory_config(
                            getattr(v_tt, "padded_shape", None) or getattr(v_tt, "shape", None),
                            core_grid=attn_core_grid,
                            strategy=ttnn.ShardStrategy.HEIGHT,
                            orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        )
                        q_tt = ttnn.to_memory_config(q_tt, memory_config=q_mc)
                        k_tt = ttnn.to_memory_config(k_tt, memory_config=k_mc)
                        v_tt = ttnn.to_memory_config(v_tt, memory_config=v_mc)
                        # Keep AV output interleaved for robustness: `concatenate_heads` on
                        # height-sharded ctx_tt can produce invalid shard specs on some runtimes.
                        # The attention output is re-sharded back to encoder token sharding after
                        # concatenation when `tokens_shard_mc` is provided.
                        av_memcfg = None
                    else:
                        # dp-batched runs (B>1): keep Q/K/V interleaved (avoid split-heads + sharding
                        # layout constraints) and only request a sharded scores/probs buffer. Let AV
                        # output interleave to avoid width-vs-grid_x shard constraints on D=64.
                        av_memcfg = None
        except (RuntimeError, TypeError, ValueError):
            if strict_program_config_enabled():
                raise
            scores_memcfg = split_memcfg
            av_memcfg = split_memcfg

        attn_scores = _ttnn_matmul_with_optional_program_config(
            a=q_tt,
            b=k_tt,
            dtype=attn_mm_dtype,
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
            memory_config=scores_memcfg,
            op_name="attn_softmax",
        )
        ctx_tt = _ttnn_matmul_with_optional_program_config(
            a=attn_probs,
            b=v_tt,
            dtype=attn_mm_dtype,
            memory_config=av_memcfg,
            program_config=av_pc,
            op_name="attn_av_matmul",
        )

        # Release large attention intermediates before concatenation/reshard to reduce
        # L1 pressure and avoid static-CB vs L1 clashes during interleaved->sharded moves.
        _ttnn_deallocate(attn_scores, attn_probs, q_tt, k_tt, v_tt)

        merged = None
        # `concatenate_heads` defaults the output memory_config to the input tensor's
        # memory_config. When ctx_tt is height-sharded over head_dim (shard width == D),
        # that default can be incompatible with the concatenated hidden width (H*D).
        if merged is None:
            try:
                merged = ttnn.transformer.concatenate_heads(ctx_tt)
            except (RuntimeError, TypeError, ValueError):
                # Fallback: interleave ctx_tt before concatenation if the sharded layout is unsupported.
                ctx_for_concat = ctx_tt
                if _ttnn_is_sharded(ctx_tt):
                    ctx_for_concat = _ttnn_to_memory_config(ctx_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                merged = ttnn.transformer.concatenate_heads(ctx_for_concat)
                if ctx_for_concat is not ctx_tt:
                    _ttnn_deallocate(ctx_for_concat)
        _ttnn_deallocate(ctx_tt)

        if tokens_shard_mc is not None and not _ttnn_is_sharded(merged):
            shard_dtype = mm_opts.get("tokens_shard_dtype", ttnn.bfloat16)
            merged = _ttnn_to_memory_config(merged, memory_config=tokens_shard_mc, dtype=shard_dtype)

        return merged

    def _run_sdpa_attention(
        self,
        *,
        q_tt: ttnn.Tensor,
        k_tt: ttnn.Tensor,
        v_tt: ttnn.Tensor,
        N: int,
        D: int,
        memcfg,
        cfg,
        mm_opts,
    ) -> ttnn.Tensor:
        # Legacy SDPA path (interleaved). Keep it for non-sharded / debug.
        scale = 1.0 / math.sqrt(D)
        sdpa_kwargs = dict(is_causal=False, scale=scale)
        if mm_opts.get("attn_mask", None) is not None:
            sdpa_kwargs["attn_mask"] = mm_opts["attn_mask"]

        pc = cfg
        if pc is not None:
            try:
                if hasattr(pc, "sdpa_program_config"):
                    maybe_pc = pc.sdpa_program_config(N)
                else:
                    maybe_pc = pc
                if maybe_pc is not None:
                    sdpa_kwargs["program_config"] = maybe_pc
            except (RuntimeError, TypeError, ValueError):
                pass

        k_for_sdpa = k_tt
        # split-heads returns K as [B, H, D, N] (transposed for QK matmul). SDPA expects [B, H, N, D].
        try:
            k_shape = tuple(getattr(k_tt, "shape", ()))
            if len(k_shape) == 4 and int(k_shape[2]) == int(D) and int(k_shape[3]) == int(N):
                k_for_sdpa = ttnn.permute(k_tt, (0, 1, 3, 2))
        except (AttributeError, TypeError, ValueError):
            k_for_sdpa = k_tt

        ctx_tt = ttnn.transformer.scaled_dot_product_attention(q_tt, k_for_sdpa, v_tt, **sdpa_kwargs)
        if k_for_sdpa is not k_tt:
            _ttnn_deallocate(k_for_sdpa)
        try:
            merged = ttnn.transformer.concatenate_heads(ctx_tt, memory_config=memcfg)
        except TypeError:
            merged = ttnn.transformer.concatenate_heads(ctx_tt)
        return merged

    def _run_output_projection(
        self,
        ctx: ttnn.Tensor,
        *,
        B: int,
        N: int,
        C: int,
        cfg,
        input_is_sharded: bool,
        tokens_shard_mc,
        mm_opts,
    ) -> ttnn.Tensor:
        # Output projection on device.
        ctx3 = ctx if len(getattr(ctx, "shape", ())) == 3 else ttnn.reshape(ctx, (int(B), int(N), int(C)))
        memcfg = getattr(cfg, "proj_memcfg", None) if cfg is not None else None
        if memcfg is None:
            memcfg = getattr(self, "output_mem", None) or ttnn.DRAM_MEMORY_CONFIG
        reshard_after_proj = False
        # When SDPA is forced on sharded tokens (e.g., 512x512), ctx3 can be interleaved
        # and projection program configs may be unavailable. Some TTNN matmul variants can
        # ignore a provided sharded output memory_config and compute an invalid shard spec
        # on harvested grids. Keep projection output interleaved and reshard explicitly.
        if input_is_sharded and tokens_shard_mc is not None:
            if _ttnn_is_sharded(ctx3):
                if memcfg is not getattr(ttnn, "DRAM_MEMORY_CONFIG", None):
                    memcfg = tokens_shard_mc
            else:
                memcfg = getattr(ttnn, "DRAM_MEMORY_CONFIG", None) or memcfg
                reshard_after_proj = True
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
        if (reshard_after_proj or input_is_sharded) and tokens_shard_mc is not None and not _ttnn_is_sharded(out3):
            shard_dtype = mm_opts.get("tokens_shard_dtype", ttnn.bfloat16)
            t0 = time.perf_counter()
            out3 = _ttnn_to_memory_config(out3, memory_config=tokens_shard_mc, dtype=shard_dtype)
            inc_attn_island_reshard((time.perf_counter() - t0) * 1000.0)
        return out3


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
    ):
        self.device = device
        self.output_mem = output_mem
        self.program_config = program_config
        # Pre-upload FC1/FC2 weights/biases for device path.
        w1_t = torch.transpose(fc1_w.detach(), -1, -2).contiguous()
        w2_t = torch.transpose(fc2_w.detach(), -1, -2).contiguous()
        self.w1_tt = ttnn.from_torch(w1_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        # Bias in TILE layout keeps sharded linear bias adds on the fast path and avoids
        # layout asserts in some TTNN matmul/bias kernels under trace.
        self.b1_tt = ttnn.from_torch(
            fc1_b.detach().reshape(1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.w2_tt = ttnn.from_torch(w2_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.b2_tt = ttnn.from_torch(
            fc2_b.detach().reshape(1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        # Cache sharded memory configs for running MLP matmuls on a separate grid.
        self._mlp_block_shard_mc_cache: dict[tuple[int, ...], object] = {}

    def __call__(self, x: ttnn.Tensor, **mm_opts) -> ttnn.Tensor:
        shape = tuple(getattr(x, "shape", ()))
        try:
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
            if memcfg is None:
                memcfg = getattr(self, "output_mem", None) or ttnn.DRAM_MEMORY_CONFIG
            ff1_out_memcfg = getattr(cfg, "ff1_out_memcfg", None) if cfg is not None else None
            ff2_out_memcfg = getattr(cfg, "ff2_out_memcfg", None) if cfg is not None else None

            ff1_pc = getattr(cfg, "ff1_program_config", None) if cfg is not None else None
            ff2_pc = getattr(cfg, "ff2_program_config", None) if cfg is not None else None
            # When routing FF1 output to an interleaved buffer (e.g., DRAM) as a pressure-relief
            # workaround, prefer the runtime-chosen kernel for FC1 rather than a heavy sharded
            # program_config that can over-allocate static CBs under trace.
            if incoming_sharded and ff1_out_memcfg is not None:
                ff1_pc = None
            if incoming_sharded and ff2_out_memcfg is not None:
                ff2_pc = None
            if ff1_pc is not None and not _ttnn_is_sharded(x3):
                ff1_pc = None
            if ff2_pc is not None and not _ttnn_is_sharded(x3):
                ff2_pc = None

            # If MLP outputs are routed interleaved (e.g., DRAM pressure-relief mode), run the
            # MLP matmuls interleaved as well to avoid sharded->interleaved linear paths that can
            # hang under trace on N300. The final output is re-sharded to the encoder token spec.
            mlp_interleaved = incoming_sharded and (ff1_out_memcfg is not None or ff2_out_memcfg is not None)
            if mlp_interleaved:
                # When routing FF activations interleaved (e.g., DRAM pressure-relief), do not force
                # sharded program configs for those interleaved matmuls. Keep FF2 program configs
                # only when FF2 itself is intended to run sharded.
                ff1_pc = None
                if ff2_out_memcfg is not None:
                    ff2_pc = None

            tokens_shard_dtype = mm_opts.get("tokens_shard_dtype", ttnn.bfloat16)

            mlp_dtype = ttnn.bfloat16
            if incoming_sharded and hasattr(ttnn, "bfloat8_b"):
                # Stage-3: sharded-token runs are bandwidth/L1-pressure bound on N300.
                # Prefer BF8 activations for the MLP matmuls even when routing through
                # interleaved/DRAM buffers for trace stability; residual adds remain BF16.
                mlp_dtype = ttnn.bfloat8_b

            # Residual adds run in token shard dtype (BF16). Keep internal FF2 activations in
            # BF8 to reduce bandwidth/L1 pressure and cast back to BF16 before the residual add.
            ff2_dtype = mlp_dtype

            # Optionally reshard activations to a different MLP grid (e.g., 8x2) to reduce per-core M
            # and avoid static circular-buffer clashes on N300.
            mlp_grid = getattr(cfg, "mlp_core_grid", None) if cfg is not None else None
            did_reshard_for_mlp = False
            did_reshard_for_fc2 = False
            mlp_shard_mc = None
            y2_shard_mc = None
            if (
                incoming_sharded
                and (not mlp_interleaved)
                and mlp_grid is not None
                and hasattr(ttnn, "create_sharded_memory_config")
            ):
                try:
                    grid_x, grid_y = int(mlp_grid[0]), int(mlp_grid[1])
                    cache_key = (int(B), int(N), int(C), int(grid_x), int(grid_y))
                    mlp_shard_mc = self._mlp_block_shard_mc_cache.get(cache_key)
                    if mlp_shard_mc is None:
                        core_grid = ttnn.CoreGrid(y=int(grid_y), x=int(grid_x))
                        mlp_shard_mc = ttnn.create_sharded_memory_config(
                            getattr(x3, "padded_shape", (int(B), int(N), int(C))),
                            core_grid=core_grid,
                            strategy=ttnn.ShardStrategy.BLOCK,
                            orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        )
                        self._mlp_block_shard_mc_cache[cache_key] = mlp_shard_mc
                    # `to_memory_config` is implemented as a move op for many sharded tensors. The runtime
                    # expects the input buffer to have no other live views/aliases after the move.
                    # Drop the original Python alias so move-based reshard paths can consume buffers safely.
                    x = None
                    # `reshard` for block-sharded activations has been observed to hang under trace on N300.
                    # Use a conservative device-only path: materialize interleaved, then reshard.
                    if (
                        _ttnn_is_sharded(x3)
                        and hasattr(ttnn, "sharded_to_interleaved")
                        and hasattr(ttnn, "interleaved_to_sharded")
                    ):
                        x3_int = ttnn.sharded_to_interleaved(x3, ttnn.DRAM_MEMORY_CONFIG, output_dtype=mlp_dtype)
                        x3 = ttnn.interleaved_to_sharded(x3_int, mlp_shard_mc, output_dtype=mlp_dtype)
                        _ttnn_deallocate(x3_int)
                    else:
                        x3 = _ttnn_to_memory_config(x3, memory_config=mlp_shard_mc, dtype=mlp_dtype)
                    did_reshard_for_mlp = True
                except (RuntimeError, TypeError, ValueError):
                    did_reshard_for_mlp = False
            if incoming_sharded and (not mlp_interleaved) and mlp_grid is not None and not did_reshard_for_mlp:
                # Avoid using mismatched program configs if we couldn't reshard to the requested grid.
                ff1_pc = None
                ff2_pc = None
            if mlp_interleaved and _ttnn_is_sharded(x3):
                # Materialize the activation interleaved before FC1 when FF outputs are interleaved.
                # This avoids sharded->interleaved linear paths that can hang under trace.
                x3 = _ttnn_to_memory_config(x3, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=mlp_dtype)

            ff1_memcfg = ff1_out_memcfg or memcfg
            x4 = ttnn.reshape(x3, (int(B), 1, int(N), int(C)))
            y1 = _ttnn_linear_with_optional_program_config(
                x=x4,
                w=self.w1_tt,
                bias=self.b1_tt,
                dtype=mlp_dtype,
                memory_config=ff1_memcfg,
                program_config=ff1_pc,
                op_name="mlp_ff1",
            )
            # Trace capture can keep allocations alive longer; proactively release the input activation
            # once FC1 has consumed it to reduce L1/CB pressure (Stage-2 MLP crash unblocker).
            x4 = None
            if incoming_sharded:
                _ttnn_deallocate(x3)
            if not _program_config_fuses_activation(ff1_pc):
                y1_gelu = ttnn.gelu(y1)
                if incoming_sharded:
                    _ttnn_deallocate(y1)
                y1 = y1_gelu

            # If FF1 output is forced interleaved (e.g., DRAM) as a pressure-relief workaround,
            # reshard it back before FF2 when FC2 is intended to run sharded. Build the shard spec
            # for y1's expanded hidden dim (4*C); do not reuse the [B,N,C] spec.
            if (
                incoming_sharded
                and ff1_out_memcfg is not None
                and ff2_out_memcfg is None
                and not _ttnn_is_sharded(y1)
                and hasattr(ttnn, "create_sharded_memory_config")
            ):
                try:
                    y1_grid = mlp_grid or (getattr(cfg, "grid", None) if cfg is not None else None)
                    if y1_grid is None:
                        raise RuntimeError("No grid configured for sharding FF2 input")
                    grid_x, grid_y = int(y1_grid[0]), int(y1_grid[1])
                    y1_C = int(getattr(y1, "shape")[-1])
                    cache_key_y1 = (int(B), int(N), int(y1_C), int(grid_x), int(grid_y))
                    y1_shard_mc = self._mlp_block_shard_mc_cache.get(cache_key_y1)
                    if y1_shard_mc is None:
                        core_grid = ttnn.CoreGrid(y=int(grid_y), x=int(grid_x))
                        y1_shard_mc = ttnn.create_sharded_memory_config(
                            getattr(y1, "padded_shape", (int(B), 1, int(N), int(y1_C))),
                            core_grid=core_grid,
                            strategy=ttnn.ShardStrategy.BLOCK,
                            orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        )
                        self._mlp_block_shard_mc_cache[cache_key_y1] = y1_shard_mc
                    # Materialize the FF2 output shard spec explicitly. Some runtimes can
                    # mis-pick defaults when only a generic BLOCK sharded memory_config is provided.
                    cache_key_y2 = (int(B), 1, int(N), int(C), int(grid_x), int(grid_y))
                    y2_shard_mc = self._mlp_block_shard_mc_cache.get(cache_key_y2)
                    if y2_shard_mc is None:
                        core_grid = ttnn.CoreGrid(y=int(grid_y), x=int(grid_x))
                        y2_shard_mc = ttnn.create_sharded_memory_config(
                            (int(B), 1, int(N), int(C)),
                            core_grid=core_grid,
                            strategy=ttnn.ShardStrategy.BLOCK,
                            orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        )
                        self._mlp_block_shard_mc_cache[cache_key_y2] = y2_shard_mc
                    y1 = _ttnn_to_memory_config(y1, memory_config=y1_shard_mc, dtype=mlp_dtype)
                    did_reshard_for_fc2 = True
                except (RuntimeError, TypeError, ValueError):
                    did_reshard_for_fc2 = False

            ff2_memcfg = ff2_out_memcfg or memcfg
            if did_reshard_for_fc2 and y2_shard_mc is not None:
                ff2_memcfg = y2_shard_mc
            y2 = _ttnn_linear_with_optional_program_config(
                x=y1,
                w=self.w2_tt,
                bias=self.b2_tt,
                dtype=ff2_dtype,
                memory_config=ff2_memcfg,
                program_config=ff2_pc,
                op_name="mlp_ff2",
            )
            if incoming_sharded:
                _ttnn_deallocate(y1)

            if not wants_4d:
                y2 = ttnn.reshape(y2, (int(B), int(N), int(y2.shape[-1])))
            if incoming_sharded and tokens_shard_mc is not None:
                # Restore the encoder-wide token sharding spec for residual adds.
                try:
                    y2 = ttnn.to_memory_config(y2, memory_config=tokens_shard_mc, dtype=tokens_shard_dtype)
                except TypeError:
                    y2 = ttnn.to_memory_config(y2, memory_config=tokens_shard_mc)
                    if tokens_shard_dtype is not None:
                        try:
                            if hasattr(ttnn, "typecast"):
                                y2 = ttnn.typecast(y2, dtype=tokens_shard_dtype, memory_config=tokens_shard_mc)
                            else:
                                y2 = ttnn.to_dtype(y2, dtype=tokens_shard_dtype)
                        except (RuntimeError, TypeError, ValueError):
                            # If dtype conversion fails, keep the tensor as-is; strict perf runs will surface issues.
                            pass

            if wants_4d:
                y2 = ttnn.reshape(y2, (int(B), 1, int(N), int(y2.shape[-1])))
            return y2
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError("TTMLP device path failed") from exc


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
        )
        self.ln2 = TTLayerNorm(
            ln2_w,
            ln2_b,
            eps,
            device,
            output_mem=output_mem,
            program_config=program_config,
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
        )
        self.mlp = TTMLP(
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
            device,
            output_mem=output_mem,
            program_config=program_config,
        )

    def __call__(self, x: ttnn.Tensor, **mm_opts) -> ttnn.Tensor:
        # Stage-2/3: perf path must remain fully on-device with no DRAM "islands".
        # Pass LN outputs directly into the next op so sharding moves can "consume"
        # temporaries without fighting Python-level aliases during trace capture.
        add_dtype = mm_opts.get("tokens_shard_dtype", ttnn.bfloat16)
        add_mc = mm_opts.get("tokens_shard_mc", None)
        add_kwargs = {"memory_config": add_mc} if add_mc is not None else {}
        h = self.attn(self.ln1(x), **mm_opts)
        try:
            # Prefer in-place residual updates to reduce allocation pressure during trace capture.
            x = ttnn.add(x, h, dtype=add_dtype, output_tensor=x)
        except (RuntimeError, TypeError, ValueError):
            try:
                x = ttnn.add(x, h, dtype=add_dtype, **add_kwargs)
            except TypeError:
                try:
                    x = ttnn.add(x, h, **add_kwargs)
                except TypeError:
                    x = ttnn.add(x, h)
        _ttnn_deallocate(h)
        h = self.mlp(self.ln2(x), **mm_opts)
        try:
            x = ttnn.add(x, h, dtype=add_dtype, output_tensor=x)
        except (RuntimeError, TypeError, ValueError):
            try:
                x = ttnn.add(x, h, dtype=add_dtype, **add_kwargs)
            except TypeError:
                try:
                    x = ttnn.add(x, h, **add_kwargs)
                except TypeError:
                    x = ttnn.add(x, h)
        _ttnn_deallocate(h)
        return x
