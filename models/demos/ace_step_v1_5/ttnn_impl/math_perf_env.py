# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step TTNN throughput helpers (alignment with ``tt-perf-report`` / ``perf*.txt`` stacks).

Stacked E2E summaries often show large DRAM-interleaved shares for:

- ``PermuteDeviceOperation`` (~26 %)
- ``ReshapeViewDeviceOperation`` (~22 %)

Both ``ttnn.reshape`` and ``ttnn.permute`` accept ``memory_config``; this module **always** requests
L1 outputs where supported so reshape/permute chains avoid unnecessary DRAM round-trips.

E2E Tracy ``BinaryNgDeviceOperation (in0:dram_interleaved)`` (~24% device time) was dominated by
**VAE Snake** in FP32 (``BF16→FP32`` typecast, ~784 μs FP32 ``multiply``/``add`` per layer).
``TtSnake1d`` now always uses BF16 compute to match conv activations.

DiT linears are often DRAM-bound at HiFi4 without tuning:

- ``256×1024×1024`` — attn ``q_proj`` / ``o_proj``
- ``256×2048×2048`` — fused attn ``wkv``
- ``256×3072×3072`` — MLP ``gate_proj`` / ``up_proj`` / ``down_proj``

VAE decode exposes large-M matmuls inside ``conv1d`` / ``conv_transpose2d`` im2col (e.g.
``1920×512×512``, ``30720×128×128``, ``61440×128×128``). VAE conv uses **DRAM** activations
(``act_block_h_override=32``); L1 conv inputs clash with circular buffers on Blackhole.

Condition encoder linears (lyric/timbre, ``hidden_size=2048``) are often DRAM-bound:

- ``32×2048×2048`` — attn ``q``/``k``/``v``/``o`` (short packed sequences)
- ``32×6144×6144`` — MLP ``gate``/``up``
- ``288×2048×2048`` — longer lyric windows
"""

from __future__ import annotations

from typing import Any


def _l1_memory_kwargs(ttnn: Any) -> dict:
    mc = getattr(ttnn, "L1_MEMORY_CONFIG", None)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_reshape_kwargs(ttnn: Any) -> dict:
    """Keyword args for ``ttnn.reshape`` to place outputs in L1 when ``L1_MEMORY_CONFIG`` exists."""
    return _l1_memory_kwargs(ttnn)


def ace_step_permute_kwargs(ttnn: Any) -> dict:
    """Keyword args for ``ttnn.permute`` to place outputs in L1 when ``L1_MEMORY_CONFIG`` exists."""
    return _l1_memory_kwargs(ttnn)


def ace_step_dit_linear_perf_enabled() -> bool:
    """Deprecated: perf kwargs are always enabled. Kept for call-site compatibility."""
    return True


def ace_step_cond_linear_perf_enabled() -> bool:
    """Deprecated: perf kwargs are always enabled. Kept for call-site compatibility."""
    return True


def ace_step_vae_conv_perf_enabled() -> bool:
    """Deprecated: VAE conv perf path is always enabled. Kept for call-site compatibility."""
    return True


def ace_step_vae_large_m_matmul_program_config(
    device: Any,
    *,
    m_dim: int,
    k_dim: int,
    n_dim: int,
):
    """Matmul program config for VAE conv im2col shapes (e.g. 1920×512, 30720×128, 61440×128).

    TTNN conv does not accept ``program_config`` directly; this documents the target geometry when
    conv is re-packed with L1 activations (``prepare_conv_*`` + ``memory_config``).
    """
    tile = 32
    m = max(1, int(m_dim))
    k = max(tile, int(k_dim))
    n = max(tile, int(n_dim))
    if m >= 61440:
        return _mcast_1d_linear_program_config(
            device,
            seq_len=m,
            in_dim=k,
            out_dim=n,
            in0_block_w_cap=2,
            out_subblock_h_cap=2,
            out_subblock_w=2,
        )
    if m >= 7680:
        return _mcast_1d_linear_program_config(
            device,
            seq_len=m,
            in_dim=k,
            out_dim=n,
            in0_block_w_cap=2,
            out_subblock_h_cap=1,
            out_subblock_w=4,
        )
    return _mcast_1d_linear_program_config(
        device,
        seq_len=m,
        in_dim=k,
        out_dim=n,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


def ace_step_linear_l1_memory_config(ttnn: Any):
    """L1 interleaved buffer config for linear activations / outputs."""
    return getattr(ttnn, "L1_MEMORY_CONFIG", None)


def ace_step_dit_linear_l1_memory_config(ttnn: Any):
    """Alias for :func:`ace_step_linear_l1_memory_config`."""
    return ace_step_linear_l1_memory_config(ttnn)


def ace_step_nlp_concat_heads(ttnn: Any, ctx: Any, *, l1_mc: Any | None = None) -> Any:
    """Replace output permute+reshape with ``ttnn.experimental.nlp_concat_heads``.

    Converts ``[B, H, S, Dh]`` → ``[B, 1, S, H*Dh]`` in a single device kernel,
    eliminating one permute (~360 μs) and one non-view reshape (~405 μs) per attention block.

    Falls back to the original permute+reshape path if the op is unavailable.
    """
    experimental = getattr(ttnn, "experimental", None)
    nlp_concat = getattr(experimental, "nlp_concat_heads", None) if experimental is not None else None
    mc = l1_mc if l1_mc is not None else ace_step_linear_l1_memory_config(ttnn)
    if nlp_concat is not None:
        kw = {"memory_config": mc} if mc is not None else {}
        return nlp_concat(ctx, **kw)
    # Fallback: original two-op path.
    B, H, S, Dh = int(ctx.shape[0]), int(ctx.shape[1]), int(ctx.shape[2]), int(ctx.shape[3])
    _kw = {"memory_config": mc} if mc is not None else {}
    ctx = ttnn.permute(ctx, (0, 2, 1, 3), **_kw)
    ctx = ttnn.reshape(ctx, (B, 1, S, H * Dh), **_kw)
    return ctx


def ace_step_eltwise_l1_memory_config(ttnn: Any):
    """L1 config for BinaryNg / ``add`` / ``multiply`` / ``softmax`` activations (Tracy ``in0`` bucket)."""
    return ace_step_linear_l1_memory_config(ttnn)


def ace_step_sdpa_mask_memory_config(ttnn: Any):
    """SDPA requires ``attn_mask`` buffers in DRAM (see ``sdpa_device_operation.cpp``)."""
    return getattr(ttnn, "DRAM_MEMORY_CONFIG", None)


def ace_step_ensure_l1_activation(ttnn: Any, tensor: Any, l1_mc: Any | None = None) -> Any:
    """Move a device tensor to L1 interleaved so BinaryNg tags ``in0:l1_interleaved``."""
    if tensor is None:
        return tensor
    mc = l1_mc if l1_mc is not None else ace_step_eltwise_l1_memory_config(ttnn)
    if mc is None or not hasattr(ttnn, "to_memory_config"):
        return tensor
    return ttnn.to_memory_config(tensor, mc)


def ace_step_binary_kwargs(ttnn: Any, l1_mc: Any | None = None) -> dict:
    """``memory_config`` for ``add`` / ``multiply`` / ``softmax`` with L1 output."""
    mc = l1_mc if l1_mc is not None else ace_step_eltwise_l1_memory_config(ttnn)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_init_hifi2_linear_compute_kernel_config(device: Any):
    """HiFi2 linear config for DRAM-bound projections (DiT + condition encoder)."""
    import ttnn

    init_ck = getattr(ttnn, "init_device_compute_kernel_config", None)
    if not callable(init_ck):
        return None
    return init_ck(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def ace_step_init_dit_linear_compute_kernel_config(device: Any):
    """Alias for :func:`ace_step_init_hifi2_linear_compute_kernel_config`."""
    return ace_step_init_hifi2_linear_compute_kernel_config(device)


def _mcast_1d_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    batch_size: int = 1,
    in0_block_w_cap: int = 2,
    out_subblock_h_cap: int = 4,
    out_subblock_w: int = 1,
):
    """Shared 1D mcast matmul program config builder for ACE-Step linears.

    With ``fuse_batch=True``, TTNN fuses batch into the ``M`` dim (tile rows) for tensors
    shaped ``[B, 1, S, K]``. TILE layout pads ``S`` to the tile height, so runtime
    ``M`` (tiles) is ``B * ceil(seq_len / tile)`` once ``S`` is padded to TILE height.
    ``seq_len`` without batch—or ignoring TILE padding along ``S``—can under-report
    ``per_core_M``. For ``N`` (tiles across the output inner dim), derive ``per_core_N`` from
    ``ceil(out_dim / tile)`` spread across ``grid.x`` so ``num_blocks_x`` stays within core count.
    ``out_subblock_w`` is clipped so ``per_core_N % out_subblock_w == 0`` (default ``out_block_w``
    equals ``per_core_N``).
    """
    import ttnn

    cfg_cls = getattr(ttnn, "MatmulMultiCoreReuseMultiCast1DProgramConfig", None)
    if cfg_cls is None or not hasattr(device, "compute_with_storage_grid_size"):
        return None

    grid = device.compute_with_storage_grid_size()
    tile = int(getattr(ttnn, "TILE_SIZE", 32))
    bsz = max(1, int(batch_size))
    seq = max(1, int(seq_len))
    # Match get_M_dim for fuse_batch: batch × sequence span in tile rows (~ceil(S_pad / tile)).
    s_tiles = (seq + tile - 1) // tile
    per_core_m = max(1, bsz * s_tiles)
    k = max(tile, int(in_dim))

    k_tiles = max(1, k // tile)
    in0_block_w = min(int(in0_block_w_cap), k_tiles)

    # per_core_N is in TILE columns along N (same units as Nt in MatmulReuseMcast1D factory).
    # Spreading ceil(out_dim / tile) across grid.x avoids the old stripe formula mixing element
    # widths with a large (grid.x * tile) denominator—it could set per_core_N too low on wide
    # grids so num_blocks_x exceeded available cores.
    n_width_tiles = max(1, (int(out_dim) + tile - 1) // tile)
    gx = max(1, int(grid.x))
    per_core_n = max(1, (n_width_tiles + gx - 1) // gx)

    out_subblock_h = min(int(out_subblock_h_cap), per_core_m)
    while per_core_m % out_subblock_h != 0 and out_subblock_h > 1:
        out_subblock_h -= 1

    # Default ``out_block_w`` is ``per_core_N``; TTNN requires ``out_block_w % out_subblock_w == 0``.
    out_subblock_w_eff = min(int(out_subblock_w), max(1, int(per_core_n)))
    while per_core_n % out_subblock_w_eff != 0 and out_subblock_w_eff > 1:
        out_subblock_w_eff -= 1

    return cfg_cls(
        compute_with_storage_grid_size=(int(grid.x), 1),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w_eff,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def ace_step_dit_attn_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    batch_size: int = 1,
):
    """``MatmulMultiCoreReuseMultiCast1DProgramConfig`` for square DiT ``q`` / ``o`` (e.g. 256×1024×1024)."""
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=in_dim,
        out_dim=out_dim,
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


def ace_step_cond_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    batch_size: int = 1,
):
    """Program config for condition encoder linears (e.g. 32×2048×2048, 288×2048×2048)."""
    short = int(seq_len) <= 64
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=in_dim,
        out_dim=out_dim,
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=2 if short else 4,
        out_subblock_w=2 if short else 1,
    )


def ace_step_cond_mlp_gate_up_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    batch_size: int = 1,
):
    """Program config for condition MLP gate/up (e.g. 32×6144×6144).

    Lyric/timbre encoders use intermediate 6144×2048; ``in0_block_w=2`` plus L1-hosted
    activations can overrun per-core circular-buffer budget (static CB vs tensor L1).
    Use ``in0_block_w_cap=1`` on this wide path by default.
    """
    short = int(seq_len) <= 64
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=hidden_size,
        out_dim=intermediate_size,
        batch_size=batch_size,
        in0_block_w_cap=1,
        out_subblock_h_cap=2 if short else 4,
        out_subblock_w=2 if short else 1,
    )


def ace_step_dit_fused_wkv_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    hidden_size: int,
    fused_kv_dim: int,
    batch_size: int = 1,
):
    """Program config for fused ``wkv`` (e.g. 256×2048×2048 when ``hidden_size=1024``, GQA ``fused_kv_dim=2048``)."""
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=hidden_size,
        out_dim=fused_kv_dim,
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


def ace_step_dit_mlp_gate_up_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    batch_size: int = 1,
):
    """Program config for MLP ``gate_proj`` / ``up_proj`` (e.g. 256×3072×3072)."""
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=hidden_size,
        out_dim=intermediate_size,
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


def ace_step_dit_mlp_down_proj_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    intermediate_size: int,
    hidden_size: int,
    batch_size: int = 1,
):
    """Program config for MLP ``down_proj`` (e.g. 256×3072×3072 when intermediate==hidden)."""
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=intermediate_size,
        out_dim=hidden_size,
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )
