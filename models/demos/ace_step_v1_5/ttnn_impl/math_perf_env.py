# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step TTNN throughput helpers (alignment with ``tt-perf-report`` / ``perf*.txt`` stacks).

Stacked E2E summaries often show large DRAM-interleaved shares for:

- ``PermuteDeviceOperation`` (~26 %)
- ``ReshapeViewDeviceOperation`` (~22 %)

Both ``ttnn.reshape`` and ``ttnn.permute`` accept ``memory_config``; placing outputs in L1 trims
DRAM traffic. **On by default** for demo, E2E, and Tracy paths (set env vars to ``0`` to disable).

DiT linears are often DRAM-bound at HiFi4 without tuning:

- ``256×1024×1024`` — attn ``q_proj`` / ``o_proj``
- ``256×2048×2048`` — fused attn ``wkv``
- ``256×3072×3072`` — MLP ``gate_proj`` / ``up_proj`` / ``down_proj``

VAE decode exposes large-M matmuls inside ``conv1d`` / ``conv_transpose2d`` im2col (e.g.
``1920×512×512``, ``30720×128×128``, ``61440×128×128``). Enable L1 activations via
``ACE_STEP_VAE_CONV_PERF`` (TTNN picks matmul program config from conv + L1 geometry).

Condition encoder linears (lyric/timbre, ``hidden_size=2048``) are often DRAM-bound:

- ``32×2048×2048`` — attn ``q``/``k``/``v``/``o`` (short packed sequences)
- ``32×6144×6144`` — MLP ``gate``/``up``
- ``288×2048×2048`` — longer lyric windows

``perf5.txt`` recommends L1 activations, ``in0_block_w≥2``, and HiFi2 when not FLOP-bound.

Environment (all default **on**; set to ``0`` / ``false`` to disable for PCC/debug):

- ``ACE_STEP_TM_OUTPUT_L1``: L1 for **both** reshape and permute outputs (shortcut).
- ``ACE_STEP_RESHAPE_OUTPUT_L1``: L1 only for ``ttnn.reshape``.
- ``ACE_STEP_PERMUTE_OUTPUT_L1``: L1 only for ``ttnn.permute``.
- ``ACE_STEP_DIT_LINEAR_PERF``: HiFi2 + L1 + matmul program config on DiT attn + MLP gate/up/down.
- ``ACE_STEP_COND_LINEAR_PERF``: same on condition lyric/timbre encoders and Qwen3 text encoder.
- ``ACE_STEP_VAE_CONV_PERF``: L1 activations/outputs on Oobleck VAE convs.
"""

from __future__ import annotations

import os
from typing import Any


def _env_truthy(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if raw == "":
        return default
    if raw in ("0", "false", "no", "off", "n"):
        return False
    if raw in ("1", "true", "yes", "on", "y"):
        return True
    return default


def _l1_enabled_for(name: str) -> bool:
    if _env_truthy("ACE_STEP_TM_OUTPUT_L1", True):
        return True
    return _env_truthy(name, True)


def _l1_memory_kwargs(ttnn: Any) -> dict:
    mc = getattr(ttnn, "L1_MEMORY_CONFIG", None)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_reshape_kwargs(ttnn: Any) -> dict:
    """Keyword args for ``ttnn.reshape`` to steer outputs toward L1 when enabled."""
    if not _l1_enabled_for("ACE_STEP_RESHAPE_OUTPUT_L1"):
        return {}
    return _l1_memory_kwargs(ttnn)


def ace_step_permute_kwargs(ttnn: Any) -> dict:
    """Keyword args for ``ttnn.permute`` to steer outputs toward L1 when enabled."""
    if not _l1_enabled_for("ACE_STEP_PERMUTE_OUTPUT_L1"):
        return {}
    return _l1_memory_kwargs(ttnn)


def ace_step_dit_linear_perf_enabled() -> bool:
    """When true, DiT ``TtAceStepAttentionSDPA`` uses tuned ``ttnn.linear`` kwargs."""
    return _env_truthy("ACE_STEP_DIT_LINEAR_PERF", True)


def ace_step_cond_linear_perf_enabled() -> bool:
    """When true, condition lyric/timbre encoder layers use tuned ``ttnn.linear`` kwargs."""
    return _env_truthy("ACE_STEP_COND_LINEAR_PERF", True)


def ace_step_vae_conv_perf_enabled() -> bool:
    """When true, Oobleck VAE ``TtConv1d`` / ``TtConvTranspose1d`` use L1 + HiFi2-oriented conv paths."""
    return _env_truthy("ACE_STEP_VAE_CONV_PERF", True)


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
