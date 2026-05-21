# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Memory layouts and matmul program configs for Devstral-2 / Ministral3."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Optional

import ttnn

from models.experimental.devstral2_large.tt.model_args import is_blackhole_mesh

if TYPE_CHECKING:
    from models.experimental.devstral2_large.tt.model_args import Devstral2Args

LinearKind = Literal["qkv", "o_proj", "gate", "up", "down"]


def get_compute_kernel_config(mesh_device, *, math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2):
    """Pick a kernel config for the device architecture. Defaults to HiFi2."""
    cfg = dict(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    if is_blackhole_mesh(mesh_device):
        return ttnn.types.BlackholeComputeKernelConfig(**cfg)
    return ttnn.WormholeComputeKernelConfig(**cfg)


def get_compute_kernel_config_hifi4(mesh_device):
    """HiFi4 kernel config for matmuls quantized to bfloat8_b that need full accuracy."""
    return get_compute_kernel_config(mesh_device, math_fidelity=ttnn.MathFidelity.HiFi4)


@lru_cache(maxsize=4)
def get_sdpa_decode_program_config(mesh_device) -> ttnn.SDPAProgramConfig:
    """Limit SDPA decode to 8×8 cores (default BH grid is 11×10 and clashes with Q heads in L1)."""
    _ = mesh_device
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
    )


@lru_cache(maxsize=4)
def get_sdpa_decode_compute_kernel_config(mesh_device):
    """SDPA decode: HiFi4 + fp32 dest acc (matches tt_transformers ``compute_kernel_config_sdpa``).

    Per-step decode precision drives 88-layer PCC compounding. ``fp32_dest_acc_en=True`` actually
    *shrinks* the SDPA-decode K/V/QK circular buffers (chunk = ``dst_size = fp32_dest_acc_en ? 4 : 8``
    when ``k_chunk_size=0``), so it is strictly L1-safer than the prior HIFI2_NA config and
    cannot re-trigger the CB overlap fixed by 7e7fe6881a5. ``packer_l1_acc`` stays False (the
    actual L1-acc-buffer flag) to preserve the L1 budget.
    """
    cfg = dict(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    if is_blackhole_mesh(mesh_device):
        return ttnn.types.BlackholeComputeKernelConfig(**cfg)
    return ttnn.WormholeComputeKernelConfig(**cfg)


def get_sdpa_decode_output_mem_config(args: Devstral2Args, batch_size: int) -> ttnn.MemoryConfig:
    """Height-sharded L1 layout for SDPA decode output before ``nlp_concat_heads_decode``."""
    padded_heads = math.ceil(args.n_local_heads / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
        }
    )
    if batch_size > 1:
        core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(8, 8), row_wise=True)
    return ttnn.create_sharded_memory_config(
        shape=(padded_heads, args.head_dim),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def get_decode_residual_mem_config(args: Devstral2Args, mesh_device) -> ttnn.MemoryConfig:
    """Width-sharded L1 residual layout for single-token decode (matches tt_transformers decode)."""
    if args.num_devices <= 1:
        return ttnn.L1_MEMORY_CONFIG
    grid = ttnn.CoreGrid(y=4, x=4)
    shard_width = args.hidden_size // grid.num_cores
    return ttnn.create_sharded_memory_config(
        (ttnn.TILE_SIZE, shard_width),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def get_activation_mem_config(args: Devstral2Args, mode: str, mesh_device) -> ttnn.MemoryConfig:
    """L1 interleaved for prefill and decode.

    Width-sharded decode residuals match tt_transformers but conflict with our
    height-sharded Q/K/V heads (fused cache update needs non-overlapping shard grids).
    """
    _ = (args, mode, mesh_device)
    return ttnn.L1_MEMORY_CONFIG


def _dram_shard_core_count(k: int, n: int) -> int:
    for cores in (16, 12, 8, 4, 2, 1):
        if k % (ttnn.TILE_SIZE * cores) == 0:
            return cores
    return 1


def get_dram_sharded_matmul_program_config(
    args: Devstral2Args,
    *,
    m: int,
    k: int,
    n: int,
    fused_activation: Optional[str] = None,
) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
    """DRAM-sharded matmul for decode-width inputs (profiler: higher DRAM BW than default)."""
    num_cores = _dram_shard_core_count(k, n)
    in0_block_w = max(1, k // (ttnn.TILE_SIZE * num_cores))
    while k % (in0_block_w * ttnn.TILE_SIZE) != 0 and in0_block_w > 1:
        in0_block_w -= 1
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=max(1, math.ceil(m / ttnn.TILE_SIZE)),
        per_core_N=max(1, math.ceil(n / (ttnn.TILE_SIZE * num_cores))),
        fused_activation=fused_activation,
    )


def _largest_divisor_at_most(n: int, cap: int) -> int:
    """Largest ``d`` such that ``n % d == 0`` and ``1 ≤ d ≤ cap``."""
    cap = max(1, cap)
    for d in range(min(cap, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def _pick_1d_grid(mesh_device, *, n_tiles: int) -> tuple[int, int]:
    """For 1D-on-N (``mcast_in0=True``) matmul: pick ``(grid_x, grid_y)`` to maximize parallelism.

    Strategy:
      - If ``n_tiles ≤ max_x * max_y``: find the **smallest** rectangle ``(gx, gy)`` with
        ``gx*gy ≥ n_tiles`` that fits ``(max_x, max_y)``. This drives ``per_core_N`` toward 1
        (maximum parallelism) with at most a few idle cores. Prefer wider grids
        (larger ``gx``) at equal core count.
      - If ``n_tiles > max_x * max_y``: use the full worker grid; ``per_core_N`` becomes
        ``ceil(n_tiles / num_cores)``.

    The previous version only accepted rectangles that *exactly divided* ``n_tiles``, which
    bottlenecked Q (Nt=96) at 48 cores because 96 has no rectangular divisor inside (11, 10):
    (12, 8) violates gx ≤ 11 and (8, 12) violates gy ≤ 10. Allowing ``gx*gy ≥ n_tiles`` lifts
    that to (11, 9) = 99 cores at per_core_N=1 — roughly 2× more parallelism on Q.

    Must cap with :meth:`MeshDevice.compute_with_storage_grid_size` (not a fixed BH 13×10):
    that size is the legal tensix rectangle **excluding dispatch cores**; a larger grid
    places kernels on dispatch cores and fails ``validate_kernel_placement`` at runtime.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    max_x, max_y = int(grid.x), int(grid.y)
    max_cores = max_x * max_y
    if n_tiles >= max_cores:
        return max_x, max_y
    # Smallest cores ≥ n_tiles that forms a valid rectangle inside (max_x, max_y).
    for cores in range(n_tiles, max_cores + 1):
        for gx in range(min(max_x, cores), 0, -1):
            if cores % gx == 0 and cores // gx <= max_y:
                return gx, cores // gx
    return max_x, max_y


def get_matmul_1d_program_config(
    args: Devstral2Args,
    mesh_device,
    *,
    m: int,
    k: int,
    n: int,
    fused_activation: Optional[str] = None,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """1D-on-N (``mcast_in0=True``) multicast matmul. ``in0`` is multicast to every compute core;
    each core handles a slice of ``N``. The 2D variant is bounded by ``grid_y ≤ Mt`` (so just 4
    rows at ``M=128``); the 1D variant flattens the grid and goes up to ``Nt`` cores.

    Predicted configs on BH P150 when ``compute_with_storage_grid_size`` is 11×10:

    +---------+-----+--------------+---------+---------------------+
    | Linear  | Nt  | Grid (gx×gy) | Cores   | per_core_N (ceil)   |
    +=========+=====+==============+=========+=====================+
    | Q       | 96  | 11×9 = 99    | 99      | 1 (3 cores idle)    |
    | KV      | 16  | 8×2 = 16     | 16      | 1                   |
    | WO      | 384 | 11×10 = 110  | 110     | 4 (14 cores idle)   |
    | W1/W3   | 224 | 11×10 = 110  | 110     | 3                   |
    | W2      | 384 | 11×10 = 110  | 110     | 4 (14 cores idle)   |
    +---------+-----+--------------+---------+---------------------+

    Before this round, the strict-divisor variant gave: Q→48, KV→16, WO→64, W1/W3→56, W2→64.
    """
    _ = args
    m_tiles = max(1, math.ceil(m / ttnn.TILE_SIZE))
    n_tiles = max(1, math.ceil(n / ttnn.TILE_SIZE))
    k_tiles = max(1, math.ceil(k / ttnn.TILE_SIZE))
    grid_x, grid_y = _pick_1d_grid(mesh_device, n_tiles=n_tiles)
    num_cores = grid_x * grid_y
    per_core_M = m_tiles  # full M on each core (1D-on-N)
    # ceil so the case num_cores > n_tiles (grid slightly larger than n_tiles) → per_core_N=1
    # rather than 0. With num_cores ≤ n_tiles this matches the previous floor behavior.
    per_core_N = max(1, math.ceil(n_tiles / num_cores))
    # L1 CB budgets: in0 ≤ ~256 KB, in1 ≤ ~512 KB (BF16, double-buf, 2 KB/tile).
    cap = min(
        8,
        max(1, 64 // per_core_M),
        max(1, 128 // per_core_N),
    )
    in0_block_w = _largest_divisor_at_most(k_tiles, cap)
    # BH fp32_dest_acc_en=True caps out_subblock_h × out_subblock_w ≤ 4.
    out_subblock_w = _largest_divisor_at_most(per_core_N, 4)
    out_subblock_h = _largest_divisor_at_most(per_core_M, max(1, 4 // out_subblock_w))
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def get_linear_program_config(
    args: Devstral2Args,
    mesh_device,
    *,
    mode: str,
    kind: LinearKind,
    seq_len: int = 1,
    k: Optional[int] = None,
    n: Optional[int] = None,
) -> Optional[ttnn.ProgramConfig]:
    """Return a matmul program config. ``k``/``n`` should be supplied by the caller
    (``weight.shape[-2]`` / ``weight.shape[-1]``). Returns ``None`` if shape info is missing
    (TTNN auto-picks instead).

    All linears use 1D-on-N multicast. DRAM-sharded matmul was tried for decode QKV but
    regressed both prefill and decode: the DRAM-sharded factory caps compute at ~num_dram_banks
    (~12 cores on BH P150 — see
    matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:112-137), which is far less
    than 1D-on-N's ``Nt=56`` cores for the fused QKV. DRAM-sharded only wins when
    ``Nt <= num_dram_banks``.
    """
    _ = (mode, kind)
    if k is None or n is None:
        return None
    s = max(1, int(seq_len))
    m = max(ttnn.TILE_SIZE, math.ceil(s / ttnn.TILE_SIZE) * ttnn.TILE_SIZE)
    return get_matmul_1d_program_config(args, mesh_device, m=m, k=k, n=n)
