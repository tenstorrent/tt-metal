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
    return get_compute_kernel_config(mesh_device, math_fidelity=ttnn.MathFidelity.LoFi)


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


def pad_to_tile(dim: int) -> int:
    tile = ttnn.TILE_SIZE
    if dim % tile == 0:
        return dim
    return ((dim + tile - 1) // tile) * tile


def _fused_activation_param(activation: Optional[str]):
    """Map linear ``activation`` strings to ``UnaryWithParam`` for matmul program configs."""
    if activation is None:
        return None
    if activation == "silu":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
    raise ValueError(f"Unsupported fused activation for matmul program config: {activation!r}")


def _largest_divisor_at_most(n: int, cap: int) -> int:
    cap = max(1, cap)
    for d in range(min(cap, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def _pick_width_shard_grid(hidden: int) -> ttnn.CoreGrid:
    """32-core width grid for hidden=12288 (profiler: fastest RMSNorm on BH)."""
    hidden_padded = pad_to_tile(hidden)
    for grid_y, grid_x in ((4, 8), (8, 4), (2, 8), (4, 4)):
        num_cores = grid_y * grid_x
        shard_w = hidden_padded // num_cores
        if hidden_padded % num_cores == 0 and shard_w % ttnn.TILE_SIZE == 0:
            return ttnn.CoreGrid(y=grid_y, x=grid_x)
    return ttnn.CoreGrid(y=4, x=4)


@lru_cache(maxsize=8)
def get_decode_width_sharded_activation_mem_config(hidden_size: int) -> ttnn.MemoryConfig:
    """L1 WIDTH-sharded activations for decode (``M`` padded to one tile row, hidden split)."""
    hidden_padded = pad_to_tile(hidden_size)
    grid = _pick_width_shard_grid(hidden_padded)
    shard_w = hidden_padded // grid.num_cores
    return ttnn.create_sharded_memory_config(
        (ttnn.TILE_SIZE, shard_w),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _width_sharded_norm_program_config(
    *,
    seq_len: int,
    hidden_size: int,
) -> ttnn.LayerNormShardedMultiCoreProgramConfig:
    """Shared width-sharded ``LayerNormShardedMultiCoreProgramConfig`` (see ``layernorm_unit_test``)."""
    m_padded = pad_to_tile(seq_len)
    hidden_padded = pad_to_tile(hidden_size)
    grid = _pick_width_shard_grid(hidden_padded)
    shard_w = hidden_padded // grid.num_cores
    block_h = m_padded // ttnn.TILE_SIZE
    block_w = shard_w // ttnn.TILE_SIZE
    subblock_w = _largest_divisor_at_most(block_w, 4)
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[grid.x, grid.y],
        subblock_w=subblock_w,
        block_h=block_h,
        block_w=block_w,
        inplace=False,
    )


@lru_cache(maxsize=8)
def get_decode_width_sharded_norm_program_config(hidden_size: int) -> ttnn.LayerNormShardedMultiCoreProgramConfig:
    """``LayerNormShardedMultiCoreProgramConfig`` for decode RMSNorm (``block_h=1`` tile row)."""
    return _width_sharded_norm_program_config(seq_len=ttnn.TILE_SIZE, hidden_size=hidden_size)


@lru_cache(maxsize=32)
def get_prefill_width_sharded_activation_mem_config(seq_len: int, hidden_size: int) -> ttnn.MemoryConfig:
    """L1 WIDTH-sharded activations for prefill RMSNorm (``M`` and hidden split across cores)."""
    m_padded = pad_to_tile(seq_len)
    hidden_padded = pad_to_tile(hidden_size)
    grid = _pick_width_shard_grid(hidden_padded)
    shard_w = hidden_padded // grid.num_cores
    return ttnn.create_sharded_memory_config(
        (m_padded, shard_w),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@lru_cache(maxsize=32)
def get_prefill_width_sharded_norm_program_config(
    seq_len: int, hidden_size: int
) -> ttnn.LayerNormShardedMultiCoreProgramConfig:
    """Width-sharded norm program config for prefill (e.g. ``M=128`` → ``block_h=4``)."""
    return _width_sharded_norm_program_config(seq_len=seq_len, hidden_size=hidden_size)


def get_sharded_norm_compute_kernel_config(mesh_device) -> ttnn.DeviceComputeKernelConfig:
    """HiFi2 + fp32 dest acc, ``packer_l1_acc=False`` for sharded norm L1 CB budget on BH."""
    cfg = dict(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    if is_blackhole_mesh(mesh_device):
        return ttnn.types.BlackholeComputeKernelConfig(**cfg)
    return ttnn.WormholeComputeKernelConfig(**cfg)


def get_decode_residual_mem_config(args: Devstral2Args, mesh_device) -> ttnn.MemoryConfig:
    """Width-sharded L1 residual layout for decode (alias of decode activation layout)."""
    _ = mesh_device
    return get_decode_width_sharded_activation_mem_config(args.hidden_size)


def get_activation_mem_config(args: Devstral2Args, mode: str, mesh_device) -> ttnn.MemoryConfig:
    """Activation memory layout by mode.

    Decode matmuls use L1 interleaved; width-sharded layout is scoped to RMSNorm only
    (see ``TtRMSNorm`` / ``layernorm_unit_test``). MLP intermediates (K=3584) cannot share
    the hidden width shard (K=12288).

    Prefill stays L1 interleaved (or DRAM on BH for long sequences).

    ``Devstral2Args.prefill_activations_dram`` forces DRAM prefill on any mesh (e.g. agent demo).
    """
    if mode == "decode":
        _ = args
        return ttnn.L1_MEMORY_CONFIG
    if mode == "prefill" and (args.prefill_activations_dram or is_blackhole_mesh(mesh_device)):
        return ttnn.DRAM_MEMORY_CONFIG
    _ = mesh_device
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
        fused_activation=_fused_activation_param(fused_activation),
    )


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
        fused_activation=_fused_activation_param(fused_activation),
        mcast_in0=True,
    )


def get_matmul_1d_width_sharded_in0_program_config(
    args: Devstral2Args,
    mesh_device,
    *,
    m: int,
    k: int,
    n: int,
    fused_activation: Optional[str] = None,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """1D-on-N matmul when decode activations are L1 WIDTH-sharded (same grid as RMSNorm).

      ``in0_block_w`` must divide **per-core** K tiles (``k_tiles / width_cores``), not total
    ``k_tiles`` — see ``matmul_device_operation.cpp`` and ``matmul_unit_test/configs.py``.
    """
    _ = args
    m_tiles = max(1, math.ceil(m / ttnn.TILE_SIZE))
    k_tiles = max(1, math.ceil(k / ttnn.TILE_SIZE))
    n_tiles = max(1, math.ceil(n / ttnn.TILE_SIZE))
    width_grid = _pick_width_shard_grid(k)
    num_width_cores = width_grid.num_cores
    if k_tiles % num_width_cores != 0:
        raise ValueError(f"K={k} tiles ({k_tiles}) not divisible by width cores ({num_width_cores})")
    per_shard_k_tiles = k_tiles // num_width_cores
    per_core_M = m_tiles
    grid_x, grid_y = _pick_1d_grid(mesh_device, n_tiles=n_tiles)
    num_cores = grid_x * grid_y
    per_core_N = max(1, math.ceil(n_tiles / num_cores))
    cap = min(
        8,
        max(1, 64 // per_core_M),
        max(1, 128 // per_core_N),
    )
    in0_block_w = _largest_divisor_at_most(per_shard_k_tiles, cap)
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
        fused_activation=_fused_activation_param(fused_activation),
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
    fused_activation: Optional[str] = None,
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
    _ = kind
    if k is None or n is None:
        return None
    s = max(1, int(seq_len))
    m = max(ttnn.TILE_SIZE, math.ceil(s / ttnn.TILE_SIZE) * ttnn.TILE_SIZE)
    return get_matmul_1d_program_config(args, mesh_device, m=m, k=k, n=n, fused_activation=fused_activation)
