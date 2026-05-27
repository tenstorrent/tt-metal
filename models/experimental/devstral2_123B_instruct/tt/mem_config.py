# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Memory layouts and matmul program configs for Devstral-2 / Ministral3."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Optional

import ttnn

from models.experimental.devstral2_123B_instruct.tt.model_args import is_blackhole_mesh

if TYPE_CHECKING:
    from models.experimental.devstral2_123B_instruct.tt.model_args import Devstral2Args

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
    """SDPA decode: HiFi4 + fp32 dest acc; ``packer_l1_acc=False`` to stay within L1 CB budget."""
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
    """32-core width grid for decode RMSNorm (one tile row per core shard)."""
    hidden_padded = pad_to_tile(hidden)
    for grid_y, grid_x in ((4, 8), (8, 4), (2, 8), (4, 4)):
        num_cores = grid_y * grid_x
        shard_w = hidden_padded // num_cores
        if hidden_padded % num_cores == 0 and shard_w % ttnn.TILE_SIZE == 0:
            return ttnn.CoreGrid(y=grid_y, x=grid_x)
    return ttnn.CoreGrid(y=4, x=4)


def _pick_prefill_width_shard_grid(hidden: int) -> ttnn.CoreGrid:
    """64-core (8×8) width grid for prefill RMSNorm — halves per-core ``block_w`` vs decode."""
    hidden_padded = pad_to_tile(hidden)
    for grid_y, grid_x in ((8, 8), (4, 8), (8, 4), (2, 8), (4, 4)):
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
    prefill: bool = False,
) -> ttnn.LayerNormShardedMultiCoreProgramConfig:
    """Width-sharded LayerNorm program config; prefill uses the larger core grid."""
    m_padded = pad_to_tile(seq_len)
    hidden_padded = pad_to_tile(hidden_size)
    grid = _pick_prefill_width_shard_grid(hidden_padded) if prefill else _pick_width_shard_grid(hidden_padded)
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
    """L1 WIDTH-sharded activations for prefill RMSNorm (``M`` and hidden split across cores).

    Uses the larger 64-core prefill grid so per-core ``block_w`` stays small enough that
    sharded LayerNorm CBs fit in L1 at long prefill seq_lens.
    """
    m_padded = pad_to_tile(seq_len)
    hidden_padded = pad_to_tile(hidden_size)
    grid = _pick_prefill_width_shard_grid(hidden_padded)
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
    return _width_sharded_norm_program_config(seq_len=seq_len, hidden_size=hidden_size, prefill=True)


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


def get_prefill_width_sharded_embedding_mem_config(seq_len: int, hidden_size: int) -> ttnn.MemoryConfig:
    """L1 WIDTH-sharded ``ttnn.embedding`` output matching prefill RMSNorm layout.

    Same 8×8 grid and per-core shard ``(M, hidden/64)`` as
    ``get_prefill_width_sharded_activation_mem_config`` (e.g. seq=128 → ``[128, 192]`` per core,
    ~48 KiB vs ~768 KiB per core for HEIGHT ``[32, 12288]`` on 4 cores).

    ``EmbeddingsDeviceOperation`` tilized path supports WIDTH-sharded outputs; ``TtRMSNorm`` can
    skip ``interleaved_to_sharded`` when embed already uses this mem config.
    """
    return get_prefill_width_sharded_activation_mem_config(seq_len, hidden_size)


def get_embedding_output_mem_config(
    args: Devstral2Args,
    mode: str,
    mesh_device,
    *,
    batch_size: int = 1,
    seq_len: int = 1,
) -> ttnn.MemoryConfig:
    """Memory config for ``ttnn.embedding`` output (distinct from matmul ``act_mem``).

    - **Prefill:** WIDTH-sharded L1 aligned with width-sharded RMSNorm when ``seq_len <= kv_block_size``.
    - **Decode:** L1 interleaved — WIDTH embed needs ``input_volume % shard_h == 0`` but decode
      ``M=1`` while the norm shard height is ``TILE_SIZE`` (32).
    """
    _ = mesh_device
    if mode == "decode":
        return ttnn.L1_MEMORY_CONFIG
    seq = max(1, int(seq_len))
    if seq <= args.kv_block_size:
        return get_prefill_width_sharded_embedding_mem_config(seq, args.hidden_size)
    return ttnn.L1_MEMORY_CONFIG


def get_activation_mem_config(args: Devstral2Args, mode: str, mesh_device) -> ttnn.MemoryConfig:
    """Interleaved L1 activations for matmuls; width-sharded layout is RMSNorm-only."""
    _ = (args, mode, mesh_device)
    return ttnn.L1_MEMORY_CONFIG


def get_prefill_width_sharded_matmul_output_mem_config() -> ttnn.MemoryConfig:
    """WIDTH-sharded L1 matmul output (N sharded across the prefill norm 8×8 grid)."""
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )


def get_prefill_qkv_matmul_output_mem_config() -> ttnn.MemoryConfig:
    """Alias for :func:`get_prefill_width_sharded_matmul_output_mem_config`."""
    return get_prefill_width_sharded_matmul_output_mem_config()


def get_prefill_width_sharded_matmul_program_config(
    args: Devstral2Args,
    mesh_device,
    *,
    seq_len: int,
    n: int,
    fused_activation: Optional[str] = None,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """1D mcast matmul on the prefill RMSNorm 8×8 grid (ws in0 from norm, ws out).

    Used for QKV and MLP gate/up when ``K == hidden_size``. Uses 64 cores so ``Kt=384`` divides
    evenly for width-sharded activations from norm.
    """
    _ = mesh_device
    m_padded = pad_to_tile(max(1, int(seq_len)))
    m_tiles = m_padded // ttnn.TILE_SIZE
    n_tiles = max(1, math.ceil(n / ttnn.TILE_SIZE))
    k_tiles = max(1, math.ceil(args.hidden_size / ttnn.TILE_SIZE))
    grid = _pick_prefill_width_shard_grid(args.hidden_size)
    grid_x, grid_y = grid.x, grid.y
    num_cores = grid_x * grid_y
    if k_tiles % num_cores != 0:
        raise ValueError(
            f"width-sharded prefill matmul (K=hidden) requires Kt={k_tiles} divisible by "
            f"norm grid cores={num_cores}"
        )
    per_core_M = m_tiles
    per_core_N = max(1, math.ceil(n_tiles / num_cores))
    kt_per_core = k_tiles // num_cores
    cap = min(8, max(1, 64 // per_core_M), max(1, 128 // per_core_N))
    in0_block_w = _largest_divisor_at_most(kt_per_core, cap)
    out_subblock_h = 1
    out_subblock_w = _largest_divisor_at_most(per_core_N, 4)
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


def get_prefill_qkv_matmul_program_config(
    args: Devstral2Args,
    mesh_device,
    *,
    seq_len: int,
    n: int,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """Alias for QKV (no fused activation)."""
    return get_prefill_width_sharded_matmul_program_config(
        args, mesh_device, seq_len=seq_len, n=n, fused_activation=None
    )


def use_width_sharded_prefill_norm_matmul(args: Devstral2Args, mode: str, seq_len: int) -> bool:
    """True when prefill linears may consume width-sharded RMSNorm output (QKV, gate, up)."""
    return mode == "prefill" and int(seq_len) <= args.kv_block_size


def _pick_1d_grid(mesh_device, *, n_tiles: int) -> tuple[int, int]:
    """Pick a 1D-on-N matmul grid: smallest ``(gx, gy)`` with ``gx*gy >= n_tiles`` inside the device grid."""
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
    """1D-on-N multicast matmul (``mcast_in0=True``): broadcast ``in0``, shard ``N`` across cores."""
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
    """1D-on-N matmul program config from weight shapes; ``None`` lets TTNN auto-pick."""
    _ = kind
    if k is None or n is None:
        return None
    s = max(1, int(seq_len))
    m = max(ttnn.TILE_SIZE, math.ceil(s / ttnn.TILE_SIZE) * ttnn.TILE_SIZE)
    return get_matmul_1d_program_config(args, mesh_device, m=m, k=k, n=n, fused_activation=fused_activation)
