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


def get_compute_kernel_config(mesh_device):
    """Pick HiFi2 kernel config for the device architecture."""
    cfg = dict(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    if is_blackhole_mesh(mesh_device):
        return ttnn.types.BlackholeComputeKernelConfig(**cfg)
    return ttnn.WormholeComputeKernelConfig(**cfg)


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
    """SDPA decode: HiFi2 without packer L1 acc (matches tt_transformers HIFI2_NA)."""
    cfg = dict(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
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


def get_linear_program_config(
    args: Devstral2Args,
    mesh_device,
    *,
    mode: str,
    kind: LinearKind,
    seq_len: int = 1,
) -> Optional[ttnn.ProgramConfig]:
    """Return a program config when it helps; ``None`` lets TTNN autoselect.

    Prefill multicast configs are disabled: hand-tuned ``per_core_M`` / ``per_core_N`` grids
    can reference logical cores that do not exist on Blackhole (e.g. ``(23, 0)``).
    """
    _ = seq_len
    if mode != "decode" or args.num_devices <= 1:
        return None
    # DRAM-sharded decode matmuls help on multi-device WH; on BH use TTNN defaults until tuned.
    if is_blackhole_mesh(mesh_device):
        return None
    m = ttnn.TILE_SIZE
    k = args.hidden_size
    if kind == "qkv":
        n = (args.n_local_heads + 2 * args.n_local_kv_heads) * args.head_dim
        return get_dram_sharded_matmul_program_config(args, m=m, k=k, n=n)
    if kind == "o_proj":
        return get_dram_sharded_matmul_program_config(
            args, m=m, k=args.n_local_heads * args.head_dim, n=args.hidden_size
        )
    if kind in ("gate", "up"):
        n = args.intermediate_size // args.tp
        return get_dram_sharded_matmul_program_config(args, m=m, k=k, n=n)
    if kind == "down":
        return get_dram_sharded_matmul_program_config(
            args, m=m, k=args.intermediate_size // args.tp, n=args.hidden_size
        )
    return None
