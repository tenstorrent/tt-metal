# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS expert program configurations."""

from dataclasses import dataclass

from models.demos.gpt_oss.tt.experts.config import ProgramConfig


@dataclass
class GPTOSSProgramConfig(ProgramConfig):
    """
    GPT-OSS expert configuration.

    Optimized for: hidden=2088, intermediate=360
    """

    # Decode. Gate and up are computed by one fused sparse_matmul with N = 2 * 384 = 24 tiles per
    # device at TP=8; one output tile per core (6x4) is ~2x faster than two per core (3x4) because
    # the per-expert work is compute-bound at these shapes. The down projection (N = 90 tiles) uses
    # 5x6 (3 tiles per core); for multi-user steps Blackhole's 13x10 grid allows one tile per core
    # (9x10, see gpt_oss_program_config()). ProgramConfig._build_matmul_config shrinks these grids
    # automatically when a different TP factor gives a tile count that does not fill them.
    decode_gate_up_cores: tuple[int, int] = (6, 4)
    decode_gate_up_in0_block_w: int = 30
    decode_down_cores: tuple[int, int] = (5, 6)
    decode_down_in0_block_w: int = 12

    # Prefill
    prefill_gate_up_cores: tuple[int, int] = (6, 4)
    prefill_gate_up_in0_block_w: int = 30
    prefill_down_cores: tuple[int, int] = (5, 6)
    prefill_down_in0_block_w: int = 12

    # Memory
    sequence_chunk_size: int = 4 * 1024
    base_down_split_size: int = 1024


def gpt_oss_program_config(mesh_device) -> GPTOSSProgramConfig:
    """GPT-OSS expert program config tuned for the device's compute grid.

    On Blackhole (13x10 compute grid) the multi-user decode down projection (N = 90 tiles) runs one
    output tile per core on a 9x10 grid: measured 385 -> 289 us per layer at batch 32 with all 128
    experts active on P150x8. Single-user decode keeps 5x6 (the 128-slot sparsity scan is cheaper
    with fewer multicast receivers: 120 vs 174 us per layer), as does Wormhole's 8x8 grid.
    9x10 is an exactly-filled 90-core rectangle inside the 13x10 grid (the sparse-matmul factory requires
    the cores with work to fill the multicast rectangle exactly); 10x9 would qualify as well, 9x10 uses
    the full height of the grid.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    if grid.x >= 9 and grid.y >= 10:
        return GPTOSSProgramConfig(decode_down_cores_batched=(9, 10))
    return GPTOSSProgramConfig()
