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

    # Decode
    decode_gate_up_cores: tuple[int, int] = (3, 4)
    decode_gate_up_in0_block_w: int = 30
    decode_down_cores: tuple[int, int] = (5, 6)
    decode_down_in0_block_w: int = 12

    # Prefill
    prefill_gate_up_cores: tuple[int, int] = (3, 4)
    prefill_gate_up_in0_block_w: int = 30
    prefill_down_cores: tuple[int, int] = (5, 6)
    prefill_down_in0_block_w: int = 12

    # Memory
    sequence_chunk_size: int = 4 * 1024
    base_down_split_size: int = 1024


# One device (TP=1): prefill computes all 90 output tile columns of every expert on one chip. Spread them over 90
# cores (one column per core, same K blocks) where the grid allows (Blackhole 11x10): measured on one P150 with BFP8
# experts, 2x faster expert prefill than the 12 / 30-core defaults (ISL 1024: 95 vs 187 ms per layer).
TP1_PREFILL_GATE_UP_CORES = (10, 9)
TP1_PREFILL_DOWN_CORES = (9, 10)


def gpt_oss_program_config(mesh_device, mesh_config):
    """GPTOSSProgramConfig for this mesh (the TP=1 prefill grids above on a large enough single device)."""
    grid = mesh_device.compute_with_storage_grid_size()
    fits = all(
        grid.x >= cores[0] and grid.y >= cores[1] for cores in (TP1_PREFILL_GATE_UP_CORES, TP1_PREFILL_DOWN_CORES)
    )
    if mesh_config is not None and mesh_config.prefill.tp == 1 and mesh_device.get_num_devices() == 1 and fits:
        return GPTOSSProgramConfig(
            prefill_gate_up_cores=TP1_PREFILL_GATE_UP_CORES, prefill_down_cores=TP1_PREFILL_DOWN_CORES
        )
    return GPTOSSProgramConfig()
