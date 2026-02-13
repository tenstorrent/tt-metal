# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
CB pointer validation utilities for verifying CB pointer wrapping in fused kernels.

Used to check that all CB read/write pointers return to their initial addresses
after one iteration — a prerequisite for adding looping to fused kernels.
"""

import numpy as np
import torch
from loguru import logger

import ttnn

NUM_CBS = 64
SECTION_SIZE = 6 * NUM_CBS  # uint32_t per before/after section
PER_CORE_BF16 = 12 * NUM_CBS * 2  # bfloat16 values per core (before + after)

RISC_NAMES = ["BRISC_rd", "BRISC_wr", "NCRISC_rd", "NCRISC_wr", "UNPACK_rd", "PACK_wr"]


def create_validation_tensor(device):
    """Create a validation tensor sharded across all device cores.

    Returns:
        (validation_tensor, grid_x, grid_y)
    """
    device_grid_size = device.compute_with_storage_grid_size()
    grid_x, grid_y = device_grid_size.x, device_grid_size.y
    num_cores = grid_x * grid_y

    tile_1x32 = ttnn.Tile([1, 32])
    full_device_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))])
    shard_spec = ttnn.ShardSpec(full_device_grid, (1, PER_CORE_BF16), ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    tensor = ttnn.from_torch(
        torch.zeros(num_cores, PER_CORE_BF16, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=tile_1x32,
    )
    return tensor, grid_x, grid_y


def get_validation_compile_time_args(validation_tensor):
    """Return named compile-time args list for the validation tensor address.

    Add to each RISC's compile-time arg list:
        ncrisc_args += get_validation_compile_time_args(tensor)
        brisc_args += get_validation_compile_time_args(tensor)
        trisc_args += get_validation_compile_time_args(tensor)
    """
    if validation_tensor is None:
        return [("validation_addr", 0)]
    return [("validation_addr", validation_tensor.buffer_address())]


def validate_cb_pointers(validation_tensor, grid_x, grid_y, skip_cbs=None):
    """Read back validation tensor and check CB pointer wrapping across all cores.

    Args:
        validation_tensor: The validation tensor to read back.
        grid_x: Device grid width.
        grid_y: Device grid height.
        skip_cbs: Set of CB indices to skip (e.g. DRAM matmul CBs with internal reset).

    Returns:
        True if all CB pointers wrapped back to initial addresses (excluding skip_cbs).
    """
    if skip_cbs is None:
        skip_cbs = set()

    validation_torch = ttnn.to_torch(validation_tensor)
    num_cores = grid_x * grid_y

    all_wrapped = True
    for core_idx in range(num_cores):
        core_x = core_idx % grid_x
        core_y = core_idx // grid_x
        core_row = validation_torch[core_idx]
        core_uint32 = np.frombuffer(core_row.contiguous().view(torch.int16).numpy().tobytes(), dtype=np.uint32)

        if np.all(core_uint32 == 0):
            continue

        core_has_mismatch = False
        for risc_idx, risc_name in enumerate(RISC_NAMES):
            before_offset = risc_idx * NUM_CBS
            after_offset = SECTION_SIZE + risc_idx * NUM_CBS
            for cb in range(NUM_CBS):
                if cb in skip_cbs:
                    continue
                before_val = int(core_uint32[before_offset + cb])
                after_val = int(core_uint32[after_offset + cb])
                if before_val != after_val:
                    if not core_has_mismatch:
                        logger.warning(f"Core ({core_x},{core_y}) — CB pointer mismatches:")
                        core_has_mismatch = True
                    logger.warning(f"  CB[{cb}] {risc_name}: BEFORE=0x{before_val:08x} AFTER=0x{after_val:08x}")
                    all_wrapped = False

        if not core_has_mismatch:
            logger.info(f"Core ({core_x},{core_y}) — all CB pointers wrapped OK")

    if all_wrapped:
        logger.info("ALL cores: CB pointers wrapped back to initial addresses!")
    else:
        logger.warning("Some CB pointers did NOT wrap — looping would cause PCC issues!")

    return all_wrapped
