# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Debug checks of the bank_run_write program past validate(); kept as a record of the kernel's reach.

Each case runs the op's own descriptor over a range poisoned with a loud pattern, so an unwritten
page shows. It covers what the acceptance suite cannot reach in Phase 0:
- bfloat16 / bfloat8_b pages (2048 / 1088 B; 1088 does not divide the 16 KiB packet),
- WRITER_GRID cut below the bank count (a core walks up to 8 runs),
- a page count that is not a multiple of the bank count (banks of unequal slot counts).
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_zeros import g6_zeros_program_descriptor as pdm

_HOST_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
_BITS_VIEW = {torch.float32: torch.int32, torch.bfloat16: torch.int16}


def _poisoned_output(shape, dtype, device):
    """An output tensor on a range that held 1.0 everywhere (bfloat8_b cannot hold a bit pattern)."""
    dirty = ttnn.from_torch(
        torch.ones(shape, dtype=_HOST_DTYPE[dtype]),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    address = dirty.buffer_address()
    ttnn.deallocate(dirty)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    assert output.buffer_address() == address, "the output did not land on the poisoned range"
    return output


def _run(shape, dtype, device, grid=None):
    full_grid, num_dram_banks = pdm.device_geometry(device)
    output = _poisoned_output(shape, dtype, device)
    plan = pdm.make_plan(shape, dtype, grid or full_grid, num_dram_banks)
    ttnn.generic_op([output], pdm.create_program_descriptor(output, plan))
    readback = ttnn.to_torch(output)
    if dtype == ttnn.bfloat8_b:
        readback = readback.to(torch.float32)
    bits = readback.contiguous().view(_BITS_VIEW[readback.dtype])
    wrong = int((bits != 0).sum().item())
    assert wrong == 0, f"{wrong} elements are not +0.0 bits; plan max runs/core {max(len(c.runs) for c in plan.cores)}"
    return plan


@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param((1, 4, 256, 128), ttnn.bfloat16, id="p4-bf16"),
        pytest.param((32831, 1, 32, 256), ttnn.bfloat8_b, id="p1-bf8b"),
    ],
)
def test_dtypes_outside_phase0(shape, dtype, device):
    _run(shape, dtype, device)


@pytest.mark.parametrize("grid", [(1, 1), (3, 1), (8, 1)], ids=lambda g: f"grid{g[0]}x{g[1]}")
def test_grid_below_bank_count(grid, device):
    plan = _run((8, 12, 128, 128), ttnn.float32, device, grid=grid)
    assert plan.cores_used == grid[0] * grid[1]


@pytest.mark.parametrize("grid", [None, (2, 1)], ids=["full-grid", "grid2x1"])
def test_unequal_banks(grid, device):
    # 13 pages: banks 0..4 hold 2 slots, banks 5..7 hold 1.
    plan = _run((1, 1, 32, 13 * 32), ttnn.float32, device, grid=grid)
    assert plan.tensor_pages == 13
