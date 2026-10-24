# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole


@pytest.mark.skipif(not is_blackhole(), reason="Allocation layout regression for Blackhole")
@pytest.mark.parametrize("op", [ttnn.move, ttnn.reallocate], ids=["move", "reallocate"])
@pytest.mark.parametrize("rows_per_bank,gap_per_bank,reject", [(320, 1, True), (8, 1, False), (320, 330, False)])
def test_move_overlap_capacity(device, expect_error, op, rows_per_bank, gap_per_bank, reject):
    """An overlapping copy must either stage the input or reject before launching a copy."""
    # Keep the 41,600-page failing allocation layout fixed on the 110-bank P300.
    grid = device.compute_with_storage_grid_size()
    if grid.x * grid.y != 110:
        pytest.skip("This allocation layout requires 110 interleaved L1 banks")
    size_multiplier = 130

    def blank(rows):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([rows, 256]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )

    fill = []
    output = None
    try:
        while True:
            try:
                fill.append(blank(16 * size_multiplier))
            except RuntimeError as error:
                assert "Out of Memory" in str(error)
                break
        fill.sort(key=lambda tensor: tensor.buffer_address())
        freed = 0
        while freed < rows_per_bank + gap_per_bank + 16:
            ttnn.deallocate(fill.pop(0))
            freed += 16
        spacer = blank(gap_per_bank * size_multiplier)
        host = torch.randint(
            0,
            2**31 - 1,
            (rows_per_bank * size_multiplier, 256),
            generator=torch.Generator().manual_seed(7),
            dtype=torch.int32,
        )
        source = ttnn.from_torch(
            host,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        source_address = source.buffer_address()
        ttnn.deallocate(spacer)
        if reject:
            with expect_error(RuntimeError, "overlapping buffers require enough L1 staging space"):
                output = op(source)
        else:
            output = op(source)
            delta = output.buffer_address() - source_address
            assert delta > 0
            size_per_bank = ((rows_per_bank * size_multiplier + 109) // 110) * 1024
            assert (delta < size_per_bank) == (gap_per_bank == 1)
            assert torch.equal(ttnn.to_torch(output).to(torch.int64), host.to(torch.int64))
    finally:
        if output is not None:
            ttnn.deallocate(output)
        for tensor in fill:
            ttnn.deallocate(tensor)
