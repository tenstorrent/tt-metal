# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Both grid layouts (rect, L) forced through the program-descriptor override, at the op_design.md pinned shapes."""

import pytest
import ttnn

from ttnn.operations.moe_down_proj import default_compute_kernel_config, validate
from ttnn.operations.moe_down_proj.moe_down_proj_program_descriptor import HIDDEN, LAYOUTS, create_program_descriptor
from tests.ttnn.unit_tests.operations.moe_down_proj.test_moe_down_proj import LAYER3, check, make_operands

CASES = [
    ("t1", 1, (1, 0, 0, 0, 0, 0, 1, 0)),
    ("t37", 37, (20, 0, 9, 14, 1, 11, 7, 12)),
    ("t1858_mult32_plus1", 1858, (1024, 1025, 512, 513, 256, 257, 64, 65)),
    ("t4096_layer3", 4096, LAYER3),
    ("t4096_one_expert", 4096, (8192, 0, 0, 0, 0, 0, 0, 0)),
    ("t4096_zeros_between", 4096, (0, 3000, 0, 0, 2192, 0, 3000, 0)),
]


@pytest.mark.parametrize("layout_name", list(LAYOUTS))
@pytest.mark.parametrize("name, tokens, counts", CASES, ids=[case[0] for case in CASES])
def test_forced_layout(device, layout_name, name, tokens, counts):
    if LAYOUTS[layout_name](device.compute_with_storage_grid_size()) is None:
        pytest.skip(f"layout {layout_name} does not exist on this grid")
    x_dev, w_dev, counts_dev, xq, wq = make_operands(device, tokens, counts)
    validate(x_dev, w_dev, counts_dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, xq.shape[-2], HIDDEN]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    descriptor = create_program_descriptor(
        x_dev, w_dev, counts_dev, output, default_compute_kernel_config(), layout_override=layout_name
    )
    out = ttnn.generic_op([x_dev, w_dev, counts_dev, output], descriptor)
    check(out, xq, wq, counts, ttnn.L1_MEMORY_CONFIG)
