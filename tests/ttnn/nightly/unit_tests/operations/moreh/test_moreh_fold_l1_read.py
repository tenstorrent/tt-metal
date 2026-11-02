# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
def test_fold_input_page_bounds(device, dtype, memory_config):
    """Run with TT_METAL_WATCHER=1 to catch reads beyond the last L1 row."""
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    values = (torch.arange(391 * 20) % 127 - 63).reshape(391, 20).to(torch_dtype)
    # Allocate the input first: a rounded DRAM read must not run past the last L1 row.
    source = ttnn.from_torch(
        values, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    result = ttnn.operations.moreh.fold(
        source, None, output_size=[4, 5], kernel_size=[1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    expected = torch.nn.functional.fold(values.unsqueeze(0), output_size=(4, 5), kernel_size=(1, 1))
    torch.testing.assert_close(ttnn.to_torch(result).reshape(expected.shape), expected, rtol=0, atol=0)
