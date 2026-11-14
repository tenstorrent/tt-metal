# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("channels", [40, 64])
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
def test_upsample_input_page_bounds(device, channels, memory_config):
    """Run with TT_METAL_WATCHER=1 to catch reads beyond the last L1 row."""
    values = (torch.arange(56 * 7 * channels) % 127 - 63).reshape(1, 56, 7, channels).to(torch.bfloat16)
    # Allocate the source before any other L1 tensor so its last bank row is at the L1 boundary.
    source = ttnn.from_torch(
        values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    result = ttnn.upsample(source, scale_factor=(2, 2), mode="nearest", memory_config=ttnn.DRAM_MEMORY_CONFIG)
    expected = values.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    torch.testing.assert_close(ttnn.to_torch(result), expected, rtol=0, atol=0)
