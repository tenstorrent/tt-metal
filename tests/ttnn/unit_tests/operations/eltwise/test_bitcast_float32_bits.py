# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "source_dtype,target_dtype",
    [
        (ttnn.uint32, ttnn.float32),
        (ttnn.int32, ttnn.float32),
        (ttnn.float32, ttnn.uint32),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_bitcast_preserves_all_float32_bits(device, source_dtype, target_dtype, layout, memory):
    words = (
        torch.tensor(
            [
                0x00000001,
                0x007FFFFF,
                0x00800001,
                0x3F800001,
                0x32B779B1,
                0xBF800123,
                0x7F7FFFFF,
                0x80000001,
            ],
            dtype=torch.int64,
        )
        .repeat(128)
        .reshape(32, 32)
    )
    signed = words.to(torch.int32)
    host = signed.view(torch.float32) if source_dtype == ttnn.float32 else signed
    source = ttnn.from_torch(host, dtype=source_dtype, layout=layout, device=device, memory_config=memory)
    output = ttnn.bitcast(source, target_dtype, memory_config=memory)
    actual = ttnn.to_torch(output)
    if target_dtype == ttnn.float32:
        actual = actual.view(torch.int32)
    actual = actual.to(torch.int64) & 0xFFFFFFFF
    assert torch.equal(actual, words), "Bitcast changed the stored 32-bit words"
