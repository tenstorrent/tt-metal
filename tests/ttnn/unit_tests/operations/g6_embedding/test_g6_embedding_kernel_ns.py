# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device kernel time of the production perf cases (p1..p4) and the most frequent decode calls.

Run under `scripts/run_safe_pytest.sh --profile --no-py-zones`; the CSV holds one g6_embedding row
per case in parametrization order. Every result is still checked bit for bit.
"""

import functools

import pytest
import torch
import torch.nn.functional as F
import ttnn

from ttnn.operations.g6_embedding import g6_embedding, select_plan

QWEN_SMALL = (262144, 64)
QWEN_LARGE = (248320, 1280)
FLUX_LARGE = (131072, 2560)

CASES = [
    pytest.param((16, 4096), QWEN_LARGE, id="p1_16x4096_qwen1280"),
    pytest.param((8, 4096), QWEN_LARGE, id="p2_8x4096_qwen1280"),
    pytest.param((16, 1024), QWEN_LARGE, id="p3_16x1024_qwen1280"),
    pytest.param((1, 3844), QWEN_LARGE, id="p4_1x3844_qwen1280"),
    pytest.param((1, 4096), QWEN_LARGE, id="1x4096_qwen1280"),
    pytest.param((1, 128), QWEN_LARGE, id="1x128_qwen1280"),
    pytest.param((1, 8), QWEN_LARGE, id="1x8_qwen1280"),
    pytest.param((1, 256), QWEN_LARGE, id="1x256_qwen1280"),
    pytest.param((1, 1024), QWEN_LARGE, id="1x1024_qwen1280"),
    pytest.param((1, 77), QWEN_LARGE, id="1x77_qwen1280"),
    pytest.param((16, 4096), QWEN_SMALL, id="16x4096_qwen64"),
    pytest.param((1, 4096), QWEN_SMALL, id="1x4096_qwen64"),
    pytest.param((1, 1), QWEN_SMALL, id="1x1_qwen64"),
    pytest.param((1, 512), FLUX_LARGE, id="1x512_flux2560"),
]


@functools.lru_cache(maxsize=1)
def _host_table(weight_shape):
    generator = torch.Generator().manual_seed(42)
    return torch.randn(weight_shape, generator=generator, dtype=torch.bfloat16)


_device_tables = {}


def _device_table(device, weight_shape):
    key = (id(device), weight_shape)
    if key not in _device_tables:
        for old in list(_device_tables):
            ttnn.deallocate(_device_tables.pop(old))
        _device_tables[key] = ttnn.from_torch(
            _host_table(weight_shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return _device_tables[key]


@pytest.mark.parametrize("index_shape, weight_shape", CASES)
def test_kernel_ns(device, index_shape, weight_shape):
    torch.manual_seed(0)
    index = torch.randint(0, weight_shape[0], index_shape, dtype=torch.int32)
    tt_weight = _device_table(device, weight_shape)
    tt_index = ttnn.from_torch(
        index, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    print(f"{index_shape} x {weight_shape}: {select_plan(tt_index, tt_weight)}")
    output = g6_embedding(tt_index, tt_weight, layout=ttnn.TILE_LAYOUT)
    expected = F.embedding(index.to(torch.int64), _host_table(weight_shape))
    actual = ttnn.to_torch(output)
    assert torch.equal(actual.contiguous().view(torch.int16), expected.contiguous().view(torch.int16))
