# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-kernel-time probe for g6_embedding: the four perf cases of feature_spec.LOOSE_CASES and two
decode-like production cases. Run under `scripts/run_safe_pytest.sh --profile --no-py-zones`; read
DEVICE KERNEL DURATION [ns] from the printed CSV. Each case calls the op twice (cold + warm program).
"""

import functools
import importlib

import pytest
import torch
import ttnn

from ttnn.operations.g6_embedding import g6_embedding, select_plan

QWEN_SMALL = (262144, 64)
QWEN_LARGE = (248320, 1280)


@functools.lru_cache(maxsize=2)
def host_table(weight_shape):
    generator = torch.Generator().manual_seed(0)
    return torch.randn(weight_shape, generator=generator, dtype=torch.bfloat16)


FLUX_LARGE = (131072, 2560)

# Production cases grouped by table (host_table keeps two), spanning every row-segment width the plans
# produce: 64, 128, 256, 1024, 1280 and 2560 B.
CASES = [
    pytest.param((16, 4096), QWEN_SMALL, id="16x4096_qwen64"),
    pytest.param((1, 4096), QWEN_SMALL, id="1x4096_qwen64"),
    pytest.param((1, 128), QWEN_SMALL, id="1x128_qwen64"),
    pytest.param((1, 1024), QWEN_SMALL, id="1x1024_qwen64"),
    pytest.param((1, 8), QWEN_SMALL, id="1x8_qwen64"),
    pytest.param((1, 16), QWEN_SMALL, id="1x16_qwen64"),
    pytest.param((1, 77), QWEN_SMALL, id="1x77_qwen64"),
    pytest.param((1, 1), QWEN_SMALL, id="1x1_qwen64"),
    pytest.param((16, 4096), QWEN_LARGE, id="p1_16x4096_qwen1280"),
    pytest.param((8, 4096), QWEN_LARGE, id="p2_8x4096_qwen1280"),
    pytest.param((16, 1024), QWEN_LARGE, id="p3_16x1024_qwen1280"),
    pytest.param((1, 3844), QWEN_LARGE, id="p4_1x3844_qwen1280"),
    pytest.param((16, 128), QWEN_LARGE, id="16x128_qwen1280"),
    pytest.param((1, 1024), QWEN_LARGE, id="1x1024_qwen1280"),
    pytest.param((1, 256), QWEN_LARGE, id="1x256_qwen1280"),
    pytest.param((1, 128), QWEN_LARGE, id="1x128_qwen1280"),
    pytest.param((1, 77), QWEN_LARGE, id="1x77_qwen1280"),
    pytest.param((1, 64), QWEN_LARGE, id="1x64_qwen1280"),
    pytest.param((1, 31), QWEN_LARGE, id="1x31_qwen1280"),
    pytest.param((1, 16), QWEN_LARGE, id="1x16_qwen1280"),
    pytest.param((1, 8), QWEN_LARGE, id="1x8_qwen1280"),
    pytest.param((1, 1), QWEN_LARGE, id="1x1_qwen1280"),
    pytest.param((1, 512), FLUX_LARGE, id="1x512_flux2560"),
]


@pytest.mark.parametrize("index_shape, weight_shape", CASES)
def test_device_time(device, index_shape, weight_shape):
    torch.manual_seed(0)
    index = torch.randint(0, weight_shape[0], index_shape, dtype=torch.int32)
    weight = host_table(weight_shape)
    tt_index = ttnn.from_torch(
        index, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_weight = ttnn.from_torch(
        weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    try:
        print(f"PLAN {index_shape} x {weight_shape}: {select_plan(tt_index, tt_weight)}")
        for _ in range(2):
            output = g6_embedding(tt_index, tt_weight, layout=ttnn.TILE_LAYOUT)
        ttnn.synchronize_device(device)
        actual = ttnn.to_torch(output)
        expected = torch.nn.functional.embedding(index.to(torch.int64), weight)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    finally:
        ttnn.deallocate(tt_weight)


@pytest.mark.parametrize(
    "index_shape, weight_shape",
    [
        pytest.param((16, 4096), QWEN_SMALL, id="zones_16x4096_qwen64"),
        pytest.param((16, 4096), QWEN_LARGE, id="zones_p1_16x4096_qwen1280"),
    ],
)
def test_zone_build(device, monkeypatch, index_shape, weight_shape):
    """The zone build (BB-21) compiles, runs and stays bit-exact; zones show in the device profile."""
    op_module = importlib.import_module("ttnn.operations.g6_embedding.g6_embedding")
    monkeypatch.setattr(op_module, "DEVICE_ZONES", True)
    test_device_time(device, index_shape, weight_shape)
