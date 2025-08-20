# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.utility_functions import is_wormhole_b0, is_blackhole, skip_for_blackhole
from tests.ttnn.unit_tests.operations.prefetcher_common import run_prefetcher_mm


@pytest.mark.parametrize(
    "num_reader_cores, num_tensors, input_shapes, dtypes, num_layers",
    [
        (2, 2, [(256, 512), (256, 512)], [ttnn.bfloat4_b] * 2, 5),
        (2, 2, [(1024, 256), (1024, 256)], [ttnn.bfloat4_b] * 2, 5),
        (2, 2, [(128, 128), (128, 128)], [ttnn.bfloat4_b] * 2, 2),
        (2, 2, [(256, 1024), (256, 2048)], [ttnn.bfloat4_b, ttnn.bfloat8_b], 1),
        (2, 3, [(256, 1024), (256, 2048), (512, 256)], [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b], 5),
        (2, 2, [(256, 1024), (128, 128)], [ttnn.bfloat4_b, ttnn.bfloat8_b], 5),
        (2, 3, [(256, 1024), (128, 128), (1024, 256)], [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b], 5),
        # Padding check
        (
            2,
            3,
            [(256 + 32, 512 + 224), (128, 128 + 64), (512 + 256, 224)],
            [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b],
            5,
        ),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 23887872}],
    indirect=True,
)
def test_run_prefetcher_post_commit(
    device,
    num_tensors,
    input_shapes,
    num_layers,
    num_reader_cores,
    dtypes,
    function_level_defaults,
):
    run_prefetcher_mm(
        device,
        num_tensors,
        input_shapes,
        num_layers,
        num_reader_cores,
        dtypes,
        is_functional_test=True,
    )


# Test DRAM Prefetcher x Matmul on T3K
@pytest.mark.parametrize(
    "num_reader_cores, num_tensors, input_shapes, dtypes, num_layers",
    [
        (2, 2, [(256, 512), (256, 512)], [ttnn.bfloat4_b] * 2, 5),
        (2, 2, [(1024, 256), (1024, 256)], [ttnn.bfloat4_b] * 2, 5),
        (2, 2, [(128, 128), (128, 128)], [ttnn.bfloat4_b] * 2, 2),
        (2, 2, [(256, 1024), (256, 2048)], [ttnn.bfloat4_b, ttnn.bfloat8_b], 1),
        (2, 3, [(256, 1024), (256, 2048), (512, 256)], [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b], 5),
        (2, 2, [(256, 1024), (128, 128)], [ttnn.bfloat4_b, ttnn.bfloat8_b], 5),
        (2, 3, [(256, 1024), (128, 128), (1024, 256)], [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b], 5),
        # Padding check
        (
            2,
            3,
            [(256 + 32, 512 + 224), (128, 128 + 64), (512 + 256, 224)],
            [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b],
            5,
        ),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 23887872}],
    indirect=True,
)
def test_run_prefetcher_post_commit_multi_device(
    mesh_device,
    num_tensors,
    input_shapes,
    num_layers,
    num_reader_cores,
    dtypes,
    function_level_defaults,
):
    if mesh_device.get_num_devices() <= 1:
        pytest.skip("This test requires multiple devices")

    run_prefetcher_mm(
        mesh_device,
        num_tensors,
        input_shapes,
        num_layers,
        num_reader_cores,
        dtypes,
        is_functional_test=True,
    )
