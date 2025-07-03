# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.utility_functions import is_grayskull, is_wormhole_b0, is_blackhole
from tests.ttnn.unit_tests.operations.prefetcher_common import run_prefetcher_mm

LLAMA_INPUT_SHAPES = [(2304, 1536), (1536, 2304), (2304, 3840), (2304, 3840), (3840, 2304)]
LLAMA_INPUT_DTYPES = [ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat4_b, ttnn.bfloat8_b]


@pytest.mark.skipif(is_grayskull(), reason="GS not supported")
@pytest.mark.parametrize(
    "num_reader_cores, num_tensors, input_shapes, dtypes, num_layers",
    [
        # (2, 3, [(128, 128), (128, 128 * 2), (128, 128 * 3)], [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat16], 2),
        (2, 2, [(256, 512), (256, 512)], [ttnn.bfloat4_b] * 2, 5),
        (2, 2, [(1024, 256), (1024, 256)], [ttnn.bfloat4_b] * 2, 5),
        (2, 2, [(128, 128), (128, 128)], [ttnn.bfloat4_b] * 2, 2),
        (2, 2, [(256, 1024), (256, 1024)], [ttnn.bfloat4_b] * 2, 5),
        (
            12,
            5,
            [(2304, 3840)] * 5,
            [ttnn.bfloat4_b] * 5,
            2,
        ),  # FF1/3 = 72 tiles x 120 tiles = 8640 tiles / 24 cores = 720 tiles per receiver core
        (
            1,
            4,
            [(192, 320), (192, 320), (192, 320), (192, 320)],
            [ttnn.bfloat4_b, ttnn.bfloat8_b] * 2,
            1,
        ),
        (12, 5, [(3840, 2304)] * 5, [ttnn.bfloat8_b] * 5, 5),  # FF2
        (12, 6, [(2304, 1536)] * 6, [ttnn.bfloat8_b] * 6, 5),  # QKV
        (12, 5, [(2304, 2304)] * 5, [ttnn.bfloat8_b] * 5, 5),  # DO
        (
            12,
            5,
            [(2304, 1536), (1536, 2304), (2304, 3840), (2304, 3840), (3840, 2304)],
            [ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat4_b, ttnn.bfloat8_b],
            5,
        ),  # qkv + do + ff1 + ff3 + ff2
        # Takes really long to set up
        (
            12,
            5,
            [(2048, 1280), (1280, 2048), (2048, 3584), (2048, 3584), (3584, 2048)],
            [ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat4_b, ttnn.bfloat8_b],
            80,
        ),  # qkv + do + ff1 + ff3 + ff2
    ],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((2, 2), id="2x2_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
def test_run_prefetcher(
    mesh_device,
    num_tensors,
    input_shapes,
    num_layers,
    num_reader_cores,
    dtypes,
    function_level_defaults,
):
    device = mesh_device
    # Only run these tests on unharvested TG
    device_grid = (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Skipping test_run_prefetcher because it only works with a 7x10 grid")

    run_prefetcher_mm(
        mesh_device,
        num_tensors,
        input_shapes,
        num_layers,
        num_reader_cores,
        dtypes,
    )


@pytest.mark.parametrize(
    "num_reader_cores, num_tensors, input_shapes, dtypes, num_layers, batch_weights",
    [
        (2, 2, [(128, 128), (128, 128)], [ttnn.bfloat4_b] * 2, 2, True),
        (2, 2, [(256, 1024), (256, 1024)], [ttnn.bfloat4_b] * 2, 5, True),
        (
            12,
            5,
            [(2304, 1536), (1536, 2304), (2304, 3840), (2304, 3840), (3840, 2304)],
            [ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat4_b, ttnn.bfloat8_b],
            5,
            True,
        ),  # qkv + do + ff1 + ff3 + ff2
        # Takes really long to set up
        (
            12,
            5,
            [(2048, 1280), (1280, 2048), (2048, 3584), (2048, 3584), (3584, 2048)],
            [ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat4_b, ttnn.bfloat8_b],
            80,
            True,
        ),  # qkv + do + ff1 + ff3 + ff2
    ],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((2, 2), id="2x2_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
def test_run_prefetcher_batched_weights(
    mesh_device,
    num_tensors,
    input_shapes,
    num_layers,
    batch_weights,
    num_reader_cores,
    dtypes,
    function_level_defaults,
):
    device = mesh_device
    # Only run these tests on unharvested TG
    device_grid = (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Skipping test_run_prefetcher because it only works with a 7x10 grid")

    run_prefetcher_mm(
        mesh_device,
        num_tensors,
        input_shapes,
        num_layers,
        num_reader_cores,
        dtypes,
        batch_weights=batch_weights,
    )


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 2), id="2x2_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
def test_run_prefetcher_llama_perf(
    mesh_device,
    function_level_defaults,
):
    device = mesh_device
    # Only run these tests on unharvested TG
    device_grid = (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Skipping test_run_prefetcher because it only works with a 7x10 grid")

    num_reader_cores = 12
    num_tensors = len(LLAMA_INPUT_SHAPES)
    input_shapes = LLAMA_INPUT_SHAPES
    dtypes = LLAMA_INPUT_DTYPES
    num_layers = 1

    run_prefetcher_mm(
        mesh_device,
        num_tensors,
        input_shapes,
        num_layers,
        num_reader_cores,
        dtypes,
        enable_performance_mode=True,
    )
