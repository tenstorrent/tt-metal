# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
#  SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import skip_with_llk_assert
from models.demos.deepseek_v3.tests.unit.utils import random_torch_tensor, run_test
from models.demos.deepseek_v3.utils.config_helpers import get_fabric_config
from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_with_llk_assert("Hit LLK_ASSERT for unpacker data format conversion. Issue: #41024")
@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize(
    "shape, dtype, mem_config",
    [
        ([1, 1, 32, 256], ttnn.float32, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 32, 256], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 8, 128, 7168], ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
        ([8, 1, 32, 7168], ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 0, "fabric_config": get_fabric_config()}], indirect=True
)
def test_tilize(mesh_device, shape, dtype, mem_config, layout, enable_trace):
    torch_input = random_torch_tensor(dtype, shape)
    torch_output = torch_input

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        return ttnn.tilize(tt_input)

    def check_op(tt_output):
        assert_with_pcc(torch_output, tt_output, 0.9999)

    run_test(mesh_device, run_op, check_op, enable_trace)


@skip_with_llk_assert("Hit LLK_ASSERT for unpacker data format conversion. Issue: #41024")
@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize(
    "in_shape, out_shape, pad_val, dtype, mem_config",
    [
        ([1, 1, 32, 8], [1, 1, 32, 32], 0, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 0, "fabric_config": get_fabric_config()}], indirect=True
)
def test_tilize_with_val_padding(mesh_device, in_shape, out_shape, pad_val, dtype, mem_config, layout, enable_trace):
    torch_input = random_torch_tensor(dtype, in_shape)
    torch_output = torch_input

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        return ttnn.tilize_with_val_padding(tt_input, out_shape, pad_val)

    def check_op(tt_output):
        assert_with_pcc(torch_output, tt_output, 0.9999)

    run_test(mesh_device, run_op, check_op, enable_trace)


@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize(
    "shape, input_dtype, output_dtype, mem_config",
    [
        ([1, 1, 800, 7 * 1024], ttnn.fp8_e4m3, ttnn.bfloat8_b, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 1600, 7 * 1024], ttnn.fp8_e4m3, ttnn.bfloat8_b, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 2048, 7 * 1024], ttnn.fp8_e4m3, ttnn.bfloat8_b, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 3200, 7 * 1024], ttnn.fp8_e4m3, ttnn.bfloat8_b, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 4096, 7 * 1024], ttnn.fp8_e4m3, ttnn.bfloat8_b, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 25600, 7 * 1024], ttnn.fp8_e4m3, ttnn.bfloat8_b, ttnn.DRAM_MEMORY_CONFIG),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 11000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_tilize_fp8(mesh_device, shape, input_dtype, output_dtype, mem_config, layout, enable_trace):
    if mesh_device.arch() != ttnn.device.Arch.BLACKHOLE:
        pytest.skip("FP8 tilize is only supported on Blackhole devices")

    # Create FP8 input tensor in PyTorch
    torch_input = random_torch_tensor(input_dtype, shape)
    torch_output = torch_input

    # First create as bfloat16 with ROW_MAJOR layout, then convert to FP8
    # This works around the limitation of creating FP8 directly with ROW_MAJOR
    tt_input_bf16 = ttnn.from_torch(
        torch_input.to(torch.bfloat16),  # Convert to bfloat16 first
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=layout,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Convert to FP8 format
    tt_input = ttnn.typecast(tt_input_bf16, dtype=input_dtype)

    def run_op():
        # Tilize and convert to bfloat8_b output
        return ttnn.tilize(tt_input, dtype=output_dtype)

    def check_op(tt_output):
        # For FP8 to BFP8 conversion, we may need a lower PCC threshold
        assert_with_pcc(torch_output, tt_output, 0.99)

    run_test(mesh_device, run_op, check_op, enable_trace)
