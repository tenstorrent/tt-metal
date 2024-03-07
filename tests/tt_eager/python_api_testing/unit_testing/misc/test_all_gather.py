# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from models.utility_functions import skip_for_grayskull, get_devices_for_t3000


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (2, 1),
        (4, 1),
        (4, 2),
        (4, 1),
        (8, 1),
    ],
)
@pytest.mark.parametrize(
    "input_shape, dim, layout",
    [
        ([4, 1, 33, 256], 0, ttl.tensor.Layout.ROW_MAJOR),
        ([4, 1, 256, 32], 0, ttl.tensor.Layout.TILE),
        ([8, 5, 13, 512], 3, ttl.tensor.Layout.ROW_MAJOR),
        ([8, 5, 32, 512], 3, ttl.tensor.Layout.TILE),
        ([8, 8, 256, 384], 0, ttl.tensor.Layout.ROW_MAJOR),
        ([8, 8, 256, 384], 0, ttl.tensor.Layout.TILE),
        ([8, 8, 256, 384], 1, ttl.tensor.Layout.ROW_MAJOR),
        ([8, 8, 256, 384], 1, ttl.tensor.Layout.TILE),
        ([8, 8, 256, 384], 2, ttl.tensor.Layout.ROW_MAJOR),
        ([8, 8, 256, 384], 2, ttl.tensor.Layout.TILE),
        ([8, 8, 256, 768], 3, ttl.tensor.Layout.ROW_MAJOR),
        ([8, 8, 256, 768], 3, ttl.tensor.Layout.TILE),
        # MLP AllGather
        ([1, 1, 32, 32768], 3, ttl.tensor.Layout.TILE),
        # Input, Selfout, Final AllGather
        ([1, 1, 32, 8192], 3, ttl.tensor.Layout.TILE),
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.L1),
    ],
)
def test_all_gather_on_t3000(
    all_devices,
    num_devices,
    input_shape,
    dim,
    num_links,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
):
    if len(all_devices) != 8:
        pytest.skip("Not T3000!")

    if mem_config == ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.DRAM):
        pytest.skip("Disabling input in DRAM for all gather tests!")

    devices = get_devices_for_t3000(all_devices, num_devices)
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")
    if (
        layout == ttl.tensor.Layout.ROW_MAJOR or num_links == 2
    ) and mem_config.buffer_type == ttl.tensor.BufferType.DRAM:
        pytest.skip("All gather tests are hanging for RM or 2 links in DRAM")

    input_tensor = torch.rand(input_shape).bfloat16()

    if num_devices < 2:
        pytest.skip("Requires multiple devices to run")
    elif num_devices == 2 and num_links == 2:
        pytest.skip("Not enough links to run")

    if input_shape[dim] % num_devices != 0 or (dim == 3 and input_shape[dim] // num_devices % 32 != 0):
        pytest.skip("Unsupported test case")

    input_tensors = torch.chunk(input_tensor, num_devices, dim)
    tt_input_tensors = []
    for i, t in enumerate(input_tensors):
        tt_input_tensors.append(
            ttl.tensor.Tensor(t, ttl.tensor.DataType.BFLOAT16).to(layout).to(devices[i], mem_config)
        )

    tt_out_tensors = ttl.tensor.all_gather(tt_input_tensors, dim, num_links, output_mem_config=mem_config)

    for i, t in enumerate(tt_out_tensors):
        tt_output_tensor = t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        eq, output = comp_equal(tt_output_tensor, input_tensor)
        assert eq, f"{i} FAILED: {output}"
