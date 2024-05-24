# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull, get_devices_for_t3000
import itertools


def is_unsupported_case(input_shape, scatter_dim, math_op, mem_config, num_devices, num_links, input_dtype, layout):
    if scatter_dim != 3:
        return True, "Only support for scatter_dim=3 is tested so far"

    return False, ""


def run_reduce_scatter_test(
    all_devices,
    num_devices,
    per_chip_output_shape,
    scatter_dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    num_iters=1,
):
    if len(all_devices) != 8:
        pytest.skip("Not T3000!")

    # if num_devices != 4:
    #     pytest.skip("Only testing for 4 devices")

    debug = False
    logger.info(f"num_devices: {num_devices}")
    logger.info(f"per_chip_output_shape: {per_chip_output_shape}")
    logger.info(f"scatter_dim: {scatter_dim}")
    logger.info(f"num_links: {num_links}")
    logger.info(f"math_op: {math_op}")
    logger.info(f"input_dtype: {input_dtype}")
    logger.info(f"layout: {layout}")
    logger.info(f"mem_config: {mem_config}")

    (is_known_failure, message) = is_unsupported_case(
        per_chip_output_shape, scatter_dim, math_op, mem_config, num_devices, num_links, input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")
    devices = get_devices_for_t3000(all_devices, num_devices)

    # Generate input tensors
    canonical_input_shape = per_chip_output_shape.copy()
    canonical_input_shape[scatter_dim] *= num_devices
    logger.info(f"per_chip_output_shape: {per_chip_output_shape}")
    logger.info(f"canonical_input_tensor_shape: {canonical_input_shape}")
    tt_input_tensors = []

    numel = canonical_input_shape[0] * canonical_input_shape[1] * canonical_input_shape[2] * canonical_input_shape[3]
    input_tensors = [
        # torch.rand(canonical_input_shape).bfloat16() if not debug else torch.arange(numel).reshape(canonical_input_shape).bfloat16()
        torch.rand(canonical_input_shape).bfloat16() if not debug else torch.ones(canonical_input_shape).bfloat16()
        for _ in range(num_devices)
    ]
    if debug:
        input_tensors[-1] = torch.arange(numel).reshape(canonical_input_shape).bfloat16()
    for i, canonical_input_tensor in enumerate(input_tensors):
        logger.info(f"input_tensor[{i}].shape: {canonical_input_tensor.data.shape}")
        logger.info(f"input_tensor[{i}]: {canonical_input_tensor.data}")
        tt_input_tensors.append(
            ttl.tensor.Tensor(canonical_input_tensor, input_dtype).to(layout).to(devices[i], mem_config)
        )

    # Run the op
    # for i in range(num_iters):
    tt_out_tensors = ttl.tensor.reduce_scatter(
        tt_input_tensors,
        scatter_split_dim=scatter_dim,
        reduce_op=math_op,
        num_links=num_links,
        output_mem_config=mem_config,
    )

    for d in devices:
        ttl.device.Synchronize(d)
    logger.info(f"Done iteration {i}")

    # Compute golden
    # TODO: Make it model how reduce scatter actually works for numerical correctness/ordering
    golden_canonical_out_tensor = torch.zeros(canonical_input_shape).bfloat16()
    logger.info(f"golden_canonical_out_tensor shape: {golden_canonical_out_tensor.shape}")
    logger.info(f"canonical_input_shape: {canonical_input_shape}")
    for i, t in enumerate(input_tensors):
        logger.info(f"t shape: {t.shape}")
        golden_canonical_out_tensor = torch.add(golden_canonical_out_tensor, t).bfloat16()
        logger.info(f"golden_canonical_out_tensor[{i}]: {golden_canonical_out_tensor.data}")

    logger.info(f"golden_canonical_out_tensor: {golden_canonical_out_tensor.data}")
    golden_output_tensors = torch.chunk(golden_canonical_out_tensor, num_devices, scatter_dim)

    logger.info(f"Compare")
    # Compare
    assert len(golden_output_tensors) == len(tt_out_tensors)
    mismatch = False
    for i, t in enumerate(tt_out_tensors):
        tt_output_tensor = t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        logger.info(f"golden_output_tensors[i].shape: {golden_output_tensors[i].shape}")
        logger.info(f"tt_output_tensor.shape: {tt_output_tensor.shape}")
        eq, output = comp_pcc(tt_output_tensor, golden_output_tensors[i])
        mismatch = mismatch or not eq
        if not eq:
            logger.error(f"output mismatch for tensor {i}")
        else:
            logger.info(f"output match for tensor {i}")
        assert not mismatch, f"{i} FAILED: {output}"


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 1),
        (8, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, scatter_dim, layout",
    [
        ([1, 1, 32, 32], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 32, 64], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 64, 64], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 32, 128], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 32, 256], 3, ttl.tensor.Layout.TILE),
        # Hangs... for whatever reason. Seems like a noc sem inc from sender -> EDM gets lost
        #          somehow at some point
        # ([1, 1, 32, 512], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 32, 1024], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 32, 2048], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 128, 1024], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 128, 8192], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 2048, 1024], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 2048, 8192], 3, ttl.tensor.Layout.TILE),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.L1),
    ],
)
@pytest.mark.parametrize("math_op", [ttl.tensor.ReduceOpMath.SUM])
def test_reduce_scatter_post_commit(
    all_devices,
    num_devices,
    per_chip_output_shape,
    scatter_dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    num_iters=1,
):
    run_reduce_scatter_test(
        all_devices,
        num_devices,
        per_chip_output_shape,
        scatter_dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        num_iters,
    )
