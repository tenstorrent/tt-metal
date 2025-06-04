# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
import math
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.utility_functions import skip_for_grayskull

TILE_HEIGHT = 32
TILE_WIDTH = 32


def is_unsupported_case(input_shape, math_op, mem_config, num_devices, num_links, input_dtype, layout):
    elem_size = 2 if input_dtype == ttnn.bfloat16 else 1
    tensor_size_bytes = elem_size
    for i in input_shape:
        tensor_size_bytes *= i
    num_l1_banks = 64
    if mem_config.buffer_type == ttnn.BufferType.L1 and tensor_size_bytes > num_l1_banks * 50 * 1024:
        return True, "L1 buffer can't support large tensor sizes"
    if (input_shape[-2] == TILE_HEIGHT or input_shape[-1] == TILE_WIDTH) and input_dtype == ttnn.bfloat8_b:
        return True, "This combination is not supported for now"

    return False, ""


def run_with_trace(
    mesh_device,
    input_tensor_mesh,
    num_links,
    math_op,
    output_mem_config,
    n_worker,
    n_buffer,
    num_iters,
):
    # Compile Run
    logger.info("Compiling model")
    output_tensor_mesh = ttnn.experimental.all_reduce(
        input_tensor_mesh,
        math_op=math_op,
        num_links=num_links,
        memory_config=output_mem_config,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iters):
        output_tensor_mesh = ttnn.experimental.all_reduce(
            input_tensor_mesh,
            math_op=math_op,
            num_links=num_links,
            memory_config=output_mem_config,
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Run the op
    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)

    return output_tensor_mesh


def run_all_reduce_test(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    num_iters=1,
    topology=ttnn.Topology.Ring,
):
    if len(mesh_device.get_device_ids()) < num_devices:
        pytest.skip(
            f"Not enough devices on machine to implement test case. Wanted {num_devices} but found {len(mesh_device.get_device_ids())}"
        )

    debug = False

    (is_known_failure, message) = is_unsupported_case(
        per_chip_output_shape, math_op, mem_config, num_devices, num_links, input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    logger.info(f"Per chip output shape: {per_chip_output_shape}, devices: {num_devices}")
    # Generate input tensors

    tt_input_tensors = []
    input_tensors = []

    numel = math.prod(per_chip_output_shape)
    if debug:
        input_tensors[-1] = torch.arange(numel).reshape(per_chip_output_shape).bfloat16()
    for i in range(num_devices):
        input_tensor = torch.rand(per_chip_output_shape).bfloat16()
        t = ttnn.from_torch(input_tensor, input_dtype, layout=layout)
        tt_input_tensors.append(t)
        input_tensor = input_tensor.view(1, -1, input_tensor.shape[2], input_tensor.shape[3])
        input_tensors.append(input_tensor)

    unchunked_input_tensor = torch.cat(input_tensors)

    assert len(tt_input_tensors) == num_devices

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors).to(mesh_device, mem_config)
    # Run the op
    for i in range(num_iters):
        output_tensor_mesh = ttnn.experimental.all_reduce(
            input_tensor_mesh,
            math_op=math_op,
            num_links=num_links,
            memory_config=mem_config,
            topology=topology,
        )
        ttnn.synchronize_device(mesh_device)
        logger.info(f"Done iteration {i}")

    tt_out_tensors = ttnn.get_device_tensors(output_tensor_mesh)
    logger.info(f"Compare")
    golden_canonical_out_tensor = torch.sum(unchunked_input_tensor, 0, keepdim=True)
    golden_canonical_out_tensor = golden_canonical_out_tensor.view(per_chip_output_shape)
    # Compare
    mismatch = False
    for i, t in enumerate(tt_out_tensors):
        tt_output_tensor = ttnn.to_torch(t)

        eq, output = comp_pcc(tt_output_tensor, golden_canonical_out_tensor)
        mismatch = mismatch or not eq
        if not eq:
            logger.error(f"output mismatch for tensor {i}. Mesh device ID: {mesh_device.get_device_ids()[i]}")
            if debug:
                for w in range(tt_output_tensor.shape[0]):
                    for z in range(tt_output_tensor.shape[1]):
                        for y in range(tt_output_tensor.shape[2]):
                            for x in range(tt_output_tensor.shape[3]):
                                if tt_output_tensor[w, z, y, x] != golden_canonical_out_tensor[w, z, y, x]:
                                    logger.error(
                                        f"mismatch at {w}, {z}, {y}, {x}: {tt_output_tensor[w, z, y, x]} != {golden_canonical_out_tensor[w, z, y, x]}"
                                    )

        else:
            logger.info(f"output match for tensor {i}")
    assert not mismatch, f"{i} FAILED: {output}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (8, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape",
    [
        ([1, 1, 32, 4096]),
        ([1, 1, 32, 8192]),
        ([1, 1, 32, 1024]),
        ([1, 1, 32, 2048]),
        ([1, 1, 4096, 32]),
        ([1, 1, 8192, 32]),
        ([1, 1, 1024, 32]),
        ([1, 1, 2048, 32]),
        # ([4, 1, 32, 4096]), # Skipped as shape unsupported by L1 memory (OOM issue)
        ([8, 1, 32, 1024]),
        ([1, 4, 1024, 32]),
        # ([2, 4, 2048, 32]), # Skipped as shape unsupported by L1 memory (OOM issue)
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
def test_ring_all_reduce_post_commit(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    num_iters=2,
):
    run_all_reduce_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (2, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape",
    [
        ([2, 2, 64, 64]),
        ([1, 1, 64, 64]),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
    ],
)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
def test_ring_all_reduce_post_commit_2chip(
    pcie_mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    num_iters=2,
):
    run_all_reduce_test(
        pcie_mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
        topology=ttnn.Topology.Linear,
    )
