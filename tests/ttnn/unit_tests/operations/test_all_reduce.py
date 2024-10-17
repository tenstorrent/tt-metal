# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull, get_devices_for_t3000


def is_unsupported_case(input_shape, math_op, mem_config, num_devices, num_links, input_dtype, layout):
    elem_size = 2 if input_dtype == ttnn.bfloat16 else 1
    tensor_size_bytes = elem_size
    for i in input_shape:
        tensor_size_bytes *= i
    num_l1_banks = 64
    if mem_config.buffer_type == ttnn.BufferType.L1 and tensor_size_bytes > num_l1_banks * 50 * 1024:
        return True, "L1 buffer can't support large tensor sizes"

    return False, ""


def run_with_trace(
    t3k_mesh_device,
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
        num_workers=n_worker,
        num_buffers_per_channel=n_buffer,
    )
    for device_id in t3k_mesh_device.get_device_ids():
        ttnn.synchronize_device(t3k_mesh_device.get_device(device_id))

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
    for i in range(num_iters):
        output_tensor_mesh = ttnn.experimental.all_reduce(
            input_tensor_mesh,
            math_op=math_op,
            num_links=num_links,
            memory_config=output_mem_config,
            num_workers=n_worker,
            num_buffers_per_channel=n_buffer,
        )
    ttnn.end_trace_capture(t3k_mesh_device, trace_id, cq_id=0)
    for device_id in t3k_mesh_device.get_device_ids():
        ttnn.synchronize_device(t3k_mesh_device.get_device(device_id))

    # Run the op
    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(t3k_mesh_device, trace_id, blocking=False)
    ttnn.release_trace(t3k_mesh_device, trace_id)
    for device_id in t3k_mesh_device.get_device_ids():
        ttnn.synchronize_device(t3k_mesh_device.get_device(device_id))

    return output_tensor_mesh


def run_all_reduce_test(
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
    enable_async=True,
    num_iters=1,
    topology=ttnn.Topology.Ring,
):
    if len(t3k_mesh_device.get_device_ids()) < num_devices:
        pytest.skip(
            f"Not enough devices on machine to implement test case. Wanted {num_devices} but found {len(t3k_mesh_device.get_device_ids())}"
        )

    debug = False

    (is_known_failure, message) = is_unsupported_case(
        per_chip_output_shape, math_op, mem_config, num_devices, num_links, input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    t3k_mesh_device.enable_async(enable_async)
    if enable_async:
        logger.info(f"Using Async Mode for All Reduce Op Dispatch")

    logger.info(f"Per chip output shape: {per_chip_output_shape}, devices: {num_devices}")
    # Generate input tensors

    tt_input_tensors = []
    input_tensors = []
    for i in range(num_devices):
        input_tensor = torch.rand(per_chip_output_shape).bfloat16()
        tt_input_tensors.append(
            ttnn.Tensor(input_tensor, input_dtype)
            .to(layout)
            .to(t3k_mesh_device.get_device(t3k_mesh_device.get_device_ids()[i]), mem_config)
        )
        input_tensors.append(input_tensor)
    unchunked_input_tensor = torch.cat(input_tensors)

    assert len(tt_input_tensors) == num_devices

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)
    # Run the op
    for i in range(num_iters):
        output_tensor_mesh = ttnn.experimental.all_reduce(
            input_tensor_mesh,
            math_op=math_op,
            num_links=num_links,
            memory_config=mem_config,
            topology=topology,
        )

        for device_id in t3k_mesh_device.get_device_ids():
            ttnn.synchronize_device(t3k_mesh_device.get_device(device_id))
        logger.info(f"Done iteration {i}")

    tt_out_tensors = ttnn.get_device_tensors(output_tensor_mesh)
    logger.info(f"Compare")
    golden_canonical_out_tensor = torch.sum(unchunked_input_tensor, 0)

    # Compare
    mismatch = False
    for i, t in enumerate(tt_out_tensors):
        tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        eq, output = comp_pcc(tt_output_tensor, golden_canonical_out_tensor)
        mismatch = mismatch or not eq
        if not eq:
            logger.error(f"output mismatch for tensor {i}")
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
        # ttnn.bfloat8_b,
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
@pytest.mark.parametrize("enable_async", [True])
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
    enable_async,
    num_iters=1,
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
        enable_async=enable_async,
    )
