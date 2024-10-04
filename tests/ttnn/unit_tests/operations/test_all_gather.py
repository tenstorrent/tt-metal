# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
import itertools
from ttnn import ShardTensorToMesh


def is_unsupported_case(input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout):
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        return True, "Invalid combination"

    if input_shape[dim] % num_devices != 0 or (dim == 3 and input_shape[dim] // num_devices % 32 != 0):
        return True, "Unsupported test case"

    ## Check that we can readback results
    fast_dispatch_page_size_limit = 55 * 1024
    elem_size = 2 if input_dtype == ttnn.bfloat16 else 1
    if layout == ttnn.ROW_MAJOR_LAYOUT and (input_shape[dim] * elem_size) > fast_dispatch_page_size_limit:
        # Fast dispatch currently can't breakup readback of large pages into multiple smaller pages and is
        # limited to ~55K pages.
        return True, "Fast dispatch can't support reading back this page size in one shot"

    # Check that we can fit in L1 (if L1 config)
    tensor_size_bytes = elem_size
    for i in input_shape:
        tensor_size_bytes *= i
    num_l1_banks = 64
    if mem_config.buffer_type == ttnn.BufferType.L1 and tensor_size_bytes > num_l1_banks * 50 * 1024:
        return True, "L1 buffer can't support large tensor sizes"

    # Check that each chip has a non-zero amount of data available
    min_sized_chunks_on_dim = input_shape[dim]
    if dim == 3:
        min_sized_chunks_on_dim //= 32
    if dim == 2:
        if layout == ttnn.TILE_LAYOUT:
            min_sized_chunks_on_dim //= 32
    if min_sized_chunks_on_dim < num_devices:
        return (
            True,
            f"Input shape {input_shape} incompatible with {num_devices} on dim {dim} because some chips will have no tensor",
        )

    if input_shape == [8, 8, 256, 384] and dim == 1 and layout == ttnn.TILE_LAYOUT and input_dtype == ttnn.bfloat8_b:
        return True, "Known failure"

    return False, ""


def is_unsupported_case_t3k(input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout):
    if num_devices < 2:
        return True, "Requires multiple devices to run"
    elif num_devices == 2 and num_links <= 2:
        return True, "Not enough links to run"

    return is_unsupported_case(input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout)


def is_unsupported_case_n300(input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout):
    return is_unsupported_case(input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout)


def run_with_trace(
    mesh_device,
    devices,
    all_gather_topology,
    input_tensor_mesh,
    dim,
    num_links,
    output_mem_config,
    n_worker,
    n_buffer,
    num_iter,
):
    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.all_gather(
        input_tensor_mesh,
        dim,
        num_links=num_links,
        memory_config=output_mem_config,
        num_workers=n_worker,
        num_buffers_per_channel=n_buffer,
        topology=all_gather_topology,
    )
    for d in mesh_device.get_devices():
        ttnn.synchronize_device(d)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        tt_out_tensor = ttnn.all_gather(
            input_tensor_mesh,
            dim,
            num_links=num_links,
            memory_config=output_mem_config,
            num_workers=n_worker,
            num_buffers_per_channel=n_buffer,
            topology=all_gather_topology,
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    for d in mesh_device.get_devices():
        ttnn.synchronize_device(d)

    # Run the op
    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    for d in mesh_device.get_devices():
        ttnn.synchronize_device(d)

    return tt_out_tensor


def run_all_gather_impl(
    mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    all_gather_topology,
    num_iters=1,
    enable_async=False,
):
    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")
    # Use Async mode based on test input config
    mesh_device.enable_async(enable_async)

    if enable_async:
        logger.info(f"Using Async Mode for All Gather Op Dispatch")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    input_tensor = torch.rand(input_shape).bfloat16()

    input_tensors = torch.chunk(input_tensor, num_devices, dim)
    tt_input_tensors = []
    for i, t in enumerate(input_tensors):
        tt_input_tensors.append(ttnn.Tensor(t, input_dtype).to(layout).to(mesh_device.get_devices()[i], mem_config))

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)
    for i in range(num_iters):
        tt_out_tensor = ttnn.all_gather(
            input_tensor_mesh, dim, num_links=num_links, memory_config=mem_config, topology=all_gather_topology
        )

        for d in mesh_device.get_devices():
            ttnn.synchronize_device(d)
        logger.info(f"Done iteration {i}")

    for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
        tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if input_dtype == ttnn.bfloat16:
            eq, output = comp_equal(tt_output_tensor, input_tensor)
        else:
            eq, output = comp_pcc(tt_output_tensor, input_tensor)
        if not eq:
            logger.error(f"output mismatch for tensor {i}")
        assert eq, f"{i} FAILED: {output}"


def run_all_gather_on_n300_impl(
    mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    all_gather_topology,
    num_iters=1,
    enable_async=False,
):
    if mesh_device.get_num_devices() != 2:
        pytest.skip("Not N300!")

    (is_known_failure, message) = is_unsupported_case_n300(
        input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    return run_all_gather_impl(
        mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=all_gather_topology,
        num_iters=num_iters,
        enable_async=enable_async,
    )


def run_all_gather_on_t3000_impl(
    mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    all_gather_topology,
    num_iters=1,
    enable_async=False,
):
    if mesh_device.get_num_devices() < num_devices:
        pytest.skip("Not T3000!")

    (is_known_failure, message) = is_unsupported_case_t3k(
        input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    return run_all_gather_impl(
        mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=all_gather_topology,
        num_iters=num_iters,
        enable_async=enable_async,
    )


def run_all_gather_on_t3000_impl_tight_loop(
    mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    all_gather_topology,
    num_iters,
    enable_async=False,
):
    run_all_gather_on_t3000_impl(
        mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=all_gather_topology,
        num_iters=num_iters,
        enable_async=enable_async,
    )


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        # (4, 2, [4, 1, 256, 32], 0, ttnn.TILE_LAYOUT),        # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [8, 1, 256, 32], 0, ttnn.TILE_LAYOUT),        # https://github.com/tenstorrent/tt-metal/issues/9686
        (8, 1, [1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),
        # (4, 2, [1, 1, 32, 32768], 3, ttnn.TILE_LAYOUT),      # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 2, [4, 1, 256, 32], 0, ttnn.ROW_MAJOR_LAYOUT),   # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [8, 1, 256, 32], 0, ttnn.ROW_MAJOR_LAYOUT),   # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [1, 1, 32, 16384], 3, ttnn.ROW_MAJOR_LAYOUT), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 2, [1, 1, 32, 32768], 3, ttnn.ROW_MAJOR_LAYOUT), # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,        # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),  # https://github.com/tenstorrent/tt-metal/issues/9686
        # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("num_iters", [1000])  # restore to 500: https://github.com/tenstorrent/tt-metal/issues/9686
@pytest.mark.parametrize("enable_async", [True])
def test_all_gather_on_t3000_post_commit_looping(
    t3k_mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_on_t3000_impl_tight_loop(
        t3k_mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
        num_iters=num_iters,
        enable_async=enable_async,
    )


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        (8, 1, [8, 1, 256, 32], 0, ttnn.TILE_LAYOUT),
        (8, 1, [1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),
        (8, 1, [8, 1, 256, 32], 0, ttnn.ROW_MAJOR_LAYOUT),
        (8, 1, [1, 1, 32, 16384], 3, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,        # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),        # https://github.com/tenstorrent/tt-metal/issues/9686
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("num_iters", [1000])  # TODO: restore to 500
@pytest.mark.parametrize("enable_async", [True, False])
def test_all_gather_on_t3000_nightly_commit_looping(
    t3k_mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_on_t3000_impl_tight_loop(
        t3k_mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
        num_iters=num_iters,
        enable_async=enable_async,
    )


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        (4, 2, [4, 1, 256, 32], 0, ttnn.TILE_LAYOUT),
        (4, 2, [1, 1, 32, 32768], 3, ttnn.TILE_LAYOUT),
        (4, 2, [4, 1, 256, 32], 0, ttnn.ROW_MAJOR_LAYOUT),
        (4, 2, [1, 1, 32, 32768], 3, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,        # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),        # https://github.com/tenstorrent/tt-metal/issues/9686
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("num_iters", [1000])  # TODO: restore to 500
@pytest.mark.parametrize("enable_async", [True, False])
def test_all_gather_on_t3000_nightly_commit_looping_4chip_ring(
    pcie_mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_on_t3000_impl_tight_loop(
        pcie_mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
        num_iters=num_iters,
        enable_async=enable_async,
    )


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        (8, 1, [8, 1, 33, 256], 0, ttnn.ROW_MAJOR_LAYOUT),
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
def test_all_gather_on_t3000_post_commit_for_profiler_regression(
    t3k_mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
):
    run_all_gather_on_t3000_impl(
        t3k_mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
    )


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        (8, 1, [8, 1, 33, 256], 0, ttnn.ROW_MAJOR_LAYOUT),  # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [8, 8, 256, 384], 1, ttnn.ROW_MAJOR_LAYOUT),           # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [8, 8, 256, 384], 1, ttnn.TILE_LAYOUT),           # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [8, 5, 13, 512], 3, ttnn.ROW_MAJOR_LAYOUT),           # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [8, 5, 32, 512], 3, ttnn.TILE_LAYOUT),
        # Only for BFP8B
        # # ([1, 1, 640, 32768], 3, ttnn.TILE_LAYOUT),        # https://github.com/tenstorrent/tt-metal/issues/9686
        # # MLP AllGather,  Llama 2 decode attn, mlp. Llama2, Falcon 40B decode mlp attn
        # (8, 1, [1, 1, 32, 32768], 3, ttnn.TILE_LAYOUT),        # https://github.com/tenstorrent/tt-metal/issues/9686
        # # (4, 2, [1, 1, 32, 32768], 3, ttnn.TILE_LAYOUT),        # https://github.com/tenstorrent/tt-metal/issues/9686
        # # (8, 1, [1, 1, 32, 32768], 3, ttnn.ROW_MAJOR_LAYOUT),        # https://github.com/tenstorrent/tt-metal/issues/9686
        # # Input, Selfout, Final AllGather,  Llama2, Falcon 40B decode mlp attn
        # (8, 1, [1, 1, 32, 8192], 3, ttnn.TILE_LAYOUT),        # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [1, 1, 32, 8192], 3, ttnn.ROW_MAJOR_LAYOUT),        # https://github.com/tenstorrent/tt-metal/issues/9686
        # # Falcon 40B prefill
        # # 8 chips
        # (8, 1, [1, 1, 2048, 8192], 3, ttnn.TILE_LAYOUT),          # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [1, 1, 2048, 8192], 3, ttnn.ROW_MAJOR_LAYOUT),          # https://github.com/tenstorrent/tt-metal/issues/9686
        # # Falcon 40B prefill, also mixtral expert reduction (w/ zero filled tensor)
        # # 8 chips
        # (8, 1, [1, 1, 2048, 32768], 3, ttnn.TILE_LAYOUT),  # https://github.com/tenstorrent/tt-metal/issues/9686
        # # Llama/falcon40B galaxy mlp weights stationary -> emulation of row/col reduce
        # (8, 1, [1, 1, 256, 1024], 2, ttnn.TILE_LAYOUT),          # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [1, 1, 246, 4096], 2, ttnn.ROW_MAJOR_LAYOUT),          # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [1, 1, 246, 4096], 2, ttnn.TILE_LAYOUT),          # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [1, 1, 8192, 32], 2, ttnn.TILE_LAYOUT),          # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [1, 1, 1024, 256], 3, ttnn.TILE_LAYOUT),          # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [1, 1, 256, 2048], 2, ttnn.TILE_LAYOUT),          # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [1, 1, 256, 8192], 2, ttnn.TILE_LAYOUT),  # double on reduction dim for 8 chip          # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [8, 1, 256, 32], 0, ttnn.TILE_LAYOUT),          # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [8, 8, 128, 4096], 1, ttnn.TILE_LAYOUT),          # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,          # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),  # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
def test_all_gather_on_t3000_post_commit(
    t3k_mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
):
    run_all_gather_on_t3000_impl(
        t3k_mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
    )


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        (4, 2, [4, 1, 33, 256], 0, ttnn.ROW_MAJOR_LAYOUT),  # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 2, [8, 8, 256, 384], 1, ttnn.ROW_MAJOR_LAYOUT),           # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 2, [8, 8, 256, 384], 1, ttnn.TILE_LAYOUT),           # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 2, [8, 5, 13, 384], 3, ttnn.ROW_MAJOR_LAYOUT),           # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 2, [8, 5, 32, 384], 3, ttnn.TILE_LAYOUT),           # https://github.com/tenstorrent/tt-metal/issues/968
        # (4, 2, [1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),        # https://github.com/tenstorrent/tt-metal/issues/9686
        # # (4, 2, [1, 1, 32, 32768], 3, ttnn.TILE_LAYOUT),        # https://github.com/tenstorrent/tt-metal/issues/9686
        # # Input, Selfout, Final AllGather,  Llama2, Falcon 40B decode mlp attn
        # (4, 2, [1, 1, 32, 8192], 3, ttnn.TILE_LAYOUT),        # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,          # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),  # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
def test_all_gather_on_t3000_post_commit_4chip_ring(
    pcie_mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
):
    run_all_gather_on_t3000_impl(
        pcie_mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
    )


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        # (4, 2, [1, 4, 32, 3584], 1, ttnn.TILE_LAYOUT),
        (8, 1, [1, 8, 32, 2048], 1, ttnn.TILE_LAYOUT),
        (8, 1, [1, 8, 32, 4096], 1, ttnn.TILE_LAYOUT),
        # (4, 1, [4, 1, 33, 256], 0, ttnn.ROW_MAJOR_LAYOUT), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [8, 1, 33, 256], 0, ttnn.ROW_MAJOR_LAYOUT), # https://github.com/tenstorrent/tt-metal/issues/9686
        # # (8, 1, [8, 1, 256, 32], 0, ttnn.TILE_LAYOUT),
        # (8, 1, [8, 8, 256, 384], 1, ttnn.ROW_MAJOR_LAYOUT), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 2, [8, 8, 256, 384], 1, ttnn.TILE_LAYOUT), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [8, 8, 256, 384], 1, ttnn.TILE_LAYOUT), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 1, [8, 5, 13, 384], 3, ttnn.ROW_MAJOR_LAYOUT), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [8, 5, 13, 512], 3, ttnn.ROW_MAJOR_LAYOUT), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 1, [8, 5, 32, 384], 3, ttnn.TILE_LAYOUT), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [8, 5, 32, 512], 3, ttnn.TILE_LAYOUT), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 1, [1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,  # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),  # https://github.com/tenstorrent/tt-metal/issues/9686
        # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("enable_async", [True, False])
def test_line_all_gather_on_t3000_post_commit(
    t3k_mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=1,
):
    if t3k_mesh_device.get_num_devices() < num_devices:
        pytest.skip("Not T3000!")

    run_all_gather_on_t3000_impl(
        t3k_mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Linear,
        enable_async=enable_async,
        num_iters=num_iters,
    )


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        # (4, 2, [1, 4, 32, 3584], 1, ttnn.TILE_LAYOUT),
        (4, 2, [1, 4, 32, 3584], 1, ttnn.TILE_LAYOUT),
        # (4, 1, [4, 1, 33, 256], 0, ttnn.ROW_MAJOR_LAYOUT), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 2, [8, 8, 256, 384], 1, ttnn.TILE_LAYOUT), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 1, [8, 5, 13, 384], 3, ttnn.ROW_MAJOR_LAYOUT), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 1, [8, 5, 32, 384], 3, ttnn.TILE_LAYOUT), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 1, [1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,  # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),  # https://github.com/tenstorrent/tt-metal/issues/9686
        # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("enable_async", [True, False])
def test_line_all_gather_on_t3000_post_commit_4chip_ring(
    pcie_mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=1,
):
    if pcie_mesh_device.get_num_devices() < num_devices:
        pytest.skip("Not T3000!")

    run_all_gather_on_t3000_impl(
        pcie_mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Linear,
        enable_async=enable_async,
        num_iters=num_iters,
    )


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        (4, 1, [4, 1, 33, 256], 0, ttnn.ROW_MAJOR_LAYOUT),
        (8, 1, [8, 1, 33, 256], 0, ttnn.ROW_MAJOR_LAYOUT),
        # (8, 1, [8, 1, 256, 32], 0, ttnn.TILE_LAYOUT),
        (8, 1, [8, 8, 256, 384], 1, ttnn.ROW_MAJOR_LAYOUT),
        # (4, 2, [8, 8, 256, 384], 1, ttnn.TILE_LAYOUT),
        (8, 1, [8, 8, 256, 384], 1, ttnn.TILE_LAYOUT),
        (4, 1, [8, 5, 13, 384], 3, ttnn.ROW_MAJOR_LAYOUT),
        (8, 1, [8, 5, 13, 512], 3, ttnn.ROW_MAJOR_LAYOUT),
        (4, 1, [8, 5, 32, 384], 3, ttnn.TILE_LAYOUT),
        (8, 1, [8, 5, 32, 512], 3, ttnn.TILE_LAYOUT),
        (4, 1, [1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("enable_async", [True, False])
def test_line_all_gather_on_t3000_nightly(
    t3k_mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=1,
):
    if t3k_mesh_device.get_num_devices() < num_devices:
        pytest.skip("Not T3000!")

    run_all_gather_on_t3000_impl(
        t3k_mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Linear,
        enable_async=enable_async,
        num_iters=num_iters,
    )


nightly_all_gather_shape_dim_layouts = [
    ([4, 1, 33, 256], 0, ttnn.ROW_MAJOR_LAYOUT),
    ([4, 1, 256, 32], 0, ttnn.TILE_LAYOUT),
    ([8, 5, 13, 512], 3, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 5, 32, 512], 3, ttnn.TILE_LAYOUT),
    ([8, 5, 13, 384], 3, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 5, 32, 384], 3, ttnn.TILE_LAYOUT),
    ([8, 8, 256, 384], 0, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 8, 256, 384], 0, ttnn.TILE_LAYOUT),
    ([8, 8, 256, 384], 1, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 8, 256, 384], 1, ttnn.TILE_LAYOUT),
    ([8, 8, 256, 384], 2, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 8, 256, 384], 2, ttnn.TILE_LAYOUT),
    ([8, 8, 256, 384], 3, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 8, 256, 384], 3, ttnn.TILE_LAYOUT),
    ([8, 8, 256, 768], 3, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 8, 256, 768], 3, ttnn.TILE_LAYOUT),
    ([8, 8, 1024, 4096], 1, ttnn.TILE_LAYOUT),
    ([8, 8, 2048, 4096], 1, ttnn.TILE_LAYOUT),
    ([8, 8, 128, 4096], 1, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 8, 1024, 4096], 1, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 8, 2048, 4096], 1, ttnn.ROW_MAJOR_LAYOUT),
    # Only for BFP8B
    # ([1, 1, 640, 32768], 3, ttnn.TILE_LAYOUT),
    # MLP AllGather. Llama 2 decode attn, mlp. Llama2, Falcon 40B decode mlp attn
    # Mixtral 8x7B, functional bringup with expanded tensor getting allgathered
    # Full shape for 8 chips
    ([1, 1, 32, 32768], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 32, 32768], 3, ttnn.ROW_MAJOR_LAYOUT),
    # Input, Selfout, Final AllGather
    ([1, 1, 32, 8192], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 32, 8192], 3, ttnn.ROW_MAJOR_LAYOUT),
    # MLP AllGather. Llama 2 decode attn, mlp. Llama2, Falcon 40B decode mlp attn
    # Half shape for 4 chips, same per chip shape as 8 chips
    ([1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 32, 16384], 3, ttnn.ROW_MAJOR_LAYOUT),
    # Input, Selfout, Final AllGather. Llama2, Falcon 40B decode mlp attn
    # Full shape for 8 chips
    ([1, 1, 32, 8192], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 32, 8192], 3, ttnn.ROW_MAJOR_LAYOUT),
    # Input, Selfout, Final AllGather. Llama2, Falcon 40B decode mlp attn
    # Half shape for running on 4 chips, same per chip shape as for 8 chips
    ([1, 1, 32, 4096], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 32, 4096], 3, ttnn.ROW_MAJOR_LAYOUT),
    # Falcon 40B prefill
    # 8 chips
    ([1, 1, 2048, 8192], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 2048, 8192], 3, ttnn.ROW_MAJOR_LAYOUT),
    # 4 chips, same per chip shape as 8 chips
    ([1, 1, 2048, 4096], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 2048, 4096], 3, ttnn.ROW_MAJOR_LAYOUT),
    # Falcon 40B prefill
    # 8 chips
    ([1, 1, 2048, 32768], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 2048, 32768], 3, ttnn.ROW_MAJOR_LAYOUT),
    # 4 chips, same per chip shape as 8 chips
    ([1, 1, 2048, 16384], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 2048, 16384], 3, ttnn.ROW_MAJOR_LAYOUT),
    # Mixtral 8x7B, Min sequence length
    # 8 chips
    # ([1, 1, 32768, 32768], 3, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 1, 32768, 32768], 3, ttnn.TILE_LAYOUT),  # ultra slow?
    # 4 chips, per chip shape same as 8 chips
    # ([1, 1, 32768, 16384], 3, ttnn.ROW_MAJOR_LAYOUT),
    # ([1, 1, 32768, 16384], 3, ttnn.TILE_LAYOUT),
    # Llama galaxy mlp weights stationary -> emulation of row/col reduce
    ([1, 1, 128, 1024], 2, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 1, 128, 1024], 2, ttnn.TILE_LAYOUT),
    # ([1, 1, 32, 8192], 3, ttnn.ROW_MAJOR_LAYOUT), # ALREADY LISTED PREVIOUSLY
    # ([1, 1, 32, 8192], 3, ttnn.TILE_LAYOUT),      # ALREADY LISTED PREVIOUSLY
    ([1, 1, 128, 4096], 2, ttnn.ROW_MAJOR_LAYOUT),  #
    ([1, 1, 128, 4096], 2, ttnn.TILE_LAYOUT),
    # ([1, 1, 32, 16384], 3, ttnn.ROW_MAJOR_LAYOUT), # ALREADY LISTED PREVIOUSLY. Update for 8 chip, actuall 32k for 8 chip but we are halving it for our 4 chip test
    # ([1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),      # ALREADY LISTED PREVIOUSLY. Update for 8 chip, actuall 32k for 8 chip but we are halving it for our 4 chip test
    ([1, 1, 8192, 32], 2, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 1, 8192, 32], 2, ttnn.TILE_LAYOUT),
    ([1, 1, 1024, 128], 3, ttnn.ROW_MAJOR_LAYOUT),  # double on reduction dim for 8 chip
    ([1, 1, 1024, 128], 3, ttnn.TILE_LAYOUT),  # double on reduction dim for 8 chip
    ([1, 1, 16384, 32], 2, ttnn.ROW_MAJOR_LAYOUT),  # double on reduction dim for 8 chip
    ([1, 1, 16384, 32], 2, ttnn.TILE_LAYOUT),  # double on reduction dim for 8 chip
    ([1, 1, 32768, 32], 2, ttnn.ROW_MAJOR_LAYOUT),  # double on reduction dim for 8 chip
    ([1, 1, 32768, 32], 2, ttnn.TILE_LAYOUT),  # double on reduction dim for 8 chip
    ([1, 1, 4096, 128], 3, ttnn.ROW_MAJOR_LAYOUT),  # only for 4 chip
    ([1, 1, 4096, 128], 3, ttnn.TILE_LAYOUT),  # only for 4 chip
    ([1, 1, 128, 2048], 2, ttnn.ROW_MAJOR_LAYOUT),  # double on reduction dim for 8 chip
    ([1, 1, 128, 2048], 2, ttnn.TILE_LAYOUT),  # double on reduction dim for 8 chip
    # ([1, 1, 32, 8192], 3, ttnn.ROW_MAJOR_LAYOUT), # only for 4 chip - ALREADY LISTED PREVIOUSLY
    # ([1, 1, 32, 8192], 3, ttnn.TILE_LAYOUT),      # only for 4 chip - ALREADY LISTED PREVIOUSLY
    ([1, 1, 128, 8192], 2, ttnn.ROW_MAJOR_LAYOUT),  # double on reduction dim for 8 chip
    ([1, 1, 128, 8192], 2, ttnn.TILE_LAYOUT),  # double on reduction dim for 8 chip
    ([4, 1, 256, 32], 0, ttnn.TILE_LAYOUT),
    ([8, 8, 256, 384], 1, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 1, 256, 1024], 2, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 1, 1024, 256], 3, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 1, 256, 2048], 2, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 1, 256, 8192], 2, ttnn.ROW_MAJOR_LAYOUT),  # double on reduction dim for 8 chip
]


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (2, 1),
        (8, 1),
    ],
)
@pytest.mark.parametrize("input_shape, dim, layout", nightly_all_gather_shape_dim_layouts)
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
def test_all_gather_on_t3000_nightly(
    mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
):
    run_all_gather_on_t3000_impl(
        mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 2),
        (4, 1),
    ],
)
@pytest.mark.parametrize("input_shape, dim, layout", nightly_all_gather_shape_dim_layouts)
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
def test_all_gather_on_t3000_nightly(
    pcie_mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
):
    if (
        input_shape == [8, 8, 256, 384]
        and dim == 1
        and layout == ttnn.TILE_LAYOUT
        and num_devices == 4
        and num_links == 1
        and input_dtype == ttnn.bfloat16
        and mem_config.buffer_type == ttnn.BufferType.DRAM
    ):
        pytest.xfail(reason="Known failure")

    if (
        input_shape == [8, 8, 256, 384]
        and dim == 2
        and layout == ttnn.TILE_LAYOUT
        and num_devices == 4
        and num_links == 1
        and input_dtype == ttnn.bfloat16
        and mem_config.buffer_type == ttnn.BufferType.DRAM
    ):
        pytest.xfail(reason="Known failure")

    if (
        input_shape == [8, 8, 256, 384]
        and dim == 2
        and layout == ttnn.TILE_LAYOUT
        and num_devices == 4
        and num_links == 1
        and input_dtype == ttnn.bfloat8_b
        and mem_config.buffer_type == ttnn.BufferType.DRAM
    ):
        pytest.xfail(reason="Known failure")

    if (
        input_shape == [8, 8, 256, 384]
        and dim == 2
        and layout == ttnn.TILE_LAYOUT
        and num_devices == 4
        and num_links == 2
        and input_dtype == ttnn.bfloat8_b
        and mem_config.buffer_type == ttnn.BufferType.DRAM
    ):
        pytest.xfail(reason="Known failure")

    run_all_gather_on_t3000_impl(
        pcie_mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
    )


def run_all_gather_sharded(
    mesh_device,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    # num_cores,
    use_program_cache,
    function_level_defaults,
    all_gather_topology,
    enable_async,
    n_worker=None,
    n_buffer=None,
    num_iter=1,
    trace_mode=False,
):
    numel = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * num_devices
    unchunked_input_shape = list(input_shape)
    unchunked_input_shape[dim] *= num_devices

    unchunked_input_tensor = torch.rand(unchunked_input_shape).bfloat16()

    debug = False
    if debug:
        tile_id = 0
        for w in range(unchunked_input_shape[0]):
            for z in range(unchunked_input_shape[1]):
                for y in range(0, unchunked_input_shape[2], 32):
                    for x in range(0, unchunked_input_shape[3], 32):
                        for yy in range(32):
                            for xx in range(32):
                                unchunked_input_tensor[w][z][y + yy][x + xx] = tile_id
                        tile_id += 1

    unchunked_input_tensor = unchunked_input_tensor.bfloat16()

    input_tensors = torch.chunk(unchunked_input_tensor, num_devices, dim)

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"unchunked_input_shape: {unchunked_input_shape}")
    logger.info(f"dim: {dim}")
    logger.info(f"num_devices: {num_devices}")
    logger.info(f"num_links: {num_links}")
    logger.info(f"input_dtype: {input_dtype}")
    logger.info(f"tensor_layout: {tensor_layout}")
    logger.info(f"tensor_mem_layout: {tensor_mem_layout}")
    logger.info(f"orientation: {orientation}")
    # logger.info(f"num_cores: {num_cores}")
    logger.info(f"shard_grid: {shard_grid}")
    logger.info(f"input_shard_shape: {input_shard_shape}")

    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        input_shard_shape,
        orientation,
        False,
    )
    input_mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)
    output_shard_shape = list(input_shard_shape)
    if dim == 3:
        output_shard_shape[1] *= num_devices
    else:
        output_shard_shape[0] *= num_devices
    output_shard_spec = ttnn.ShardSpec(
        shard_grid,
        output_shard_shape,
        orientation,
        False,
    )
    output_mem_config = ttnn.MemoryConfig(
        tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
    )

    if num_devices < 2:
        pytest.skip("Requires multiple devices to run")
    elif num_devices == 2 and num_links == 2:
        pytest.skip("Not enough links to run")

    if unchunked_input_shape[dim] % num_devices != 0 or (
        dim == 3 and unchunked_input_shape[dim] // num_devices % 32 != 0
    ):
        pytest.skip("Unsupported test case")

    tt_input_tensors_dups = []
    tt_input_tensors = []

    for i, t in enumerate(input_tensors):
        tt_input_tensors_dups.append(
            ttnn.Tensor(t, input_dtype).to(tensor_layout).to(mesh_device.get_devices()[i], input_mem_config)
        )
        tt_input_tensors.append(
            ttnn.Tensor(t, input_dtype).to(tensor_layout).to(mesh_device.get_devices()[i], input_mem_config)
        )

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)

    if trace_mode:
        tt_out_tensor = run_with_trace(
            mesh_device,
            all_gather_topology,
            input_tensor_mesh,
            dim,
            num_links,
            output_mem_config,
            n_worker,
            n_buffer,
            num_iter,
        )
    else:
        ## Run the actual allgather operation
        for i in range(num_iter):
            tt_out_tensor = ttnn.all_gather(
                input_tensor_mesh,
                dim,
                num_links=num_links,
                memory_config=output_mem_config,
                num_workers=n_worker,
                num_buffers_per_channel=n_buffer,
                topology=all_gather_topology,
            )
        ## Wait for completion
        for d in mesh_device.get_devices():
            ttnn.synchronize_device(d)

    torch.set_printoptions(sci_mode=False)
    all_eq = True
    reported_mismatch = False
    for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
        tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if input_dtype == ttnn.bfloat16:
            eq, output = comp_equal(tt_output_tensor, unchunked_input_tensor)
        else:
            eq, output = comp_pcc(tt_output_tensor, unchunked_input_tensor)
        if not eq:
            all_eq = False
            logger.error(f"output mismatch for tensor {i}")
            for w in range(input_shape[0]):
                for z in range(input_shape[1]):
                    for y in range(0, input_shape[2], 32):
                        for x in range(0, input_shape[3], 32):
                            xx = 0
                            yy = 0
                            # for yy in range(32):
                            #     for xx in range(32):
                            if tt_output_tensor[w, z, y + yy, x + xx] != unchunked_input_tensor[w, z, y + yy, x + xx]:
                                logger.error(
                                    f"mismatch at {w}, {z}, {y + yy}, {x + xx}: {tt_output_tensor[w, z, y + yy, x + xx]} != {unchunked_input_tensor[w, z, y + yy, x + xx]}"
                                )
                                # if not reported_mismatch:
                                #     reported_mismatch = True

    assert all_eq, f"{i} FAILED: {output}"


def run_all_gather_sharded_t3k(
    t3k_mesh_device,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    # num_cores,
    use_program_cache,
    function_level_defaults,
    all_gather_topology,
    enable_async,
    n_worker=None,
    n_buffer=None,
    num_iter=1,
    trace_mode=False,
):
    if t3k_mesh_device.get_num_devices() < num_devices:
        pytest.skip("Not T3000!")

    t3k_mesh_device.enable_async(enable_async)

    return run_all_gather_sharded(
        t3k_mesh_device,
        num_devices,
        input_shape,
        input_shard_shape,
        shard_grid,
        dim,
        num_links,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        # num_cores,
        use_program_cache,
        function_level_defaults,
        all_gather_topology,
        enable_async,
        n_worker,
        n_buffer,
        num_iter,
        trace_mode,
    )


def run_all_gather_sharded_n300(
    mesh_device,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    # num_cores,
    use_program_cache,
    function_level_defaults,
    all_gather_topology,
    enable_async,
    n_worker=None,
    n_buffer=None,
    num_iter=1,
    trace_mode=False,
):
    if mesh_device.get_num_devices() != 2:
        pytest.skip("Not N300!")

    mesh_device.enable_async(enable_async)

    return run_all_gather_sharded(
        mesh_device,
        num_devices,
        input_shape,
        input_shard_shape,
        shard_grid,
        dim,
        num_links,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        # num_cores,
        use_program_cache,
        function_level_defaults,
        all_gather_topology,
        enable_async,
        n_worker,
        n_buffer,
        num_iter,
        trace_mode,
    )


# @pytest.mark.parametrize("num_devices", [4, 8])
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
# @pytest.mark.parametrize("num_cores", [1])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,  # https://github.com/tenstorrent/tt-metal/issues/9686
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "tensor_mem_layout",
    [
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        # ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        # ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "input_shape, input_shard_shape,shard_grid",
    (
        # LLama
        (
            (1, 1, 32, 1024),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 4096),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 2048),
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 1792),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}),
        ),
    ),
)
@pytest.mark.parametrize("enable_async", [True])
def test_all_gather_sharded_post_commit(
    t3k_mesh_device,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    # num_cores,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_sharded_t3k(
        t3k_mesh_device,
        num_devices,
        input_shape,
        input_shard_shape,
        shard_grid,
        dim,
        num_links,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        # num_cores,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
        enable_async=enable_async,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
# @pytest.mark.parametrize("num_cores", [1])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,  # https://github.com/tenstorrent/tt-metal/issues/9686
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "tensor_mem_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        # ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "input_shape, input_shard_shape,shard_grid",
    (
        # LLama
        (
            (1, 1, 1024, 32),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 4096, 32),
            (128, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 4096, 32),
            (128, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 2048, 32),
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 1792, 32),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}),
        ),
    ),
)
@pytest.mark.parametrize("enable_async", [True])
def test_all_gather_height_sharded_post_commit(
    t3k_mesh_device,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    # num_cores,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_sharded_t3k(
        t3k_mesh_device,
        num_devices,
        input_shape,
        input_shard_shape,
        shard_grid,
        dim,
        num_links,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        # num_cores,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
        enable_async=enable_async,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
# @pytest.mark.parametrize("num_cores", [1])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,  # https://github.com/tenstorrent/tt-metal/issues/9686
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "tensor_mem_layout",
    [
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "input_shape, input_shard_shape,shard_grid",
    (
        # LLama
        (
            (1, 1, 128, 256),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        # (  # https://github.com/tenstorrent/tt-metal/issues/9686
        #     (1, 1, 512, 32),
        #     (128, 32),
        #     ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        # ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 512, 256),
            (128, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 512, 512),
            (128, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
    ),
)
@pytest.mark.parametrize("enable_async", [True])
def test_all_gather_block_sharded_post_commit(
    t3k_mesh_device,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    # num_cores,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_sharded_t3k(
        t3k_mesh_device,
        num_devices,
        input_shape,
        input_shard_shape,
        shard_grid,
        dim,
        num_links,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        # num_cores,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
        enable_async=enable_async,
    )


# @pytest.mark.parametrize("num_devices", [4, 8])
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
# @pytest.mark.parametrize("num_cores", [1])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,  # https://github.com/tenstorrent/tt-metal/issues/9686
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "tensor_mem_layout",
    [
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        # ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        # ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "input_shape, input_shard_shape,shard_grid",
    (
        # LLama
        (
            (1, 1, 32, 1024),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 4096),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 4096),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 2048),
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 1792),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}),
        ),
    ),
)
@pytest.mark.parametrize("enable_async", [True])
def test_line_all_gather_sharded_post_commit(
    t3k_mesh_device,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    # num_cores,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_sharded_t3k(
        t3k_mesh_device,
        num_devices,
        input_shape,
        input_shard_shape,
        shard_grid,
        dim,
        num_links,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        # num_cores,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Linear,
        enable_async=enable_async,
    )


# @pytest.mark.parametrize("num_devices", [4, 8])
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
# @pytest.mark.parametrize("num_cores", [1])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,  # https://github.com/tenstorrent/tt-metal/issues/9686
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "tensor_mem_layout",
    [
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        # ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        # ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "input_shape, input_shard_shape,shard_grid",
    (
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 128),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 256),
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 64, 128),
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 64, 256),
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 64),
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 128),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 64),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 128),
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (
            (1, 1, 32, 128),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
        ),
        (
            (1, 1, 64, 128),
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
        ),
        (
            (1, 1, 32, 256),
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
        ),
        (
            (1, 1, 64, 256),
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 256),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 512),
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
        ),
        # LLama
        (
            (1, 1, 32, 1024),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 4096),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 4096),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 2048),
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 1792),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}),
        ),
    ),
)
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("all_gather_topology", [ttnn.Topology.Ring, ttnn.Topology.Linear])
def test_sharded_all_gather_nightly(
    t3k_mesh_device,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    # num_cores,
    use_program_cache,
    function_level_defaults,
    all_gather_topology,
    enable_async,
):
    run_all_gather_sharded_t3k(
        t3k_mesh_device,
        num_devices,
        input_shape,
        input_shard_shape,
        shard_grid,
        dim,
        num_links,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        # num_cores,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=all_gather_topology,
        enable_async=enable_async,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.skip("#7705: Hanging on various configs")
@pytest.mark.parametrize(
    "input_shape, dim, layout",
    [([4, 1, 33, 256], 0, ttnn.ROW_MAJOR_LAYOUT), ([4, 1, 256, 32], 0, ttnn.TILE_LAYOUT)],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("num_links", [1, 2])
def test_all_gather_fp32(  # https://github.com/tenstorrent/tt-metal/issues/9686 ... need to tag with post_commit
    pcie_devices, input_shape, dim, num_links, layout, mem_config, use_program_cache, function_level_defaults
):
    if (layout == ttnn.ROW_MAJOR_LAYOUT or num_links == 2) and mem_config.buffer_type == ttnn.BufferType.DRAM:
        pytest.skip("All gather tests are hanging for RM in DRAM")
    devices = pcie_devices
    input_tensor = torch.rand(input_shape).bfloat16()
    num_devices = len(devices)
    if num_devices < 2:
        pytest.skip("Requires multiple devices to run")
    elif num_devices == 2 and num_links == 2:
        pytest.skip("Not enough links to run")

    if input_shape[dim] % num_devices != 0 or (dim == 3 and input_shape[dim] // num_devices % 32 != 0):
        pytest.skip("Unsupported test case")

    input_tensors = torch.chunk(input_tensor, num_devices, dim)
    tt_input_tensors = []
    for i, t in enumerate(input_tensors):
        tt_input_tensors.append(ttnn.Tensor(t, ttnn.float32).to(layout).to(devices[i], mem_config))

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)
    tt_out_tensor = ttnn.all_gather(input_tensor_mesh, dim, num_links=num_links, memory_config=mem_config)

    for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
        tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        eq, output = comp_equal(tt_output_tensor, input_tensor)
        assert eq, f"{i} FAILED: {output}"
