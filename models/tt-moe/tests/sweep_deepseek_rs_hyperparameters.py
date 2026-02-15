# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Reduce-scatter hyperparameter sweep configurations and unit tests for DeepSeek-V3.
This file contains the actual test implementations that will be called by the performance test.
"""

import pytest
from loguru import logger

import ttnn
from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl

# DeepSeek-V3 specific shapes for reduce-scatter operations
# Format: (num_devices, num_links, rs_input_shape, dim)
# num_devices: 8 for expert parallel, 4 for tensor parallel
DEEPSEEK_V3_DECODE_CONFIGS = [
    # Decode mode - Expert Parallel (8 devices)
    (8, 1, [1, 32, 1, 7168], 3),  # seq_len=1, 1 link
    (8, 2, [1, 32, 1, 7168], 3),  # seq_len=1, 2 links
    (8, 4, [1, 32, 1, 7168], 3),  # seq_len=1, 4 links
    (8, 1, [1, 32, 4, 7168], 3),  # seq_len=4, 1 link
    (8, 2, [1, 32, 4, 7168], 3),  # seq_len=4, 2 links
    (8, 4, [1, 32, 4, 7168], 3),  # seq_len=4, 4 links
    (8, 1, [1, 32, 8, 7168], 3),  # seq_len=8, 1 link
    (8, 2, [1, 32, 8, 7168], 3),  # seq_len=8, 2 links
    (8, 4, [1, 32, 8, 7168], 3),  # seq_len=8, 4 links
    (8, 1, [1, 32, 16, 7168], 3),  # seq_len=16, 1 link
    (8, 2, [1, 32, 16, 7168], 3),  # seq_len=16, 2 links
    (8, 4, [1, 32, 16, 7168], 3),  # seq_len=16, 4 links
    (8, 1, [1, 32, 32, 7168], 3),  # seq_len=32, 1 link
    (8, 2, [1, 32, 32, 7168], 3),  # seq_len=32, 2 links
    (8, 4, [1, 32, 32, 7168], 3),  # seq_len=32, 4 links
    # Decode mode - Tensor Parallel (4 devices)
    (4, 1, [1, 32, 1, 7168], 3),  # seq_len=1, 1 link
    (4, 2, [1, 32, 1, 7168], 3),  # seq_len=1, 2 links
    (4, 4, [1, 32, 1, 7168], 3),  # seq_len=1, 4 links
    (4, 1, [1, 32, 4, 7168], 3),  # seq_len=4, 1 link
    (4, 2, [1, 32, 4, 7168], 3),  # seq_len=4, 2 links
    (4, 4, [1, 32, 4, 7168], 3),  # seq_len=4, 4 links
    (4, 1, [1, 32, 8, 7168], 3),  # seq_len=8, 1 link
    (4, 2, [1, 32, 8, 7168], 3),  # seq_len=8, 2 links
    (4, 4, [1, 32, 8, 7168], 3),  # seq_len=8, 4 links
    (4, 1, [1, 32, 16, 7168], 3),  # seq_len=16, 1 link
    (4, 2, [1, 32, 16, 7168], 3),  # seq_len=16, 2 links
    (4, 4, [1, 32, 16, 7168], 3),  # seq_len=16, 4 links
    (4, 1, [1, 32, 32, 7168], 3),  # seq_len=32, 1 link
    (4, 2, [1, 32, 32, 7168], 3),  # seq_len=32, 2 links
    (4, 4, [1, 32, 32, 7168], 3),  # seq_len=32, 4 links
]

DEEPSEEK_V3_PREFILL_CONFIGS = [
    # Prefill mode - Expert Parallel (8 devices)
    (8, 1, [1, 32, 512, 7168], 3),  # seq_len=512, 1 link
    (8, 2, [1, 32, 512, 7168], 3),  # seq_len=512, 2 links
    (8, 4, [1, 32, 512, 7168], 3),  # seq_len=512, 4 links
    (8, 1, [1, 32, 1024, 7168], 3),  # seq_len=1024, 1 link
    (8, 2, [1, 32, 1024, 7168], 3),  # seq_len=1024, 2 links
    (8, 4, [1, 32, 1024, 7168], 3),  # seq_len=1024, 4 links
    (8, 1, [1, 32, 2048, 7168], 3),  # seq_len=2048, 1 link
    (8, 2, [1, 32, 2048, 7168], 3),  # seq_len=2048, 2 links
    (8, 4, [1, 32, 2048, 7168], 3),  # seq_len=2048, 4 links
    # Prefill mode - Tensor Parallel (4 devices)
    (4, 1, [1, 32, 512, 7168], 3),  # seq_len=512, 1 link
    (4, 2, [1, 32, 512, 7168], 3),  # seq_len=512, 2 links
    (4, 4, [1, 32, 512, 7168], 3),  # seq_len=512, 4 links
    (4, 1, [1, 32, 1024, 7168], 3),  # seq_len=1024, 1 link
    (4, 2, [1, 32, 1024, 7168], 3),  # seq_len=1024, 2 links
    (4, 4, [1, 32, 1024, 7168], 3),  # seq_len=1024, 4 links
    (4, 1, [1, 32, 2048, 7168], 3),  # seq_len=2048, 1 link
    (4, 2, [1, 32, 2048, 7168], 3),  # seq_len=2048, 2 links
    (4, 4, [1, 32, 2048, 7168], 3),  # seq_len=2048, 4 links
]

# Combine all configurations
CONFIGS = DEEPSEEK_V3_DECODE_CONFIGS + DEEPSEEK_V3_PREFILL_CONFIGS

# Generate configuration IDs for pytest parametrization
CONFIGS_IDS = []
for i, (num_devices, num_links, shape, dim) in enumerate(CONFIGS):
    seq_len = shape[2]
    axis_type = "ep" if num_devices == 8 else "tp"
    mode = "decode" if seq_len <= 32 else "prefill"
    CONFIGS_IDS.append(f"{mode}_seq{seq_len}_{axis_type}_{num_links}L")

# Hyperparameter sweep ranges
WORKERS_PER_LINK = [None, 1, 2, 4]  # None means use default
WORKERS_PER_LINK_IDS = [f"{worker}-workers" if worker else "default-workers" for worker in WORKERS_PER_LINK]

CHUNKS_PER_SYNC = [None, 32, 16, 8]  # None means use default (no MAX for reduce-scatter)
CHUNKS_PER_SYNC_IDS = [f"{chunk}-chunks" if chunk else "default-chunks" for chunk in CHUNKS_PER_SYNC]

TOPOLOGY = ["ring", "linear"]


@pytest.mark.parametrize(
    "num_devices, num_links, rs_input_shape, dim",
    CONFIGS,
    ids=CONFIGS_IDS,
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        # DRAM configurations (most common for DeepSeek-V3)
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        # L1 configurations (for performance comparison on smaller shapes)
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1_SMALL),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1_SMALL),
        ),
    ],
    ids=["dram", "l1"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),  # Performance mode with tracing (fewer iterations for reduce-scatter)
    ],
    ids=["perf"],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 2000000}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 2000000}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=TOPOLOGY,
)
@pytest.mark.parametrize("chunks_per_sync", CHUNKS_PER_SYNC, ids=CHUNKS_PER_SYNC_IDS)
@pytest.mark.parametrize("num_workers_per_link", WORKERS_PER_LINK, ids=WORKERS_PER_LINK_IDS)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)  # TensorGrid (4x8 = 32 devices)
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16])  # DeepSeek-V3 uses bfloat16
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("num_outer_iters", [4])  # Number of outer iterations for warming up
def test_reduce_scatter_chunks_per_sync(
    mesh_device,
    num_devices,
    rs_input_shape,
    dim,
    num_links,
    rs_input_dtype,
    layout,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    rs_topology,
    num_iters,
    chunks_per_sync,
    num_workers_per_link,
    num_outer_iters,
):
    """
    Test reduce-scatter operation with various hyperparameter combinations for DeepSeek-V3.

    This test will be called by the performance sweep to measure bandwidth and latency
    for different configurations.
    """
    # Skip invalid combinations
    if chunks_per_sync != None and num_workers_per_link == None:
        pytest.skip("Invalid combination: chunks_per_sync requires num_workers_per_link")
    elif chunks_per_sync == None and num_workers_per_link != None:
        pytest.skip("Invalid combination: num_workers_per_link requires chunks_per_sync")

    # Skip L1 configurations for large prefill shapes (would OOM)
    if mem_config_rs.buffer_type == ttnn.BufferType.L1_SMALL:
        seq_len = rs_input_shape[2]
        if seq_len >= 512:
            pytest.skip("L1 configuration not supported for large prefill shapes")

    # Create submesh based on number of devices
    cluster_axis = 0
    if num_devices == 4:
        # Tensor parallel - use 4 devices in a column
        submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
        cluster_axis = 1
    elif num_devices == 8:
        # Expert parallel - use 8 devices in a row
        submesh_device = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
        cluster_axis = 0
    elif num_devices == 2:
        # Special case for 2 devices (testing only)
        if rs_topology == ttnn.Topology.Ring:
            pytest.skip("Ring topology is not supported for 2 devices")
        submesh_device = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
        cluster_axis = 0
    else:
        pytest.skip(f"Unsupported number of devices: {num_devices}")

    # Log test configuration
    seq_len = rs_input_shape[2]
    mode = "decode" if seq_len <= 32 else "prefill"
    axis_type = "ep" if num_devices == 8 else "tp"

    logger.info(
        f"Testing reduce-scatter with shape {rs_input_shape}, "
        f"mode={mode}, axis={axis_type}, num_devices={num_devices}, "
        f"num_links={num_links}, topology={rs_topology}, "
        f"chunks_per_sync={chunks_per_sync}, num_workers_per_link={num_workers_per_link}"
    )

    # Run multiple outer iterations for better performance measurement
    for outer_iter in range(num_outer_iters):
        logger.debug(f"Running outer iteration {outer_iter + 1}/{num_outer_iters}")

        run_reduce_scatter_impl(
            submesh_device,
            num_devices,
            rs_input_shape,
            dim,
            num_links,
            rs_input_dtype,
            layout,
            mem_config_input,
            mem_config_rs,
            rs_topology=rs_topology,
            enable_trace=enable_trace,
            num_iters=num_iters,
            cluster_axis=cluster_axis,
            chunks_per_sync=chunks_per_sync,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=None,
            verify_output=False,  # Skip validation for performance tests
            use_persistent_buffers=False,
        )
