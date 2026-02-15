# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
All-gather hyperparameter sweep configurations and unit tests for DeepSeek-V3.
This file contains the actual test implementations that will be called by the performance test.
"""

import math

import pytest
from loguru import logger

import ttnn
from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl


def get_max_chunks_per_sync(num_devices, ag_output_shape, num_links):
    """Calculate maximum chunks per sync based on shape and device configuration."""
    packet_elems = 2048
    total_elems = math.prod(ag_output_shape)
    return (total_elems // packet_elems) // (num_devices * num_links)


# DeepSeek-V3 specific shapes for all-gather operations
# Format: (ag_output_shape, cluster_axis, dim)
# cluster_axis: 0 for expert parallel (8 devices), 1 for tensor parallel (4 devices)
DEEPSEEK_V3_DECODE_CONFIGS = [
    # Decode mode shapes with varying sequence lengths
    ([1, 32, 1, 7168], 0, 3),  # seq_len=1, expert parallel
    ([1, 32, 1, 7168], 1, 3),  # seq_len=1, tensor parallel
    ([1, 32, 4, 7168], 0, 3),  # seq_len=4, expert parallel
    ([1, 32, 4, 7168], 1, 3),  # seq_len=4, tensor parallel
    ([1, 32, 8, 7168], 0, 3),  # seq_len=8, expert parallel
    ([1, 32, 8, 7168], 1, 3),  # seq_len=8, tensor parallel
    ([1, 32, 16, 7168], 0, 3),  # seq_len=16, expert parallel
    ([1, 32, 16, 7168], 1, 3),  # seq_len=16, tensor parallel
    ([1, 32, 32, 7168], 0, 3),  # seq_len=32, expert parallel
    ([1, 32, 32, 7168], 1, 3),  # seq_len=32, tensor parallel
]

DEEPSEEK_V3_PREFILL_CONFIGS = [
    # Prefill mode shapes with larger sequence lengths
    ([1, 32, 512, 7168], 0, 3),  # seq_len=512, expert parallel
    ([1, 32, 512, 7168], 1, 3),  # seq_len=512, tensor parallel
    ([1, 32, 1024, 7168], 0, 3),  # seq_len=1024, expert parallel
    ([1, 32, 1024, 7168], 1, 3),  # seq_len=1024, tensor parallel
    ([1, 32, 2048, 7168], 0, 3),  # seq_len=2048, expert parallel
    ([1, 32, 2048, 7168], 1, 3),  # seq_len=2048, tensor parallel
]

# Combine all configurations
CONFIGS = DEEPSEEK_V3_DECODE_CONFIGS + DEEPSEEK_V3_PREFILL_CONFIGS

# Generate configuration IDs for pytest parametrization
CONFIGS_IDS = []
for i, (shape, cluster_axis, dim) in enumerate(CONFIGS):
    seq_len = shape[2]
    axis_type = "ep" if cluster_axis == 0 else "tp"
    mode = "decode" if seq_len <= 32 else "prefill"
    CONFIGS_IDS.append(f"{mode}_seq{seq_len}_{axis_type}")

# Hyperparameter sweep ranges
WORKERS_PER_LINK = [None, 1, 2, 4]  # None means use default
WORKERS_PER_LINK_IDS = [f"{worker}-workers" if worker else "default-workers" for worker in WORKERS_PER_LINK]

CHUNKS_PER_SYNC = [None, "MAX", 32, 16, 8]  # None means use default
CHUNKS_PER_SYNC_IDS = [
    f"{chunk}-chunks" if chunk and chunk != "MAX" else ("max-chunks" if chunk == "MAX" else "default-chunks")
    for chunk in CHUNKS_PER_SYNC
]

TOPOLOGY = ["ring", "linear"]

# Number of links to test
NUM_LINKS = [1, 2, 4]
NUM_LINKS_IDS = [f"{n}link" for n in NUM_LINKS]


@pytest.mark.parametrize(
    "ag_output_shape, cluster_axis, dim",
    CONFIGS,
    ids=CONFIGS_IDS,
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        # DRAM configurations (most common for DeepSeek-V3)
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        # L1 configurations (for performance comparison)
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
        (True, 30),  # Performance mode with tracing
    ],
    ids=["perf"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=TOPOLOGY,
)
@pytest.mark.parametrize("chunks_per_sync", CHUNKS_PER_SYNC, ids=CHUNKS_PER_SYNC_IDS)
@pytest.mark.parametrize("num_workers_per_link", WORKERS_PER_LINK, ids=WORKERS_PER_LINK_IDS)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)  # TensorGrid (4x8 = 32 devices)
@pytest.mark.parametrize("ag_input_dtype", [ttnn.bfloat16])  # DeepSeek-V3 uses bfloat16
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("num_links", NUM_LINKS, ids=NUM_LINKS_IDS)
def test_all_gather_chunks_per_sync(
    mesh_device,
    ag_output_shape,
    cluster_axis,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    all_gather_topology,
    num_iters,
    chunks_per_sync,
    num_workers_per_link,
):
    """
    Test all-gather operation with various hyperparameter combinations for DeepSeek-V3.

    This test will be called by the performance sweep to measure bandwidth and latency
    for different configurations.
    """
    # Skip invalid combinations
    if chunks_per_sync != None and num_workers_per_link == None:
        pytest.skip("Invalid combination: chunks_per_sync requires num_workers_per_link")
    elif chunks_per_sync == None and num_workers_per_link != None:
        pytest.skip("Invalid combination: num_workers_per_link requires chunks_per_sync")

    # Skip L1 configurations for large prefill shapes (would OOM)
    if mem_config_ag.buffer_type == ttnn.BufferType.L1_SMALL:
        seq_len = ag_output_shape[2]
        if seq_len >= 512:
            pytest.skip("L1 configuration not supported for large prefill shapes")

    # Determine number of devices based on cluster axis
    num_devices = 4
    if cluster_axis == 0:
        # Expert parallel - use 8 devices in a row
        num_devices = 8
        submesh_device = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    else:
        # Tensor parallel - use 4 devices in a column
        num_devices = 4
        submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))

    # Calculate actual chunks_per_sync if MAX is specified
    if chunks_per_sync == "MAX":
        chunks_per_sync = get_max_chunks_per_sync(num_devices, ag_output_shape, num_links)
        logger.debug(f"Using MAX chunks_per_sync: {chunks_per_sync}")

    # Log test configuration
    logger.info(
        f"Testing all-gather with shape {ag_output_shape}, "
        f"cluster_axis={cluster_axis}, num_devices={num_devices}, "
        f"num_links={num_links}, topology={all_gather_topology}, "
        f"chunks_per_sync={chunks_per_sync}, num_workers_per_link={num_workers_per_link}"
    )

    # Run the all-gather implementation
    run_all_gather_impl(
        submesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        skip_check=True,  # Skip validation for performance tests
    )
