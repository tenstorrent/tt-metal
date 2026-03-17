# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0

from tests.nightly.t3000.ccl.test_minimal_matmul_strided_reduce_scatter_async import (
    MinimalMatmulStridedReduceScatterTestConfig,
    run_minimal_matmul_strided_reduce_scatter_impl,
)


def _make_fabric_router_config(max_packet_payload_size_bytes):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_packet_payload_size_bytes
    return config


# Llama 70B specific dimensions for FF1/FF3 operations (8K sequence length only)
# M = sequence_length, K = dim//num_devices_per_row, N = hidden_dim//num_devices_per_col
# For 8x4 mesh: dim=8192, hidden_dim=28672 -> K=1024, N=7168
LLAMA_70B_8K_DIMS = {"M": 8192, "K": 1024, "N": 7168}


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("num_links", [1, 2, 3], ids=["1link", "2link", "3link"])
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis_0", "axis_1"])  # RS along height(0) or width(1)
@pytest.mark.parametrize(
    "mm_core_grid_x, mm_core_grid_y",
    [
        (6, 8),  # 48 cores - your best config (testing this one first)
    ],
    ids=["6x8"],
)
@pytest.mark.parametrize(
    "chunk_width_in_mm_blocks",
    [1, 2],  # Focus on most promising values
    ids=["cwimb1", "cwimb2"],
)
@pytest.mark.parametrize(
    "num_workers_per_link",
    [3, 4],  # Focus on middle range
    ids=["nw3", "nw4"],
)
@pytest.mark.parametrize(
    "mm_block_size",
    [
        (256, 128, 256),  # Your optimal block size from analysis
    ],
    ids=["blk256_128_256"],
)
@pytest.mark.parametrize(
    "rs_mode",
    ["separate", "fused"],
    ids=["separate", "fused"],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_mm, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
    ids=["DRAM"],
)
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_llama_70b_rs_mm_sweep(
    mesh_device,
    num_links,
    cluster_axis,
    mm_core_grid_x,
    mm_core_grid_y,
    chunk_width_in_mm_blocks,
    num_workers_per_link,
    mm_block_size,
    rs_mode,
    mem_config_input,
    mem_config_mm,
    mem_config_rs,
    topology,
):
    """
    Sweep test for Llama 70B FF1/FF3 operations (8K sequence length) with RS+MM optimization.
    Tests both separate and fused implementations across different configurations.
    """

    # Skip invalid configurations
    if mesh_device.shape[cluster_axis] == 1:
        pytest.skip(f"cluster_axis={cluster_axis} has only 1 device in this mesh, reduce-scatter ring size must be > 1")

    # num_links must divide grid x-dimension for proper RS core placement
    if mm_core_grid_x % num_links != 0:
        pytest.skip(f"num_links={num_links} must divide mm_core_grid.x={mm_core_grid_x}")

    # Check core grid fits within device limits
    if is_wormhole_b0() and (mm_core_grid_x > 8 or mm_core_grid_y > 8):
        pytest.skip("core grid exceeds wormhole_b0 compute grid (8x8)")
    if is_blackhole() and (mm_core_grid_x > 12 or mm_core_grid_y > 10):
        pytest.skip("core grid exceeds blackhole compute grid (12x10)")

    # Get Llama 8K dimensions
    M, K, N = LLAMA_70B_8K_DIMS["M"], LLAMA_70B_8K_DIMS["K"], LLAMA_70B_8K_DIMS["N"]

    # Unpack block sizes
    mm_block_m, mm_block_k, mm_block_n = mm_block_size

    # Validate block sizes fit within tensor dimensions
    TILE_SIZE = 32
    Nt = N // TILE_SIZE
    Nt_per_core = Nt // mm_core_grid_x
    if Nt_per_core < (mm_block_n // TILE_SIZE):
        pytest.skip(f"block_n size is {mm_block_n // TILE_SIZE} tiles, but only {Nt_per_core} tiles of work per core")

    # Create test config
    test_config = MinimalMatmulStridedReduceScatterTestConfig(
        M=M,
        K=K,
        N=N,
        dim=3,  # Reduce scatter along width dimension
        mm_block_m=mm_block_m,
        mm_block_k=mm_block_k,
        mm_block_n=mm_block_n,
        mm_core_grid=ttnn.CoreCoord(mm_core_grid_x, mm_core_grid_y),
        chunk_width_in_mm_blocks=chunk_width_in_mm_blocks,
        num_workers_per_link=num_workers_per_link,
    )

    # Run the test
    run_minimal_matmul_strided_reduce_scatter_impl(
        mesh_device,
        test_config.M,
        test_config.K,
        test_config.N,
        test_config.dim,
        num_links,
        test_config.input_dtype,
        test_config.layout,
        mem_config_input,
        mem_config_mm,
        mem_config_rs,
        topology=topology,
        enable_trace=False,
        num_iters=1,
        num_workers_per_link=test_config.num_workers_per_link,
        mm_block_m=test_config.mm_block_m,
        mm_block_k=test_config.mm_block_k,
        mm_block_n=test_config.mm_block_n,
        subblock_h=test_config.subblock_h,
        subblock_w=test_config.subblock_w,
        mm_core_grid=test_config.mm_core_grid,
        chunk_width_in_mm_blocks=test_config.chunk_width_in_mm_blocks,
        rs_mode=rs_mode,
        cluster_axis=cluster_axis,
    )


# Focused test for best known configurations from your analysis (8K sequence length)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["8x4"])
@pytest.mark.parametrize(
    "config_name, mm_core_grid, num_links, chunk_width, num_workers",
    [
        ("best_6x8_3links", (6, 8), 3, 2, 3),  # Your best config: ~19% improvement
        ("alt_6x7_3links", (6, 7), 3, 2, 3),  # Alternative good config
        ("baseline_7x7_1link", (7, 7), 1, 2, 3),  # Baseline for comparison
    ],
    ids=["best_6x8_3links", "alt_6x7_3links", "baseline_7x7_1link"],
)
@pytest.mark.parametrize(
    "rs_mode",
    ["separate", "fused"],
    ids=["separate", "fused"],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_mm, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
    ids=["DRAM"],
)
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_llama_70b_rs_mm_best_configs(
    mesh_device,
    config_name,
    mm_core_grid,
    num_links,
    chunk_width,
    num_workers,
    rs_mode,
    mem_config_input,
    mem_config_mm,
    mem_config_rs,
    topology,
):
    """
    Focused test for the best performing configurations identified in analysis (8K sequence length).
    Compares separate vs fused implementations for known good configs.
    """

    # Get Llama 8K dimensions
    M, K, N = LLAMA_70B_8K_DIMS["M"], LLAMA_70B_8K_DIMS["K"], LLAMA_70B_8K_DIMS["N"]

    # Use optimal block sizes for these configs
    mm_block_m, mm_block_k, mm_block_n = 256, 128, 256

    test_config = MinimalMatmulStridedReduceScatterTestConfig(
        M=M,
        K=K,
        N=N,
        dim=3,
        mm_block_m=mm_block_m,
        mm_block_k=mm_block_k,
        mm_block_n=mm_block_n,
        mm_core_grid=ttnn.CoreCoord(mm_core_grid[0], mm_core_grid[1]),
        chunk_width_in_mm_blocks=chunk_width,
        num_workers_per_link=num_workers,
    )

    run_minimal_matmul_strided_reduce_scatter_impl(
        mesh_device,
        test_config.M,
        test_config.K,
        test_config.N,
        test_config.dim,
        num_links,
        test_config.input_dtype,
        test_config.layout,
        mem_config_input,
        mem_config_mm,
        mem_config_rs,
        topology=topology,
        enable_trace=False,
        num_iters=1,
        num_workers_per_link=test_config.num_workers_per_link,
        mm_block_m=test_config.mm_block_m,
        mm_block_k=test_config.mm_block_k,
        mm_block_n=test_config.mm_block_n,
        subblock_h=test_config.subblock_h,
        subblock_w=test_config.subblock_w,
        mm_core_grid=test_config.mm_core_grid,
        chunk_width_in_mm_blocks=test_config.chunk_width_in_mm_blocks,
        rs_mode=rs_mode,
        cluster_axis=1,  # RS along width for 8x4 mesh
    )
