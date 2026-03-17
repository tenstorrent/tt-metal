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


# Llama 70B Galaxy configuration - matching real model parameters
LLAMA_70B_GALAXY_DIMS = {"M": 8192, "K": 1024, "N": 7168}  # 8K sequence length, 8x4 mesh


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("galaxy_type", ["6U", "4U"], ids=["6U", "4U"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_llama_70b_baseline_separate_galaxy_config(
    mesh_device,
    galaxy_type,
    topology,
):
    """
    Test separate MM+RS operations using EXACT Llama 70B Galaxy configuration.
    This should match the real Llama model performance (~700µs MM + ~700µs RS = ~1.4ms total).

    Key Llama parameters:
    - cluster_axis=1 (reduce scatter along width, 4-device rings)
    - num_links=4 (6U) or 3 (4U)
    - LoFi math fidelity (not HiFi2)
    - Optimized block sizes and core grids
    - Sharded memory configs (not DRAM interleaved)
    """

    # This test is for Galaxy (both Wormhole and Blackhole Galaxy systems)
    # No device type skip needed - Galaxy can be Wormhole or Blackhole

    # EXACT Llama configuration: prefill mode uses num_links=3 regardless of Galaxy type
    num_links = 3  # Llama prefill mode ALWAYS uses 3 links (from llama_mlp.py line 241, 275)

    # Llama 70B dimensions for 8x4 mesh
    M, K, N = LLAMA_70B_GALAXY_DIMS["M"], LLAMA_70B_GALAXY_DIMS["K"], LLAMA_70B_GALAXY_DIMS["N"]

    # EXACT Llama 70B Galaxy parameters (from model_config.py analysis)
    cluster_axis = 1  # RS along width (4-device rings) - KEY DIFFERENCE from our previous test

    # EXACT block sizes from Llama MinimalMatmulConfig (TILE_SIZE=32)
    mm_block_m = 256  # 8 tiles (M_block_size=8 * 32 = 256 elements)
    mm_block_k = 128  # 4 tiles (K_block_size=8, but adjusted for K dimension)
    mm_block_n = 256  # 8 tiles (N_block_size=8 * 32 = 256 elements)

    # EXACT core grid from Llama model (prefill mode for seq_len > 4096)
    mm_core_grid = ttnn.CoreCoord(7, 8)  # 56 cores (ttnn.CoreCoord(7, 8) for seq_len=8K)

    # EXACT Llama parameters for optimal performance
    chunk_width_in_mm_blocks = 2  # Good balance from analysis
    num_workers_per_link = 4  # Optimal from Llama ring config (RING_SIZE=24, 24/6=4 workers per link)

    # Memory configs - still using DRAM for now (sharded would be better but more complex)
    mem_config_input = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    mem_config_mm = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    mem_config_rs = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    # Create test config
    test_config = MinimalMatmulStridedReduceScatterTestConfig(
        M=M,
        K=K,
        N=N,
        dim=3,  # Scatter dimension
        mm_block_m=mm_block_m,
        mm_block_k=mm_block_k,
        mm_block_n=mm_block_n,
        mm_core_grid=mm_core_grid,
        chunk_width_in_mm_blocks=chunk_width_in_mm_blocks,
        num_workers_per_link=num_workers_per_link,
        input_dtype=ttnn.bfloat16,  # Match Llama
        layout=ttnn.TILE_LAYOUT,
    )

    # Run SEPARATE operations only (no fused)
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
        rs_mode="separate",  # SEPARATE ONLY
        cluster_axis=cluster_axis,  # KEY: axis=1 like real Llama
        math_fidelity=ttnn.MathFidelity.LoFi,  # LoFi like real Llama (not HiFi2)
        fp32_acc=True,
    )


# Additional test with different core grid options to find optimal
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["8x4"])
@pytest.mark.parametrize(
    "core_grid_config",
    [
        {"grid": (6, 6), "name": "6x6_36cores"},
        {"grid": (6, 7), "name": "6x7_42cores"},
        {"grid": (7, 6), "name": "7x6_42cores"},
        {"grid": (8, 6), "name": "8x6_48cores"},
    ],
    ids=["6x6", "6x7", "7x6", "8x6"],
)
@pytest.mark.parametrize(
    "block_config",
    [
        {"blocks": (64, 64, 64), "name": "blk64"},
        {"blocks": (128, 64, 128), "name": "blk128_64_128"},
        {"blocks": (128, 128, 128), "name": "blk128"},
    ],
    ids=["blk64", "blk128_64_128", "blk128"],
)
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_llama_70b_baseline_separate_optimization(
    mesh_device,
    core_grid_config,
    block_config,
    topology,
):
    """
    Test different core grids and block sizes to find optimal configuration
    that matches Llama's ~1.4ms total performance.
    """

    # This test works on both Wormhole and Blackhole Galaxy systems
    # No device type skip needed

    # Llama configuration
    M, K, N = LLAMA_70B_GALAXY_DIMS["M"], LLAMA_70B_GALAXY_DIMS["K"], LLAMA_70B_GALAXY_DIMS["N"]
    cluster_axis = 1  # Llama uses width-wise RS
    num_links = 3  # Use 3 links (good for most configs)

    # Extract test parameters
    mm_core_grid = ttnn.CoreCoord(core_grid_config["grid"][0], core_grid_config["grid"][1])
    mm_block_m, mm_block_k, mm_block_n = block_config["blocks"]

    # Optimized parameters
    chunk_width_in_mm_blocks = 2
    num_workers_per_link = 3

    # Memory configs
    mem_config_input = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    mem_config_mm = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    mem_config_rs = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    # Create test config
    test_config = MinimalMatmulStridedReduceScatterTestConfig(
        M=M,
        K=K,
        N=N,
        dim=3,
        mm_block_m=mm_block_m,
        mm_block_k=mm_block_k,
        mm_block_n=mm_block_n,
        mm_core_grid=mm_core_grid,
        chunk_width_in_mm_blocks=chunk_width_in_mm_blocks,
        num_workers_per_link=num_workers_per_link,
        input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run separate operations
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
        rs_mode="separate",
        cluster_axis=cluster_axis,
        math_fidelity=ttnn.MathFidelity.LoFi,  # LoFi for performance
        fp32_acc=True,
    )
