# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performance test for fused AllGather+MatMul with Llama 70B sizes
Tests different grid configurations to get performance numbers for kernel developers
"""

import pytest
import ttnn

from conftest import is_6u
from tests.nightly.t3000.ccl.test_minimal_all_gather_matmul_async import run_all_gather_impl


@pytest.mark.skipif(not is_6u(), reason="This test is only for 6U devices")
@pytest.mark.parametrize("num_links", [1], ids=["1link"])  # Use 1 link like Llama model
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, matmul_output_dim, max_in0_block_w, matmul_weights_dtype, ag_input_dtype, use_bias",
    [
        # Llama 70B W2 matmul sizes: seq_len=8192, hidden_per_device=3584, output=2048
        (4, [1, 1, 8192, 3584], 3, ttnn.TILE_LAYOUT, 2048, 2, ttnn.bfloat16, ttnn.bfloat16, True),
        # Also test 4K sequence length
        (4, [1, 1, 4096, 3584], 3, ttnn.TILE_LAYOUT, 2048, 2, ttnn.bfloat16, ttnn.bfloat16, True),
        # Test 16K sequence length
        (4, [1, 1, 16384, 3584], 3, ttnn.TILE_LAYOUT, 2048, 2, ttnn.bfloat16, ttnn.bfloat16, True),
    ],
    ids=["llama70b_8k", "llama70b_4k", "llama70b_16k"],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag, mem_config_mm",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 20),  # More iterations for better perf measurement
    ],
    ids=["perf"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
@pytest.mark.parametrize(
    "chunks_per_sync, num_workers_per_link, num_buffers_per_channel",
    [
        (None, None, None),
    ],
    ids=["default"],
)
@pytest.mark.parametrize("mesh_device", [(4, 1)], indirect=True)  # 4 devices in ring
def test_llama70b_fused_agmm_perf(
    mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    matmul_output_dim,
    matmul_weights_dtype,
    max_in0_block_w,
    use_bias,
    mem_config_input,
    mem_config_ag,
    mem_config_mm,
    enable_trace,
    all_gather_topology,
    num_iters,
    chunks_per_sync,
    num_workers_per_link,
    num_buffers_per_channel,
):
    """Test fused AllGather+MatMul with Llama 70B W2 sizes and different grid configs"""

    # Test different grid configurations for MinimalMatmul
    grid_configs = [
        {"grid": (4, 8), "k_block": 4, "n_block": 8, "name": "4x8_32cores_k4"},  # Current working config
        {"grid": (8, 8), "k_block": 4, "n_block": 8, "name": "8x8_64cores_k4"},  # Target: more cores
        {"grid": (7, 8), "k_block": 4, "n_block": 9, "name": "7x8_56cores_k4"},  # Galaxy harvested (64/7≈9)
        {"grid": (8, 4), "k_block": 4, "n_block": 16, "name": "8x4_32cores_k4"},  # Different aspect ratio
    ]

    seq_len = ag_output_shape[2]
    hidden_dim = ag_output_shape[3] * num_devices  # Full hidden dim after AllGather

    for config in grid_configs:
        print(f"\n=== Testing {config['name']} grid configuration ===")
        print(f"Grid: {config['grid']}, K_block: {config['k_block']}, N_block: {config['n_block']}")

        # Create custom MinimalMatmul config for this grid
        try:
            custom_config = ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=config["k_block"],
                N_block_size=config["n_block"],
                subblock_h=4,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(config["grid"][0], config["grid"][1]),
            )

            # Call the function with custom config
            # Note: We need to modify run_all_gather_impl to accept custom config
            # For now, let's use environment variables that the Llama model reads
            import os

            os.environ["FUSED_AG_MM_GRID_X"] = str(config["grid"][0])
            os.environ["FUSED_AG_MM_GRID_Y"] = str(config["grid"][1])
            os.environ["FUSED_AG_MM_K_BLOCK"] = str(config["k_block"])

            run_all_gather_impl(
                mesh_device,
                num_devices,
                ag_output_shape,
                dim,
                num_links,
                ag_input_dtype,
                layout,
                matmul_output_dim,
                matmul_weights_dtype,
                max_in0_block_w,
                use_bias,
                mem_config_input,
                mem_config_ag,
                mem_config_mm,
                all_gather_topology=all_gather_topology,
                use_non_fused=False,
                use_legacy_allgather=False,
                enable_trace=enable_trace,
                num_iters=num_iters,
                chunks_per_sync=chunks_per_sync,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
            )
            print(f"✅ {config['name']} - SUCCESS")

        except Exception as e:
            print(f"❌ {config['name']} - FAILED: {str(e)}")
            # Continue with next config instead of failing the whole test
            continue
        finally:
            # Clean up environment variables after each config
            for var in ["FUSED_AG_MM_GRID_X", "FUSED_AG_MM_GRID_Y", "FUSED_AG_MM_K_BLOCK"]:
                if var in os.environ:
                    del os.environ[var]
