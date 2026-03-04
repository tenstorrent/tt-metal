# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from conftest import is_6u
from tests.nightly.t3000.ccl.test_minimal_all_gather_matmul_async import run_all_gather_impl


@pytest.mark.skipif(not is_6u(), reason="This test is only for 6U devices")
@pytest.mark.parametrize("num_links", [3], ids=["3links"])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, matmul_output_dim, max_in0_block_w, matmul_weights_dtype, ag_input_dtype, use_bias, test_name",
    [
        # EXACT BASELINE MATCH: 8192x896 → 8192x3584 → 8192x2048 (from your profiler results)
        (8, [1, 1, 8192, 896], 3, ttnn.TILE_LAYOUT, 2048, 2, ttnn.bfloat16, ttnn.bfloat16, False, "baseline_8k_exact"),
        # Tiny 2K test - FUSED version
        (8, [1, 1, 2048, 2048], 3, ttnn.TILE_LAYOUT, 2048, 2, ttnn.bfloat16, ttnn.bfloat16, False, "tiny_2k_fused"),
        # Tiny 2K test - BASELINE (separate AG+MM) for comparison
        (8, [1, 1, 2048, 2048], 3, ttnn.TILE_LAYOUT, 2048, 2, ttnn.bfloat16, ttnn.bfloat16, False, "tiny_2k_baseline"),
    ],
    ids=["baseline_8k_exact", "tiny_2k_fused", "tiny_2k_baseline"],
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
    ids=["dram_config"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (False, 1),  # Correctness check
        (True, 5),  # Performance measurement
    ],
    ids=["check", "perf"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 180224}, ttnn.Topology.Ring),
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
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_llama_all_gather_matmul_async(
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
    test_name,
):
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))

    # Use non-fused (separate AG+MM) for baseline comparison
    use_non_fused_flag = True if test_name == "tiny_2k_baseline" else False

    run_all_gather_impl(
        submesh_device,
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
        use_non_fused=use_non_fused_flag,
        use_legacy_allgather=False,
        enable_trace=enable_trace,
        num_iters=num_iters,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )
    ttnn.ReadDeviceProfiler(submesh_device)
