# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import ttnn
from models.common.utility_functions import is_wormhole_b0

from tests.nightly.t3000.ccl.test_minimal_matmul_strided_reduce_scatter_async_sweep import (
    run_minimal_matmul_strided_reduce_scatter_impl,
    write_error_to_file,
)


def _make_fabric_router_config(max_packet_payload_size_bytes):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_packet_payload_size_bytes
    return config


# Blackhole-specific subblock sweep: sweeps subblock_h x subblock_w with 8 KiB fabric
# payload and trace-based perf measurement. Fixed config: M=9472, K=3456, N=5120, grid=12x8.
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("cluster_axis", [0], ids=["axis_0"])
@pytest.mark.parametrize(
    "mem_config_input, mem_config_mm, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
    ids=["DRAM_memconfig"],
)
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
        # (
        #     {
        #         "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        #         "fabric_router_config": _make_fabric_router_config(8192),
        #         "trace_region_size": 1531456,
        #     },
        #     ttnn.Topology.Ring,
        # ),Z
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],  # "fabric_ring_8kib_payload"],
)
def test_minimal_matmul_strided_reduce_scatter_async_bh_subblock_sweep(
    mesh_device,
    mem_config_input,
    mem_config_mm,
    mem_config_rs,
    topology,
    cluster_axis,
):
    if is_wormhole_b0():
        pytest.skip("Blackhole-only config: compute grid 12x8 exceeds wormhole_b0 limit (8x8)")
    if mesh_device.shape[cluster_axis] == 1:
        pytest.skip(f"cluster_axis={cluster_axis} has only 1 device in this mesh, reduce-scatter ring size must be > 1")

    cache_file = "test_minimal_matmul_strided_reduce_scatter_async_bh_subblock_sweep_cache.log"
    processed_cache = set()
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            processed_cache = {line.strip() for line in f}

    sub_h_range = range(1, 4 + 1)
    sub_w_range = range(1, 4 + 1)
    num_buffers_per_channel_values = [None, 24]

    with open(cache_file, "a") as f:
        for num_buffers_per_channel in num_buffers_per_channel_values:
            for sub_h in sub_h_range:
                for sub_w in sub_w_range:
                    cache_key = f"sub_h={sub_h}-sub_w={sub_w}-num_buffers_per_channel={num_buffers_per_channel}"
                    if cache_key in processed_cache:
                        continue
                    f.write(cache_key + "\n")
                    f.flush()
                    try:
                        run_minimal_matmul_strided_reduce_scatter_impl(
                            mesh_device,
                            M=9472,
                            K=3456,
                            N=5120,
                            dim=3,
                            num_links=2,
                            input_dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            mem_config_input=mem_config_input,
                            mem_config_mm=mem_config_mm,
                            mem_config_rs=mem_config_rs,
                            topology=topology,
                            enable_trace=True,
                            num_iters=2,
                            num_workers_per_link=6,
                            num_buffers_per_channel=num_buffers_per_channel,
                            mm_block_m=256,
                            mm_block_k=128,
                            mm_block_n=256,
                            subblock_h=sub_h,
                            subblock_w=sub_w,
                            mm_core_grid=ttnn.CoreCoord(9, 10),
                            chunk_width_in_mm_blocks=2,
                            rs_core_grid=ttnn.CoreRangeSet(
                                {ttnn.CoreRange(ttnn.CoreCoord(9, 0), ttnn.CoreCoord(11, 9))}
                            ),
                            rs_mode="fused",
                            cluster_axis=cluster_axis,
                            math_fidelity=ttnn.MathFidelity.HiFi2,
                            sweep_key=cache_key,
                        )
                    except Exception as e:
                        write_error_to_file(f"{cache_key} - Error: {e}")

        ttnn.ReadDeviceProfiler(mesh_device)
