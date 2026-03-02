# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from conftest import is_6u
from tests.nightly.t3000.ccl.test_minimal_all_gather_matmul_async import run_all_gather_impl


@pytest.mark.skipif(not is_6u(), reason="This test is only for 6U devices")
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, matmul_output_dim, max_in0_block_w, matmul_weights_dtype, ag_input_dtype, use_bias",
    [
        # Llama 70B W2 sizes: 4 devices, seq_len=8192, full_hidden=3584 (896*4), output=2048
        # Input per device: (8192, 896), Weight: (3584, 2048), Output: (8192, 2048)
        (4, [1, 1, 8192, 3584], 3, ttnn.TILE_LAYOUT, 2048, 2, ttnn.bfloat16, ttnn.bfloat16, True),
    ],
    ids=["llama70b_8k_8x8grid"],
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
        (True, 10),
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
@pytest.mark.parametrize("mesh_device", [(4, 1)], indirect=True)
def test_all_gather_async(
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
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
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
        use_non_fused=False,
        use_legacy_allgather=False,
        enable_trace=enable_trace,
        num_iters=num_iters,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )
    ttnn.ReadDeviceProfiler(submesh_device)
