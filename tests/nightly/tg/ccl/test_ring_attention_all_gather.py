# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.nightly.t3000.ccl.test_ring_attention_all_gather import (
    run_ring_attention_all_gather_impl,
    create_ring_attention_submesh,
)


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("num_links", [1, 2, 3], ids=["1link", "2link", "3link"])
@pytest.mark.parametrize("layout, ag_input_dtype", [(ttnn.TILE_LAYOUT, ttnn.bfloat16)])
@pytest.mark.parametrize(
    "ag_output_shape, ag_num_inputs, rp_axis, rp_factor, up_factor",
    [
        ([1, 40, 4096, 64], 2, 1, 4, 4),
        ([1, 40, 4096, 64], 2, 1, 2, 8),
        ([1, 40, 4096, 64], 2, 0, 8, 2),
    ],
    ids=[
        "2input_rp4_up4",
        "2input_rp2_up8",
        "2input_rp8_up2",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    ids=["line"],
    indirect=["device_params"],
)
def test_ring_attention_all_gather(
    mesh_device,
    ag_output_shape,
    ag_num_inputs,
    rp_axis,
    rp_factor,
    up_factor,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
    all_gather_topology,
):
    submesh_device = create_ring_attention_submesh(mesh_device, rp_axis, rp_factor, up_factor)

    run_ring_attention_all_gather_impl(
        submesh_device,
        ag_output_shape,
        ag_num_inputs,
        rp_axis,
        rp_factor,
        up_factor,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
    )


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("num_links", [1, 2, 3], ids=["1link", "2link", "3link"])
@pytest.mark.parametrize("layout, ag_input_dtype", [(ttnn.TILE_LAYOUT, ttnn.bfloat16)])
@pytest.mark.parametrize(
    "ag_output_shape, ag_num_inputs, rp_axis, rp_factor, up_factor",
    [
        ([1, 5, 4096, 64], 2, 1, 4, 1),
        ([1, 5, 4096, 64], 2, 0, 2, 1),
    ],
    ids=[
        "shape2_2input_rp4",
        "shape2_2input_rp2",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (False, 1),
    ],
    ids=["check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    ids=[
        "line",
    ],
    indirect=["device_params"],
)
def test_ring_attention_all_gather_program_cache(
    mesh_device,
    ag_output_shape,
    ag_num_inputs,
    rp_axis,
    rp_factor,
    up_factor,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
    all_gather_topology,
):
    submesh_device = create_ring_attention_submesh(mesh_device, rp_axis, rp_factor, up_factor)

    dummy_tensors = []
    for i in range(3):
        dummy_tensors.append(
            ttnn.from_torch(
                torch.rand(ag_output_shape),
                device=submesh_device,
                layout=layout,
                dtype=ag_input_dtype,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    submesh_device, mesh_shape=tuple(submesh_device.shape), dims=[None, None]
                ),
            )
        )
        run_ring_attention_all_gather_impl(
            submesh_device,
            ag_output_shape,
            ag_num_inputs,
            rp_axis,
            rp_factor,
            up_factor,
            num_links,
            ag_input_dtype,
            layout,
            mem_config_input,
            mem_config_ag,
            all_gather_topology=all_gather_topology,
            enable_trace=enable_trace,
            num_iters=num_iters,
        )
        ttnn.synchronize_device(submesh_device)

    assert submesh_device.num_program_cache_entries() == 1
