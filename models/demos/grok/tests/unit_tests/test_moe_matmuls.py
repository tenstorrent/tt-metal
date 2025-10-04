import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.grok.tt.model_config import TtModelArgs
from models.tt_transformers.tt.ccl import TT_CCL


@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_moe_matmuls(mesh_device):
    model_args = TtModelArgs(mesh_device)
    model_args.n_layers = 1

    tt_ccl = TT_CCL(mesh_device)

    x_torch = torch.randn(1, 8, 32, 8192)
    tt_input = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 3),
            mesh_shape=(8, 4),
        ),  # When both dims are None, the mapper used is `ReplicateTensorToMesh`
        dtype=ttnn.bfloat8_b,
        # memory_config=model_args.model_config["MLP_ACT_MEMCFG"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    w1_torch = torch.randn(1, 8, 8192, 16384)
    w1 = ttnn.from_torch(
        w1_torch,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, -2), mesh_shape=(8, 4)),
    )

    pc_1 = model_args.model_config["FF1_3_TG_PROGCFG_SINGLE_EXPERT"]
    # w1_out = ttnn.linear(
    #     tt_input,
    #     w1,
    #     dtype=ttnn.bfloat8_b,  # TG=True
    #     core_grid=None,
    #     compute_kernel_config=model_args.compute_kernel_config_hifi2,
    #     program_config=pc_1,
    #     memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    # )
    w1_out = ttnn.matmul(tt_input, w1)
    w1_out_temp = ttnn.to_torch(
        w1_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=(8, 4))
    )

    input_mem_cfg = w1_out.memory_config()
    w1_out = ttnn.experimental.reduce_scatter_minimal_async(
        w1_out,
        persistent_output_buffers=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(1),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(1),
        num_links=model_args.num_reduce_scatter_links,
        cluster_axis=1,
        memory_config=model_args.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG_SINGLE_EXPERT"],  # decode mode
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )
    w1_out_torch = ttnn.to_torch(w1_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    ref_w1_out = x_torch @ w1_torch
    print(comp_pcc(ref_w1_out, w1_out_torch))
    breakpoint()
