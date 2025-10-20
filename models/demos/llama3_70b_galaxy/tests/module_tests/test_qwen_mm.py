import pytest

import ttnn
import torch
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_qwen_mlp_rs(mesh_device):
    dim = 5120
    hidden_dim = 25600
    # dim = 8192
    # hidden_dim = 28*1024

    # model_args = TtModelArgs(mesh_device, max_batch_size=1, dummy_weights=False, max_seq_len=128)
    model_args = TtQwenModelArgs(mesh_device, max_batch_size=1, dummy_weights=False, max_seq_len=128)
    model_config = model_args.get_model_config()

    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=3,
        n_layers=1,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, use_qwen_mlp=True)

    pc_1_3 = model_config["FF1_3_TG_RING_PROGCFG"]

    as_sharded_tensor = lambda torch_tensor, type, dim: ttnn.as_tensor(
        torch_tensor.unsqueeze(0).unsqueeze(0),  # Grab only the wX part of the name
        dtype=type,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=dim, mesh_shape=model_args.cluster_shape),
        layout=ttnn.TILE_LAYOUT,
        memory_config=model_config["W1W3_RING_MEMCFG"],
        # cache_file_name=cache_name(name),
    )
    as_sharded_tensor_w2 = lambda torch_tensor, type, dim: ttnn.as_tensor(
        torch_tensor.unsqueeze(0).unsqueeze(0),  # Grab only the wX part of the name
        dtype=type,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=dim, mesh_shape=model_args.cluster_shape),
        layout=ttnn.TILE_LAYOUT,
        memory_config=model_config["W2_RING_MEMCFG"],
        # cache_file_name=cache_name(name),
    )

    w1_torch = torch.randn((dim, hidden_dim))
    w1 = as_sharded_tensor(w1_torch, ttnn.bfloat8_b, dim=(-1, -2))

    w3_torch = torch.randn((dim, hidden_dim))
    w3 = as_sharded_tensor(w3_torch, ttnn.bfloat8_b, dim=(-1, -2))

    w2_torch = torch.randn((hidden_dim, dim))
    w2 = as_sharded_tensor_w2(w2_torch, ttnn.bfloat8_b, dim=(-2, -1))

    prefetcher_setup.insert_tensor(w1)
    prefetcher_setup.insert_tensor(w3)
    prefetcher_setup.insert_tensor(w2)

    prefetcher_setup.create_global_cb()
    ttnn.dram_prefetcher(
        prefetcher_setup.get_input_tensors(),
        num_layers=1,
        global_cb=prefetcher_setup.global_circular_buffer,
    )
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

    x_torch = torch.randn(
        (1, 1, 32, dim),
    )
    x = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 3) if model_args.is_galaxy else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),  # When both dims are None, the mapper used is `ReplicateTensorToMesh`
        dtype=ttnn.bfloat8_b,
        memory_config=(
            model_args.model_config["SHARDED_FF12_RING_MEMCFG"]
            if model_args.is_galaxy
            else model_args.model_config["SHARDED_MLP_INPUT_MEMCFG"]
        ),
        layout=ttnn.TILE_LAYOUT,
    )

    # xt = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    # xtt = ttnn.to_torch(x ,mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),)
    # xtt0 = xtt[:,:1,:,:]

    # w1t = ttnn.to_torch(w1, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 2), mesh_shape=model_args.cluster_shape))

    # breakpoint()

    w1_out = ttnn.linear(
        x,
        w1,
        compute_kernel_config=model_args.compute_kernel_config_hifi2,
        dtype=ttnn.bfloat8_b,
        program_config=pc_1_3,
        memory_config=model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
        global_cb=prefetcher_setup.global_circular_buffer,
        sub_device_id=prefetcher_setup.worker_sub_device_id,
    )

    # breakpoint()

    # w1_out_torch = ttnn.to_torch(w1_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

    breakpoint()

    # x -> [1, 1, 32, 5120 // 4 = 1280]
    # w1 -> [1, 1, 5120 // 4 = 1280, 25600 // 8 = 3200]
