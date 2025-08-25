import pytest
import ttnn
import torch
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
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
def test_qwen_tensor_cache(mesh_device):
    mode = "decode"

    model_args = TtQwenModelArgs(mesh_device, max_batch_size=32, dummy_weights=False, max_seq_len=128)

    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    good_cache = "model_cache/Qwen/Qwen3-32B/TG/tensor_cache_bfp8/layers.0.feed_forward.w1_shardedprefetcher"
    bad_cache = "model_cache/Qwen/Qwen3-32B/TG/tensor_cache_bfp8/layers.0.feed_forward.w1_sharded"

    torch_weight = lambda name: torch.transpose(state_dict[f"layers.0.feed_forward.{name}.weight"], -2, -1)

    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=2,
        n_layers=1,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, use_qwen_mlp=True)
    prefetcher_setup.create_global_cb()

    x = torch.randn(1, 1, 32, model_args.dim)
    x_tt = ttnn.from_torch(
        x,
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
        )
        if mode == "decode"
        else ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    good_w1 = ttnn.as_tensor(
        torch_weight("w1").unsqueeze(0).unsqueeze(0),  # Grab only the wX part of the name
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(None, 3) if model_args.is_galaxy else (None, None), mesh_shape=model_args.cluster_shape
        ),
        layout=ttnn.TILE_LAYOUT,
        memory_config=model_args.model_config["W1W3_RING_MEMCFG"],
        cache_file_name=good_cache,
    )

    bad_w1 = ttnn.as_tensor(
        torch_weight("w1").unsqueeze(0).unsqueeze(0),  # Grab only the wX part of the name
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(None, 3) if model_args.is_galaxy else (None, None), mesh_shape=model_args.cluster_shape
        ),
        layout=ttnn.TILE_LAYOUT,
        memory_config=model_args.model_config["W1W3_RING_MEMCFG"],
        cache_file_name=bad_cache,
    )

    prefetcher_setup.insert_tensor(good_w1)
    prefetcher_setup.insert_tensor(bad_w1)

    ttnn.dram_prefetcher(
        prefetcher_setup.get_input_tensors(),
        num_layers=1,
        global_cb=prefetcher_setup.global_circular_buffer,
    )
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

    breakpoint()

    out_good = ttnn.linear(
        x_tt,
        good_w1,
        compute_kernel_config=model_args.compute_kernel_config_lofi,
        dtype=ttnn.bfloat8_b,
        program_config=model_args.model_config["FF1_3_TG_RING_PROGCFG"],
        memory_config=model_args.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
        global_cb=prefetcher_setup.global_circular_buffer if model_args.model_config["USE_PREFETCHER"] else None,
        sub_device_id=prefetcher_setup.worker_sub_device_id if mode == "decode" else None,
    )

    breakpoint()

    out_bad = ttnn.linear(
        x_tt,
        bad_w1,
        compute_kernel_config=model_args.compute_kernel_config_lofi,
        dtype=ttnn.bfloat8_b,
        program_config=model_args.model_config["FF1_3_TG_RING_PROGCFG"],
        memory_config=model_args.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
        global_cb=prefetcher_setup.global_circular_buffer if model_args.model_config["USE_PREFETCHER"] else None,
        sub_device_id=prefetcher_setup.worker_sub_device_id if mode == "decode" else None,
    )

    breakpoint()
