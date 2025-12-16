import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc, comp_allclose


from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup


@torch.no_grad()
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
def test_matmul(mesh_device):
    w_in = torch.load("models/demos/llama3_70b_galaxy/tests/w1_weight.pt")
    x_in = torch.load("models/demos/llama3_70b_galaxy/tests/w1_in.pt")
    ref_after_w1 = torch.load("models/demos/llama3_70b_galaxy/tests/ref_after_w1.pt")
    comp_out = torch.load("models/demos/llama3_70b_galaxy/tests/comp_out.pt")

    w_in = ttnn.from_torch(
        w_in,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x_in = ttnn.from_torch(
        x_in,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn.matmul(x_in, w_in, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=[8, 4])
    out = ttnn.to_torch(out, mesh_composer=mesh_composer).sum(0)
    out = torch.permute(out, (1, 0, 2))
    passing, pcc_message = comp_pcc(ref_after_w1, out)
    print(f"Non-Prefetch Matmul PCC with reference: {pcc_message}")
    print(comp_allclose(ref_after_w1, out))

    passing, pcc_message = comp_pcc(comp_out, out)
    print(f"Non-Prefetch Matmul PCC with ring matmul output: {pcc_message}")
    print(comp_allclose(comp_out, out))


@torch.no_grad()
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
def test_ring_matmul(mesh_device):
    w_in = torch.load("models/demos/llama3_70b_galaxy/tests/w1_weight.pt")
    x_in = torch.load("models/demos/llama3_70b_galaxy/tests/w1_in.pt")
    ref_after_w1 = torch.load("models/demos/llama3_70b_galaxy/tests/ref_after_w1.pt")
    comp_out = torch.load("models/demos/llama3_70b_galaxy/tests/comp_out.pt")

    RING_SIZE = 24
    model_args = TtQwenModelArgs(mesh_device, max_batch_size=32, dummy_weights=False, max_seq_len=128)
    pf_mm_out_core_range_set = model_args.pf_receiver_cores_list

    pc_1_3 = model_args.matmul_1d_ring_config(
        1,  # B
        32,  # M
        5120 // 4,  # K = 1280
        3840,  # Use padded N
        RING_SIZE,
        prefetch=False,
    )

    compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    out_memory_config = model_args.model_config["SHARDED_FF12_OUT_RING_MEMCFG"]

    in_memory_config = model_args.model_config["SHARDED_FF12_RING_MEMCFG"]

    weight_memory_config = model_args.create_dram_sharded_mem_config(
        k=1280,
        n=3840,
    )

    w_in = ttnn.from_torch(
        w_in,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=weight_memory_config,
    )

    x_in = ttnn.from_torch(
        x_in,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=in_memory_config,
    )

    out_ring_mm = ttnn.linear(
        x_in,
        w_in,
        compute_kernel_config=compute_kernel_config_hifi2,
        dtype=ttnn.bfloat8_b,
        program_config=pc_1_3,
        memory_config=out_memory_config,
        core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_1_3 else None,
    )

    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=[8, 4])
    out = ttnn.to_torch(out_ring_mm, mesh_composer=mesh_composer).sum(0)
    out = torch.permute(out, (1, 0, 2))
    passing, pcc_message = comp_pcc(ref_after_w1, out)
    print(f"Non-Prefetch Ring Matmul PCC with torch reference: {pcc_message}")
    print(comp_allclose(ref_after_w1, out))

    passing, pcc_message = comp_pcc(comp_out, out)
    print(f"Non-Prefetch Ring Matmul PCC with Prefetcher Ring Matmul output: {pcc_message}")
    print(comp_allclose(comp_out, out))


@torch.no_grad()
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
def test_prefetcher_ring_matmul(mesh_device):
    w_in = torch.load("models/demos/llama3_70b_galaxy/tests/w1_weight.pt")
    x_in = torch.load("models/demos/llama3_70b_galaxy/tests/w1_in.pt")
    ref_after_w1 = torch.load("models/demos/llama3_70b_galaxy/tests/ref_after_w1.pt")
    comp_out = torch.load("models/demos/llama3_70b_galaxy/tests/comp_out.pt")

    # prefetcher setup
    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=1,
        n_layers=1,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    RING_SIZE = 24
    model_args = TtQwenModelArgs(mesh_device, max_batch_size=32, dummy_weights=False, max_seq_len=128)
    pf_mm_out_core_range_set = model_args.pf_receiver_cores_list

    pc_1_3 = model_args.matmul_1d_ring_config(
        1,  # B
        32,  # M
        5120 // 4,  # K = 1280
        3840,  # Use padded N
        RING_SIZE,
    )

    compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    out_memory_config = model_args.model_config["SHARDED_FF12_OUT_RING_MEMCFG"]

    in_memory_config = model_args.model_config["SHARDED_FF12_RING_MEMCFG"]

    weight_memory_config = model_args.create_dram_sharded_mem_config(
        k=1280,
        n=3840,
    )

    w_in = ttnn.from_torch(
        w_in,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=weight_memory_config,
    )

    x_in = ttnn.from_torch(
        x_in,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=in_memory_config,
    )
    prefetcher_setup.insert_tensor(w_in)
    prefetcher_setup.create_global_cb()

    ttnn.dram_prefetcher(
        prefetcher_setup.get_input_tensors(),
        num_layers=1,
        global_cb=prefetcher_setup.global_circular_buffer,
    )
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

    out_ring_mm = ttnn.linear(
        x_in,
        w_in,
        compute_kernel_config=compute_kernel_config_hifi2,
        dtype=ttnn.bfloat8_b,
        program_config=pc_1_3,
        memory_config=out_memory_config,
        core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_1_3 else None,
        global_cb=prefetcher_setup.global_circular_buffer,
        sub_device_id=prefetcher_setup.worker_sub_device_id,
    )

    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=[8, 4])
    out = ttnn.to_torch(out_ring_mm, mesh_composer=mesh_composer).sum(0)
    out = torch.permute(out, (1, 0, 2))
    passing, pcc_message = comp_pcc(ref_after_w1, out)
    print(f"Prefetcher Ring Matmul PCC with reference: {pcc_message}")
    print(comp_allclose(ref_after_w1, out))

    passing, pcc_message = comp_pcc(comp_out, out)
    print(f"Prefetcher Ring Matmul PCC with Prefetcher Ring Matmul output from model: {pcc_message}")
    print(comp_allclose(comp_out, out))
