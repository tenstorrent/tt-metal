import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc, comp_allclose


from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs

import torch.nn.functional as F


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
def test_mul_silu(mesh_device):
    t3k_sharding = False
    separate_silu = True

    in0_torch = torch.load("models/demos/llama3_70b_galaxy/tests/mul_silu_in0_8x4.pt")
    in1_torch = torch.load("models/demos/llama3_70b_galaxy/tests/mul_silu_in1_8x4.pt")
    ref_after_mul_torch = torch.load("models/demos/llama3_70b_galaxy/tests/ref_after_mul.pt")
    comp_out_torch = torch.load("models/demos/llama3_70b_galaxy/tests/comp_out_mul_silu.pt")

    in0_torch_32x1 = torch.load("models/demos/llama3_70b_galaxy/tests/mul_silu_in0_32x1.pt")
    in1_torch_32x1 = torch.load("models/demos/llama3_70b_galaxy/tests/mul_silu_in1_32x1.pt")

    model_args = TtQwenModelArgs(mesh_device, max_batch_size=32, dummy_weights=False, max_seq_len=128)

    # torch ref with same inputs
    in0_torch_ref = torch.permute(in0_torch_32x1, (0, 2, 1, 3)).squeeze(0)
    in1_torch_ref = torch.permute(in1_torch_32x1, (0, 2, 1, 3)).squeeze(0)
    torch_ref_silu_out = F.silu(in0_torch_ref)
    torch_ref_mul_out = torch_ref_silu_out * in1_torch_ref

    if t3k_sharding:
        in0 = ttnn.from_torch(
            in0_torch_32x1,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=[8, 4]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # memory_config=model_args.model_config["REDUCE_SCATTER_OUT_MEMCFG"]
        )
        in1 = ttnn.from_torch(
            in1_torch_32x1,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=[8, 4]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # memory_config=model_args.model_config["REDUCE_SCATTER_OUT_MEMCFG"]
        )
    else:
        in0 = ttnn.from_torch(
            in0_torch,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # memory_config=model_args.model_config["REDUCE_SCATTER_OUT_MEMCFG"]
        )
        in1 = ttnn.from_torch(
            in1_torch,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # memory_config=model_args.model_config["REDUCE_SCATTER_OUT_MEMCFG"]
        )

    if separate_silu:
        in0_silu = ttnn.silu(in0)

        if t3k_sharding:
            mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=[8, 4])
            tt_silu_out = ttnn.to_torch(in0_silu, mesh_composer=mesh_composer)[:1, :, :, :]
            tt_silu_out = torch.permute(tt_silu_out, (0, 2, 1, 3)).squeeze(0)
        else:
            # convert to torch
            composer_cfg = ttnn.MeshComposerConfig(dims=[3, 0], mesh_shape_override=ttnn.MeshShape(32, 1))
            mesh_composer = ttnn.create_mesh_composer(mesh_device, composer_cfg)
            tt_silu_out = ttnn.to_torch(in0_silu, mesh_composer=mesh_composer)[:, :, :, :]
            tt_silu_out = torch.permute(tt_silu_out, (0, 2, 1, 3)).squeeze(0)

        passing, pcc_message_silu = comp_pcc(torch_ref_silu_out, tt_silu_out)
        print(f"After silu PCC comparison torch reference: {pcc_message_silu}")
        print(comp_allclose(torch_ref_silu_out, tt_silu_out))

        out = ttnn.mul(
            in0_silu,
            in1,
            # input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=in0.memory_config(),
        )
    else:
        out = ttnn.mul(
            in0,
            in1,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=in0.memory_config(),
        )

    if t3k_sharding:
        mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=[8, 4])
        out = ttnn.to_torch(out, mesh_composer=mesh_composer)[:1, :, :, :]
        out = torch.permute(out, (0, 2, 1, 3)).squeeze(0)
    else:
        # convert to torch
        composer_cfg = ttnn.MeshComposerConfig(dims=[3, 0], mesh_shape_override=ttnn.MeshShape(32, 1))
        mesh_composer = ttnn.create_mesh_composer(mesh_device, composer_cfg)
        out = ttnn.to_torch(out, mesh_composer=mesh_composer)[:, :, :, :]
        out = torch.permute(out, (0, 2, 1, 3)).squeeze(0)

    passing, pcc_message_ref = comp_pcc(ref_after_mul_torch, out)
    print(f"After mul PCC comparison to Qwen model reference: {pcc_message_ref}")
    print(comp_allclose(ref_after_mul_torch, out))

    passing, pcc_message_torch = comp_pcc(torch_ref_mul_out, out)
    print(f"After mul PCC comparison to torch reference: {pcc_message_ref}")
    print(comp_allclose(torch_ref_mul_out, out))

    # _, pcc_message_comp = comp_pcc(comp_out_torch, out)
    # print(f"After mul PCC comparison to model output: {pcc_message_comp}")
    # print(comp_allclose(comp_out_torch, out))

    # assert passing, f"After mul PCC comparison to reference failed: {pcc_message_ref}"
