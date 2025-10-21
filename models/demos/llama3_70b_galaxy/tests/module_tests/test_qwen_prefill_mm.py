import pytest
import torch
import ttnn
import math
from loguru import logger

from models.common.utility_functions import (
    comp_pcc,
    comp_allclose,
)


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
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
def test_qwen_prefill_mm(mesh_device):
    max_seq_len = 2048

    def prepare_residual_tensor_prefill(x_bsh, force_replicated=False):
        """
        Prepare inputs for prefill mode.
        x: (batch, seq, hidden_dim)
        B: batch (1)
        S: sequence len
        H: dim
        """

        x_1BSH = x_bsh.unsqueeze(0)
        dims = (None, None) if force_replicated else (None, -1)

        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=(8, 4))

        # input goes to DRAM
        xs_1BSH = ttnn.from_torch(
            x_1BSH,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        return xs_1BSH

    pt_attention_input = (torch.rand(1, max_seq_len, 5120) * 2) - 1
    tt_attention_input = pt_attention_input.clone()
    attention_input = prepare_residual_tensor_prefill(tt_attention_input, force_replicated=False)

    qkv_cat = torch.rand(1, 1, 5120, 10240)
    wqkv_interleaved = ttnn.as_tensor(
        qkv_cat,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2), mesh_shape=(8, 4)),
    )

    torch_ref = pt_attention_input @ qkv_cat

    xqkv = ttnn.linear(
        attention_input,
        wqkv_interleaved,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            dst_full_sync_en=True,
        ),
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 10),
            in0_block_w=5,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=2,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=max(
                1, 8 if max_seq_len >= 2048 else max_seq_len // 32 // 8  # 8 rows
            ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=math.ceil(1280 / 32 / 7),  # N / TILE_WIDTH / grid width
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=max_seq_len <= 2048,
        ),
    )

    xqkv_torch = ttnn.to_torch(
        xqkv, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 1), mesh_shape=(8, 4))
    )
    xqkv_torch = xqkv_torch.sum(dim=1, keepdim=True)

    passing, pcc_message = comp_pcc(torch_ref, xqkv_torch)
    logger.info(comp_allclose(torch_ref, xqkv_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("Qwen_Prefill MM Passed!")
    else:
        logger.warning("Qwen_Prefill MM Failed!")
