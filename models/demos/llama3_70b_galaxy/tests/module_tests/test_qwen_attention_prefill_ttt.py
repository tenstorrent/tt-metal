# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import precompute_freqs_cis
from models.demos.llama3_70b_galaxy.tt.llama_attention import TtLlamaAttention
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.tt_transformers.tt.model_config import ModelArgs
from models.demos.llama3_70b_galaxy.tt.llama_common import (
    get_prefill_rot_mat,
    get_rot_transformation_mat,
    precompute_freqs,
    PagedAttentionConfig,
)
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Attention
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from models.common.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
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
# Model and attention prefill tests should run both with and without paged attention to debug any issues that may occur with default attention
@pytest.mark.parametrize(
    "paged_attention",
    (
        True,
        # False,
    ),
    ids=(
        "paged_attention",
        # "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 64, "page_max_num_blocks": 4096}],
)
@pytest.mark.parametrize(
    "max_seq_len",
    (
        128,
        # 2048,
        # 4096,
        # 1024 * 32,
        # 1024 * 64,
    ),
)
def test_qwen_attention_inference_prefill_ttt(
    max_seq_len,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b
    pcc = 0.99
    batch_size = 1  # For prefill we only support batch_size = 1

    # Load tt_transformers reference model args for reference attention
    model_args_ref = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, cache_hf=True)
    model_args_ref.n_layers = 1  # For the unit test, just run a single layer

    state_dict_ref = model_args_ref.load_state_dict()

    first_layer_prefix_ref = model_args_ref.get_state_dict_prefix("Attention", 0) + "."
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict_ref = {
        k[len(first_layer_prefix_ref) :]: v for k, v in state_dict_ref.items() if k.startswith(first_layer_prefix_ref)
    }

    # Use tt_transformers reference attention
    reference_model = model_args_ref.reference_attention()
    reference_model.load_state_dict(partial_state_dict_ref)
    logger.info(f"tt_transformers Reference Model Loaded")

    # Load Qwen3 model using TtQwenModelArgs
    model_args = TtQwenModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, dummy_weights=False)
    model_args.n_layers = 1
    model_args.use_prefetcher = False
    state_dict = model_args.load_state_dict()
    logger.info(f"Qwen3 Model Loaded")

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("TtLlamaAttention", 0) + "."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    reference_model_custom = Attention(args=model_args, llama3=False)  # Enable QK norm with llama3=True
    reference_model_custom.load_state_dict(partial_state_dict)
    logger.info(f"Reference Model Loaded with QK norm support")

    # pre-compute the rotational embedding matrix and send to device
    rot_mats = get_prefill_rot_mat(
        model_args.head_dim,
        model_args.max_seq_len,
        mesh_device,
        seq_len=max_seq_len,
        scale_factor=model_args.rope_scaling_factor,
    )
    transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)
    transformation_mats_prefill = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    transformation_mats = {"prefill": transformation_mats_prefill}

    generation_start_pos = 0
    generation_length = 3
    all_tests_pass = True

    # Setup page table
    page_table_tt = None
    paged_attention_config = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode="prefill")
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, mode="prefill", is_qwen=True)

    tt_model = TtLlamaAttention(
        mesh_device,
        state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        configuration=model_args,
        paged_attention_config=paged_attention_config,
        prefetcher_setup=prefetcher_setup,
        tt_ccl=tt_ccl,
    )

    # Setup freqs for Qwen
    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )
    freqs_cis = torch.complex(cos, sin)

    # Setup freqs for reference model (tt_transformers)
    from models.tt_transformers.tt.common import precompute_freqs as tt_precompute_freqs

    cos_ref, sin_ref = tt_precompute_freqs(
        model_args_ref.head_dim,
        model_args_ref.max_seq_len * 2,
        model_args_ref.rope_theta,
        model_args_ref.rope_scaling.factor if model_args_ref.rope_scaling else None,
        model_args_ref.rope_scaling.original_max_position_embeddings if model_args_ref.rope_scaling else None,
    )
    freqs_cis_ref = torch.complex(cos_ref, sin_ref)

    pt_attention_input = (
        torch.randn(
            batch_size,
            max_seq_len,
            model_args.dim,
            dtype=get_ref_model_dype(reference_model, model_args_ref.model_name),
        )
        * 0.05
    )
    tt_attention_input = pt_attention_input.clone()

    for _ in range(2):
        attention_input = model_args.prepare_residual_tensor_prefill(
            tt_attention_input,
            force_replicated=False if model_args.is_galaxy else True,
        )

        tt_out, tt_x_qkv = tt_model(
            attention_input,
            current_pos=None,
            rot_mats=rot_mats,
            user_id=0,
            mode="prefill",
            page_table=page_table_tt,
        )
        tt_out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )
        tt_output_torch = tt_out[:, 0:1, :, : model_args.dim].view(
            batch_size, max_seq_len, -1
        )  # [ batch, seq, hidden_dim]

        tt_x_qkv = ttnn.to_torch(
            tt_x_qkv,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 1), mesh_shape=model_args.cluster_shape),
        )  # [1, 1, seq_len, 8 * 1280 = 10240]
        tt_x_qkv_torch = tt_x_qkv[:, 0, :, :]  # [ batch, seq, hidden_dim]

        positions = torch.LongTensor(range(max_seq_len))
        freqs_cis_i_ref = precompute_freqs_cis(
            model_args_ref.head_dim,
            model_args_ref.max_seq_len * 2,
            model_args_ref.rope_theta,
            model_args_ref.rope_scaling.factor if model_args_ref.rope_scaling else None,
        )[positions]
        attn_mask = torch.full((max_seq_len, max_seq_len), torch.finfo(torch.float32).min)
        attn_mask_torch = torch.triu(attn_mask, diagonal=1)

        breakpoint()

        # Use tt_transformers reference model
        reference_output = reference_model(
            pt_attention_input.to(torch.bfloat16), positions[0], freqs_cis_i_ref, mask=attn_mask_torch
        )

        # Use custom reference model for comparison
        reference_output_custom, x_qkv = reference_model_custom(
            pt_attention_input, positions[0], freqs_cis_i_ref, mask=attn_mask_torch
        )

        # Verify both reference models match
        passing_ref, pcc_message_ref = comp_pcc(reference_output, reference_output_custom, pcc)
        logger.info(f"Reference models PCC: {pcc_message_ref}")

        # Compare TT output with tt_transformers reference
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        # Process QKV outputs for comparison
        xq = x_qkv[0].permute(0, 2, 1, 3).view(max_seq_len, 64 * 128)
        xk = x_qkv[1].view(max_seq_len, 8 * 128)
        xv = x_qkv[2].view(max_seq_len, 8 * 128)

        concat_x_qkv = torch.cat([xq, xk, xv], dim=1).view(1, max_seq_len, 10240)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
        logger.info(f"PCC xqkv: {pcc_message_xqkv}")

    if passing:
        logger.info(f"Qwen_Attention Prefill TTT Passed!")
    else:
        logger.warning(f"Qwen_Attention Prefill TTT Failed!")
        all_tests_pass = False

    tt_ccl.close()
    if all_tests_pass:
        logger.info("Qwen Attention Prefill TTT output Passed!")
    else:
        logger.warning("Qwen Attention Prefill TTT output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
