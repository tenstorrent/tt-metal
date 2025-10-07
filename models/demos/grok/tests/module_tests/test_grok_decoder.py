# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.grok.reference.llama_clone import GrokDecoder as ReferenceGrokDecoder
from models.demos.grok.tt.ccl import CCL_Manager
from models.demos.grok.tt.decoder import Decoder
from models.demos.grok.tt.model_config import TtModelArgs
from models.tt_transformers.tt.common import PagedAttentionConfig, precompute_freqs
from models.tt_transformers.tt.rope import RotarySetup


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "paged_attention",
    (True,),
    ids=("paged_attention",),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 64, "page_max_num_blocks": 2048}],
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_grok_decoder_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b
    pcc = 0.99

    model_args = TtModelArgs(mesh_device)
    model_args.n_layers = 1  # For the unit test, just run a single layer

    # Load state dict for both attention and MLP/MoE components
    state_dict = model_args.load_weights_to_state_dict_no_experts()
    state_dict = model_args.load_experts_weights_to_state_dict(state_dict)

    # Load weights for reference model components
    layer_num = 0
    model_args.prune_experts_except_layers(state_dict, [layer_num])

    # Create reference model
    reference_model = ReferenceGrokDecoder()

    # Load attention weights
    attention_prefix = model_args.get_state_dict_prefix("Attention", layer_num)
    attention_state_dict = {
        k[len(attention_prefix) + 1 :]: v for k, v in state_dict.items() if (k.startswith(attention_prefix))
    }
    reference_model.attention.load_state_dict(attention_state_dict)

    # Load shared MLP weights
    mlp_prefix = model_args.get_state_dict_prefix("MLP", layer_num)
    mlp_state_dict = {k[len(mlp_prefix) + 1 :]: v for k, v in state_dict.items() if (k.startswith(mlp_prefix))}
    reference_model.shared_mlp.load_state_dict(mlp_state_dict)

    # Load MoE weights (gate + experts)
    moe_prefix = f"model.layers.{layer_num}.block_sparse_moe"

    # # Load gate weights
    gate_weight = state_dict[f"{moe_prefix}.gate.weight"]
    reference_model.moe.gate.weight.data = gate_weight

    # Load expert weights
    for expert_idx in range(8):  # 8 experts
        expert_prefix = f"{moe_prefix}.experts.{expert_idx}"
        expert_state_dict = {
            k[len(expert_prefix) + 1 :]: v for k, v in state_dict.items() if k.startswith(expert_prefix)
        }
        reference_model.moe.experts[expert_idx].load_state_dict(expert_state_dict)

    reference_model.pre_attn_norm.weight = torch.nn.Parameter(
        state_dict[f"model.layers.{layer_num}.pre_attn_norm.weight"]
    )
    reference_model.post_attn_norm.weight = torch.nn.Parameter(
        state_dict[f"model.layers.{layer_num}.post_attn_norm.weight"]
    )
    reference_model.pre_moe_norm.weight = torch.nn.Parameter(
        state_dict[f"model.layers.{layer_num}.pre_moe_norm.weight"]
    )
    reference_model.post_moe_norm.weight = torch.nn.Parameter(
        state_dict[f"model.layers.{layer_num}.post_moe_norm.weight"]
    )

    seq_len = 1
    generation_start_pos = 0
    generation_length = 10
    all_tests_pass = True

    # Setup RoPE transformation matrices
    rope_setup = RotarySetup(
        mesh_device,
        batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        None,  # No rope scaling for Grok
    )

    transformation_mats = rope_setup.get_both_trans_mats()

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
        page_table = reverse_permutation.reshape(batch_size, paged_attention_config.max_num_blocks // batch_size)
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, -2) if (model_args.num_devices == 32 and batch_size > 1) else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    tt_ccl = CCL_Manager(mesh_device)
    tt_model = Decoder(
        mesh_device,
        tt_ccl,
        state_dict,
        weight_cache_path=None,  # Use dummy weights
        args=model_args,
        layer_num=layer_num,
        dtype=dtype,
        transformation_mats=transformation_mats,
        paged_attention_config=paged_attention_config,
    )

    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
        None,  # No rope scaling factor
        None,  # No original max position embeddings
    )
    freqs_cis = torch.complex(cos, sin)

    # Initial positions
    current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.num_devices == 32 and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    for i in range(generation_length):
        # Grok decoder block typically sees tensors with mean 0 and std 0.03 - 0.05 in layer 1
        pt_decoder_input = torch.randn(
            batch_size, seq_len, model_args.dim, dtype=torch.float32
        )  # Use float32 for reference

        tt_decoder_input = pt_decoder_input.clone()

        decoder_input = model_args.prepare_residual_tensor_decode(
            # tt_decoder_input, model_args.model_config["SHARDED_ATTN_INPUT_MEMCFG"], force_replicated=False
            tt_decoder_input,
            model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
            force_replicated=False,
        )

        # Get cos/sin matrices for the current position of each user
        rot_mats = rope_setup.get_rot_mats(current_pos)

        tt_out = tt_model(
            decoder_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            page_table=page_table_tt,
        )
        # multi-device decoder module returns replicated output
        tt_out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )
        tt_output_torch = tt_out[:, 0:1, :batch_size, : model_args.dim].view(-1, 1, model_args.dim)

        # In this test all users have the same position (if using batch > 1)
        freqs_cis_i = freqs_cis[current_pos[0], :].unsqueeze(0)

        reference_output = reference_model(pt_decoder_input, current_pos[0], freqs_cis_i, mask=None)
        breakpoint()

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
        if passing:
            logger.info(f"[pos={current_pos[0]}] Grok Decoder Passed!")
        else:
            logger.warning(f"[pos={current_pos[0]}] Grok Decoder Failed!")
            all_tests_pass = False

        # Increment position
        current_pos = torch.tensor([generation_start_pos + i + 1 for _ in range(batch_size)])
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 0) if (model_args.num_devices == 32 and batch_size > 1) else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    if all_tests_pass:
        logger.info("Grok Decoder output Passed!")
    else:
        logger.warning("Grok Decoder output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
