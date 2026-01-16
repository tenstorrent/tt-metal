# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralRotaryEmbedding,
    create_causal_mask,
    create_sliding_window_causal_mask,
)

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import get_prefill_rot_mat, get_rot_transformation_mat
from models.tt_transformers.tt.decoder import TransformerBlock as TtTransformerBlock
from models.tt_transformers.tt.model_config import ModelArgs

from .utils import load_hf_mixtral_config

# pytest models/tt_transformers/tests/mixtral/test_mixtral_decoder_prefill.py


def convert2ref(state_dict):
    """Convert state dict keys for compatibility with reference model naming."""
    replacements = [
        ("attention.wq.weight", "self_attn.q_proj.weight"),
        ("attention.wk.weight", "self_attn.k_proj.weight"),
        ("attention.wv.weight", "self_attn.v_proj.weight"),
        ("attention.wo.weight", "self_attn.o_proj.weight"),
        ("attention_norm.weight", "input_layernorm.weight"),
        ("ffn_norm.weight", "post_attention_layernorm.weight"),
    ]

    out = {}
    for k, v in state_dict.items():
        new_k = k
        for old, new in replacements:
            if k.startswith(old):
                new_k = k.replace(old, new)
                break
        out[new_k] = v
    return out


@pytest.mark.parametrize("layer_idx", [0])
@pytest.mark.parametrize(
    "batch",
    (32,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_mixtral_decoder_inference(mesh_device, reset_seeds, batch, device_params, layer_idx):
    """
    b: batch
    s: sequence length
    h: hidden size
    """

    pcc = 0.99
    dtype = ttnn.bfloat8_b
    mode = "prefill"
    batch = 1
    max_seq_len = 4096

    hf_config = load_hf_mixtral_config()
    model_args = ModelArgs(mesh_device, max_seq_len=max_seq_len, max_batch_size=batch)
    model_args.n_layers = 1

    mesh_device.disable_and_clear_program_cache()

    state_dict = model_args.load_state_dict()
    first_layer_prefix = model_args.get_state_dict_prefix("TransformerBlock", layer_idx)
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    reference_model = MixtralDecoderLayer(hf_config, layer_idx=layer_idx)
    reference_model.load_state_dict(convert2ref(partial_state_dict))
    reference_rotary_emb = MixtralRotaryEmbedding(config=hf_config)

    # Initialize TT model
    rot_mats = get_prefill_rot_mat(
        model_args.head_dim,
        mesh_device,
        max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling.factor if model_args.rope_scaling else None,
        model_args.max_context_len,
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

    tt_ccl = TT_CCL(mesh_device)
    tt_model = TtTransformerBlock(
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=layer_idx,
        dtype=dtype,
        transformation_mats=transformation_mats,
        args=model_args,
        tt_ccl=tt_ccl,
    )

    generation_length = 10
    all_tests_pass = True
    generation_start_pos = 0

    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")
        pt_decode_input_bsh = (torch.rand(batch, max_seq_len, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input_bsh.clone()
        decode_input = model_args.prepare_residual_tensor_prefill(
            tt_decode_input,
        )

        # Run TT model
        start_pos = generation_start_pos + i
        start_pos_ids = [start_pos for _ in range(batch)]
        tt_out_b1sh = tt_model(
            decode_input,
            None,
            rot_mats,
            user_id=0,
            mode=mode,
        )
        tt_out = (
            ttnn.to_torch(
                tt_out_b1sh,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
            )[0]
            .squeeze(0)
            .view(batch, max_seq_len, -1)
        )

        # Reference model
        positions = torch.LongTensor(range(max_seq_len))
        # Causal mask generated as in HF Mixtral model: https://github.com/huggingface/transformers/blob/a7f29523361b2cc12e51c1f5133d95f122f6f45c/src/transformers/models/mixtral/modeling_mixtral.py#L473
        mask_function = create_causal_mask if hf_config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=hf_config,
            input_embeds=pt_decode_input_bsh,
            attention_mask=None,
            cache_position=positions,
            past_key_values=None,
        )

        position_ids = positions.unsqueeze(0).expand(batch, -1)
        position_embeddings = reference_rotary_emb(pt_decode_input_bsh, position_ids)

        ref_output_bsh, *_ = reference_model(
            hidden_states=pt_decode_input_bsh,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

        # Reference model
        passing, pcc_message = comp_pcc(ref_output_bsh, tt_out, pcc)

        logger.info(comp_allclose(ref_output_bsh, tt_out))
        logger.info(pcc_message)

        if passing:
            logger.info("Mixtral Decoder Block Passed!")
        else:
            logger.warning("Mixtral Decoder Block Failed!")
            all_tests_pass = False

    if all_tests_pass:
        logger.info(f"All {generation_length} Mixtral decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Mixtral decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
