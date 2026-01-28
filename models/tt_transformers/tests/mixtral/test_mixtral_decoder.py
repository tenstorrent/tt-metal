# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer, MixtralRotaryEmbedding

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.decoder import TransformerBlock as TtTransformerBlock
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import RotarySetup

from .utils import load_hf_mixtral_config

# pytest models/tt_transformers/tests/mixtral/test_mixtral_decoder.py


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

    pcc = 0.98
    dtype = ttnn.bfloat8_b

    if batch == 32:
        generation_start_pos = 15000
        max_seq_len = 16384
    elif batch in [4, 8, 16]:
        generation_start_pos = 30000
        max_seq_len = 32768
    else:
        raise ValueError(f"Batch size {batch} not supported")

    hf_config = load_hf_mixtral_config()
    model_args = ModelArgs(mesh_device, max_seq_len=max_seq_len, max_batch_size=batch)
    model_args.use_qk_fused = False
    state_dict = model_args.load_state_dict()
    partial_state_dict = {k[9:]: v for k, v in state_dict.items() if (k.startswith(f"layers.{layer_idx}."))}
    reference_model = MixtralDecoderLayer(hf_config, layer_idx)
    reference_model.load_state_dict(convert2ref(partial_state_dict))
    reference_rotary_emb = MixtralRotaryEmbedding(config=hf_config)

    # Initialize TT model
    rope_setup = RotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling,
    )
    tt_ccl = TT_CCL(mesh_device)
    tt_model = TtTransformerBlock(
        mesh_device=mesh_device,
        state_dict=state_dict,
        args=model_args,
        layer_num=layer_idx,
        dtype=dtype,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=rope_setup.get_both_trans_mats(),
        tt_ccl=tt_ccl,
    )

    generation_length = 10
    all_tests_pass = True

    seqlen = 1

    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")

        pt_decode_input_bsh = (torch.rand(batch, seqlen, model_args.dim) * 2) - 1

        start_pos = generation_start_pos + i
        start_pos_ids = torch.tensor([start_pos for _ in range(batch)])
        current_pos_tensor = ttnn.from_torch(
            start_pos_ids,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

        tt_decode_input = pt_decode_input_bsh.clone()

        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )

        rot_mats = rope_setup.get_rot_mats(start_pos_ids)

        # Run TT model
        tt_out_b1sh = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats_global=rot_mats,
            mode="decode",
        )
        tt_out = (
            ttnn.to_torch(
                tt_out_b1sh,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
            )[0]
            .squeeze(0)
            .view(batch, 1, -1)
        )

        # Reference model
        positions = torch.LongTensor([start_pos])
        position_ids = positions.unsqueeze(0)
        position_embeddings = reference_rotary_emb(pt_decode_input_bsh, position_ids)

        ref_output_bsh, *_ = reference_model(
            hidden_states=pt_decode_input_bsh,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
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
