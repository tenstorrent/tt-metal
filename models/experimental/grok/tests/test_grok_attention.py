# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger

# Set Grok flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["GROK_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["GROK_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["GROK_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"

import ttnn
from ttnn import ConcatMeshToTensor
from models.experimental.grok.tt.grok_attention import TtGrokAttention
from models.experimental.grok.tt.grok_common import prepare_inputs_ttnn, prepare_rotation_mat_ttnn
from models.experimental.grok.reference.model import MultiHeadAttention
from models.experimental.grok.tt.model_config import TtModelArgs
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


def test_grok_attention_inference(t3k_mesh_device, use_program_cache, reset_seeds):
    for device in t3k_mesh_device.get_device_ids():
        t3k_mesh_device.get_device(device).enable_async(True)
    pcc = 0.99
    dtype = ttnn.bfloat8_b
    model_args = TtModelArgs(t3k_mesh_device.get_device(0), dummy_weights=os.getenv("CI") == "true")
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    key_start = "model.layers.0.attn."
    partial_state_dict = {k[len(key_start) :]: v for k, v in state_dict.items() if (k.startswith(key_start))}

    reference_model = MultiHeadAttention(
        hidden_size=model_args.hidden_size,
        num_heads=model_args.num_attention_heads,
        num_key_value_heads=model_args.num_key_value_heads,
        max_position_embeddings=model_args.max_position_embeddings,
        attn_output_multiplier=model_args.attn_output_multiplier,
        max_attn_val=model_args.max_attn_value,
    )
    reference_model.load_state_dict(partial_state_dict)

    batch = 32
    seq_len = 1  # length to generate

    tt_model = TtGrokAttention(t3k_mesh_device, state_dict, args=model_args, layer_num=0, dtype=dtype)

    rot_mat = prepare_rotation_mat_ttnn(
        model_args.head_dim,
        model_args.max_seq_len,
        tt_model.mesh_device,
    )

    generation_start_pos = 0  # Ref model can only start from pos 0
    generation_length = 10
    ref_past_key_value = None
    all_tests_pass = True

    for i in range(generation_length):
        pt_attention_input = (torch.rand(batch, seq_len, model_args.dim) * 2) - 1
        tt_attention_input = pt_attention_input
        current_pos = generation_start_pos + i

        attention_input, attn_mask = prepare_inputs_ttnn(
            tt_attention_input,
            # tt_model.hidden_size,
            model_args.dim,
            current_pos,
            tt_model.mesh_device,
        )

        tt_out = tt_model(
            attention_input,
            current_pos,
            attn_mask,
            rot_mat,
        )
        # Work around program cache issue https://github.com/tenstorrent/tt-metal/issues/7159
        del attention_input, attn_mask
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=0))[0]
            .squeeze(2)
            .view(batch, 1, -1)
        )  # [ batch, seq, hidden_dim]

        positions = torch.LongTensor([current_pos])
        reference_output, _, ref_past_key_value = reference_model(
            pt_attention_input,
            # attention_mask=None,
            past_key_value=ref_past_key_value,
            position_ids=positions,
            use_cache=True,
        )

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info(f"[current_pos={current_pos}] Grok_Attention Passed!")
        else:
            logger.warning(f"[current_pos={current_pos}] Grok_Attention Failed!")
            all_tests_pass = False
    if all_tests_pass:
        logger.info("Grok Attention output Passed!")
    else:
        logger.warning("Grok Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
