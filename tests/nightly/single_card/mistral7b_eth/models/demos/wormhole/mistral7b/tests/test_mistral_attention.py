# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.wormhole.mistral7b.tt.mistral_attention import TtMistralAttention
from models.demos.wormhole.mistral7b.tt.model_config import TtModelArgs
from models.demos.wormhole.mistral7b.tt.mistral_common import (
    precompute_freqs,
    prepare_inputs_ttnn,
    freqs_to_rotation_matrix,
)
from models.demos.wormhole.mistral7b.reference.model import Attention
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
def test_mistral_attention_inference(device, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat8_b
    pcc = 0.99

    model_args = TtModelArgs(device)
    state_dict = torch.load(model_args.consolidated_weights_path)

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[19:]: v for k, v in state_dict.items() if (k.startswith("layers.0.attention."))}

    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    batch = model_args.max_batch_size
    seq_len = 1

    # pre-compute the rotational embedding matrix and send to device
    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    rot_emb_matrix = freqs_to_rotation_matrix(cos, sin)

    rot_emb_matrix_list = []
    for i in range(rot_emb_matrix.shape[0]):
        rot_emb_matrix_list.append(
            ttnn.from_torch(
                rot_emb_matrix[i, :, :].unsqueeze(0).unsqueeze(0), device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT
            )
        )  # ttnn.bfloat16

    generation_start_pos = 0
    generation_length = 3
    all_tests_pass = True

    tt_model = TtMistralAttention(
        [device],
        state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        configuration=model_args,
        rot_mat=rot_emb_matrix_list,
        start_pos=generation_start_pos,
    )

    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    freqs_cis = torch.complex(cos, sin)

    for i in range(generation_length):
        pt_attention_input = (torch.rand(batch, seq_len, model_args.dim) * 2) - 1
        tt_attention_input = pt_attention_input.clone()
        current_pos = generation_start_pos + i
        attention_input, pos = prepare_inputs_ttnn(
            tt_attention_input,
            current_pos,
            model_args.dim,
            model_args.sliding_window,
            device,
        )

        tt_out = tt_model(
            [attention_input],
            pos,
        )
        # multi-device attention module returns replicated output
        assert isinstance(tt_out, list)
        tt_out = tt_out[0]
        tt_output_torch = ttnn.to_torch(tt_out).permute(1, 0, 2)[
            : model_args.max_batch_size, :, :
        ]  # [ batch, seq, hidden_dim]

        freqs_cis_i = freqs_cis[current_pos, :].unsqueeze(0)
        positions = torch.tensor([current_pos])
        # mask = torch.randn(1, 1)

        reference_output = reference_model(pt_attention_input, freqs_cis_i, positions, mask=None)

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info(f"[pos={current_pos}] Mistral_Attention Passed!")
        else:
            logger.warning(f"[pos={current_pos}] Mistral_Attention Failed!")
            all_tests_pass = False

        if False:  # FIXME: Issue #10648
            # Check kv cache
            # PyTorch output --------------------------------------------------------------------
            pytorch_layer_present = [
                reference_model.cache_k.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
                reference_model.cache_v.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
            ]
            # TT hardware execution -------------------------------------------------------------
            tt_layer_present = []
            for layer_past in tt_model.layer_past_list:
                tt_layer_present.append([ttnn.to_torch(cache) for cache in layer_past])

            tt_layer_present = tt_layer_present[0]

            for i, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
                cache_length_to_check = min(model_args.sliding_window, generation_start_pos + generation_length + 1)
                cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
                cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
                does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
                if i == 0:
                    logger.info(f"K cache output: {output_pcc}")
                else:
                    logger.info(f"V cache output: {output_pcc}")

                if does_pass:
                    logger.info(f"KV Cache Passed!")
                else:
                    logger.warning(f"KV Cache Failed! PCC value is lower than {pcc}")
                    all_tests_pass = False

    if all_tests_pass:
        logger.info("Mistral Attention output Passed!")
    else:
        logger.warning("Mistral Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
