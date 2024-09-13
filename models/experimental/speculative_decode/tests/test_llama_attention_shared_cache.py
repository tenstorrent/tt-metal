# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.wormhole.llama31_8b.tt.llama_attention import TtLlamaAttention
from models.demos.wormhole.llama31_8b.tt.model_config import TtModelArgs
from models.demos.wormhole.llama31_8b.tt.llama_common import precompute_freqs, prepare_inputs_ttnn, get_single_rot_mat
from models.experimental.speculative_decode.torch_reference.llama_modules import Attention
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


# Construct rotary mat for speculation
def get_rotary_mat_for_speculation(model_args, speculation_length, start_pos=0):
    current_rot_mat_torch, rot_matrix_torch = get_single_rot_mat(
        model_args.head_dim,
        "cpu",
        start_pos=start_pos,
    )
    # return current_rot_mat_torch, rot_matrix_torch
    rot_mats = [current_rot_mat_torch]
    for i in range(speculation_length):
        rot_mats.append(torch.matmul(rot_matrix_torch, rot_mats[-1]))
    current_rot_mat = torch.concat(rot_mats, dim=1)
    # # pad dim 1 to 32 to work around a hang in ttnn
    # current_rot_mat = torch.cat((current_rot_mat, torch.zeros(current_rot_mat.shape[0], 32 - current_rot_mat.shape[1], model_args.head_dim, model_args.head_dim)), dim=1)
    return current_rot_mat, rot_matrix_torch


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


@skip_for_grayskull("Requires wormhole_b0 to run")
def test_llama_attention_inference(device, use_program_cache, reset_seeds, debug=True):
    dtype = ttnn.bfloat8_b
    pcc = 0.99

    model_args = TtModelArgs(device, max_batch_size=4)
    model_args.share_cache = True
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[19:]: v for k, v in state_dict.items() if (k.startswith("layers.0.attention."))}

    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    batch = model_args.max_batch_size
    seq_len = 1

    generation_start_pos = 0
    generation_length = 2
    all_tests_pass = True

    # pre-compute the rotational embedding matrix and send to device
    current_rot_mat_torch, rot_matrix_torch = get_rotary_mat_for_speculation(
        model_args,
        batch - 1,
        start_pos=0,
    )
    current_rot_mat = ttnn.from_torch(
        current_rot_mat_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=model_args.model_config["ROT_MAT_MEMCONFIG"],
    )

    tt_model = TtLlamaAttention(
        [device],
        state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        configuration=model_args,
        rot_mat=None,
        start_pos=generation_start_pos,
    )

    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    freqs_cis = torch.complex(cos, sin)
    for i in range(generation_length):
        pt_attention_input = fa_rand(batch, seq_len, model_args.dim)

        tt_attention_input = pt_attention_input.clone()
        current_pos = generation_start_pos + i

        attention_input, pos = prepare_inputs_ttnn(
            tt_attention_input,
            current_pos,
            model_args.dim,
            model_args.sliding_window,
            device,
        )

        pos_list = list(range(pos, pos + batch))
        tt_out = tt_model([attention_input], pos_list, rot_mats=current_rot_mat)
        # multi-device attention module returns replicated output
        assert isinstance(tt_out, list)
        tt_out = tt_out[0]
        tt_output_torch = (
            ttnn.to_torch(tt_out).view(1, -1, 4096).permute(1, 0, 2)[: model_args.max_batch_size, :, :]
        )  # [ batch, seq, hidden_dim]

        # Reference Input
        freqs_cis_i = freqs_cis[current_pos : current_pos + batch, :]
        pt_attention_input = torch.permute(
            pt_attention_input, (1, 0, 2)
        )  # put batch into seq dimension for share cache
        mask = torch.full((batch, batch), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        mask = torch.hstack([torch.zeros((batch, current_pos)), mask])

        reference_output = reference_model(pt_attention_input, current_pos, freqs_cis_i, mask=mask)
        reference_output = torch.permute(reference_output, (1, 0, 2))  # put seq back to batch dimension

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info(f"[pos={current_pos}] Llama_Attention Passed!")
        else:
            logger.warning(f"[pos={current_pos}] Llama_Attention Failed!")
            all_tests_pass = False

        if debug:
            logger.info(f"Debug info: pcc per batch")
            for batch_idx in range(batch):
                batch_reference_output = reference_output[batch_idx, :, :]
                batch_tt_output_torch = tt_output_torch[batch_idx, :, :]
                passing, pcc_message = comp_pcc(batch_reference_output, batch_tt_output_torch, pcc)
                logger.info(f"Batch {batch_idx}: {pcc_message}")

        # Update rotation matrix for next iteration
        current_rot_mat_torch = torch.matmul(rot_matrix_torch, current_rot_mat_torch)
        current_rot_mat.deallocate()
        current_rot_mat = ttnn.from_torch(
            current_rot_mat_torch,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=model_args.model_config["ROT_MAT_MEMCONFIG"],
        )

        if True:  # FIXME: Issue #10648
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
                cache_length_to_check = min(model_args.sliding_window, pos + batch)
                cache_pt = cache_pt[:, :, pos:cache_length_to_check, :]
                cache_tt = cache_tt[:, :, pos:cache_length_to_check, :]

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

                if debug:
                    logger.info(f"Debug info:")
                    for batch_idx in range(batch):
                        batch_cache_pt = cache_pt[:, :, batch_idx, :]
                        batch_cache_tt = cache_tt[:, :, batch_idx, :]
                        passing, pcc_message = comp_pcc(batch_cache_pt, batch_cache_tt, pcc)
                        logger.info(f"Batch (seqlen) {batch_idx}: {pcc_message}")

    if all_tests_pass:
        logger.info("Llama Attention output Passed!")
    else:
        logger.warning("Llama Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
