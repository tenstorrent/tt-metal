# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.wormhole.llama31_8b.tt.llama_attention import TtLlamaAttention
from models.demos.wormhole.llama31_8b.tt.model_config import TtModelArgs
from models.demos.wormhole.llama31_8b.tt.llama_common import (
    get_prefill_rot_mat,
    prepare_inputs_ttnn_prefill,
    get_rot_transformation_mat,
)
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Attention, precompute_freqs_cis
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "seq_len",
    (4096,),
)
def test_llama_attention_inference(seq_len, device, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat8_b
    pcc = 0.99

    model_args = TtModelArgs(device)
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[19:]: v for k, v in state_dict.items() if (k.startswith("layers.0.attention."))}
    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    batch = 1

    # pre-compute the rotational embedding matrix and send to device
    rot_mats = get_prefill_rot_mat(model_args.head_dim, model_args.max_seq_len, device, seq_len=seq_len)
    transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)
    transformation_mats = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    generation_start_pos = 0
    generation_length = 3
    all_tests_pass = True

    tt_model = TtLlamaAttention(
        [device],
        state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        configuration=model_args,
    )

    pt_attention_input = (torch.rand(batch, seq_len, model_args.dim) * 2) - 1
    tt_attention_input = pt_attention_input.clone()
    attention_input = prepare_inputs_ttnn_prefill(
        tt_attention_input,
        device,
    )

    tt_out = tt_model([attention_input], 0, None, rot_mats, transformation_mats, user_id=0, mode="prefill")
    # multi-device attention module returns replicated output
    assert isinstance(tt_out, list)
    tt_out = tt_out[0]
    tt_output_torch = ttnn.to_torch(tt_out).view(batch, seq_len, -1)  # [ batch, seq, hidden_dim]

    positions = torch.LongTensor(range(seq_len))
    freqs_cis_i = precompute_freqs_cis(
        model_args.head_dim, model_args.max_seq_len * 2, model_args.rope_theta, model_args.use_scaled_rope
    )[positions]
    attn_mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min)
    attn_mask_torch = torch.triu(attn_mask, diagonal=1)
    reference_output = reference_model(pt_attention_input, positions[0], freqs_cis_i, mask=attn_mask_torch)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info(f"Llama_Attention Passed!")
    else:
        logger.warning(f"Llama_Attention Failed!")
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
        logger.info("Llama Attention output Passed!")
    else:
        logger.warning("Llama Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
