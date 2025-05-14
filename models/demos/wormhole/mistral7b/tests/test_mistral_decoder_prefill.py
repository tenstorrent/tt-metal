# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.wormhole.mistral7b.reference.model import TransformerBlock, precompute_freqs_cis
from models.demos.wormhole.mistral7b.tt.mistral_common import (
    get_prefill_rot_mat,
    get_rot_transformation_mat,
    prepare_inputs_ttnn_prefill,
)
from models.demos.wormhole.mistral7b.tt.mistral_decoder import TtTransformerBlock
from models.demos.wormhole.mistral7b.tt.model_config import TtModelArgs
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "seq_len",
    (
        4096,
        128,
    ),
)
def test_mistral_decoder_inference(device, seq_len, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(device)
    state_dict = torch.load(model_args.consolidated_weights_path)

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[9:]: v for k, v in state_dict.items() if (k.startswith("layers.0."))}

    model_args.max_batch_size = 16
    batch = 1
    reference_model = TransformerBlock(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    generation_start_pos = 0
    generation_length = 2
    all_tests_pass = True

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

    # Initialize TT model
    tt_model = TtTransformerBlock(
        args=model_args,
        device=device,
        dtype=dtype,
        state_dict=state_dict,
        layer_num=0,
        weight_cache_path=model_args.weight_cache_path(dtype),
        rot_mat=None,
        start_pos=generation_start_pos,
    )

    # TODO Update start_pos (check llama test for reference)
    for i in range(generation_length):
        print(f"[Decoder] Generating token {i}")
        pt_decode_input = (torch.rand(batch, seq_len, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input.clone()
        decode_input, attn_mask, attn_mask_torch = prepare_inputs_ttnn_prefill(
            tt_decode_input,
            tt_model.device,
        )
        positions = torch.LongTensor(range(seq_len))
        freqs_cis_i = precompute_freqs_cis(model_args.head_dim, seq_len)[positions]

        # Reference model
        ref_output = reference_model(pt_decode_input, freqs_cis_i, positions, attn_mask_torch)
        # Run TT model
        tt_out = tt_model(decode_input, 0, attn_mask, rot_mats, transformation_mats, user_id=0, mode="prefill")
        tt_output_torch = ttnn.to_torch(tt_out).view(batch, seq_len, -1)  # [seq, batch, hidden_dim]
        passing, pcc_message = comp_pcc(ref_output, tt_output_torch)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info("Mistral Decoder Block Passed!")
        else:
            logger.warning("Mistral Decoder Block Failed!")
            all_tests_pass = False

    if all_tests_pass:
        logger.info(f"All Mistral decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Mistral decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {0.99} for some of the outputs. Check Warnings!"
