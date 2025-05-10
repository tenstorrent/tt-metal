# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
from loguru import logger

import ttnn
from models.demos.wormhole.mistral7b.reference.model import TransformerBlock
from models.demos.wormhole.mistral7b.tt.mistral_common import (
    freqs_to_rotation_matrix,
    precompute_freqs,
    prepare_inputs_ttnn,
)
from models.demos.wormhole.mistral7b.tt.mistral_decoder import TtTransformerBlock
from models.demos.wormhole.mistral7b.tt.model_config import TtModelArgs
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
def test_mistral_decoder_inference(device, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(device)
    state_dict = torch.load(model_args.consolidated_weights_path)

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[9:]: v for k, v in state_dict.items() if (k.startswith("layers.0."))}
    reference_model = TransformerBlock(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    generation_start_pos = 0
    generation_length = 2
    all_tests_pass = True

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

    # Initialize TT model
    tt_model = TtTransformerBlock(
        args=model_args,
        device=device,
        dtype=dtype,
        state_dict=state_dict,
        layer_num=0,
        weight_cache_path=model_args.weight_cache_path(dtype),
        rot_mat=rot_emb_matrix_list,
        start_pos=generation_start_pos,
    )

    seqlen = 1
    batch = model_args.max_batch_size

    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    freqs_cis = torch.complex(cos, sin)

    # TODO Update start_pos (check llama test for reference)
    for i in range(generation_length):
        print(f"[Decoder] Generating token {i}")

        # input = torch.randn(1, 32, 4096)
        pt_decode_input = (torch.rand(batch, seqlen, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input.clone()
        current_pos = generation_start_pos + i

        decode_input, pos = prepare_inputs_ttnn(
            tt_decode_input,
            current_pos,
            model_args.dim,
            model_args.sliding_window,
            tt_model.device,
        )

        # Run TT model
        tt_out = tt_model(decode_input, pos)
        tt_output_torch = (
            ttnn.to_torch(tt_out).permute(2, 1, 0, 3).squeeze(1)[: model_args.max_batch_size, :, :]
        )  # [seq, batch, hidden_dim]

        freqs_cis_i = freqs_cis[current_pos, :].unsqueeze(0)
        positions = torch.tensor([current_pos])

        # Reference model
        ref_output = reference_model(pt_decode_input, freqs_cis_i, positions, mask=None)

        passing, pcc_message = comp_pcc(ref_output, tt_output_torch)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info("Mistral Decoder Block Passed!")
        else:
            logger.warning("Mistral Decoder Block Failed!")
            all_tests_pass = False

    if all_tests_pass:
        logger.info(f"All {generation_length} Mistral decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Mistral decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {0.99} for some of the outputs. Check Warnings!"
