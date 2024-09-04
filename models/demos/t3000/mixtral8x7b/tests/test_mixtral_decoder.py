# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger

import ttnn
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import prepare_inputs_ttnn, get_single_rot_mat
from models.demos.t3000.mixtral8x7b.tt.mixtral_decoder import TtTransformerBlock
from models.demos.t3000.mixtral8x7b.reference.model import TransformerBlock, precompute_freqs_cis
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.utility_functions import comp_pcc, comp_allclose
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor


@pytest.mark.parametrize(
    "batch",
    (
        32,
        16,
    ),
)
def test_mixtral_decoder_inference(t3k_mesh_device, use_program_cache, reset_seeds, batch):
    """
    b: batch
    s: sequence length
    h: hidden size
    """
    for device in t3k_mesh_device.get_device_ids():
        t3k_mesh_device.get_device(device).enable_async(True)

    pcc = 0.99
    dtype = ttnn.bfloat8_b

    if batch == 32:
        generation_start_pos = 15000
        max_seq_len = 16384
    elif batch in [4, 8, 16]:
        generation_start_pos = 30000
        max_seq_len = 32768
    else:
        raise ValueError(f"Batch size {batch} not supported")

    model_args = TtModelArgs(t3k_mesh_device.get_device(0), max_seq_len=max_seq_len, max_batch_size=batch)
    state_dict = model_args.load_state_dict()
    partial_state_dict = {k[9:]: v for k, v in state_dict.items() if (k.startswith("layers.0."))}
    reference_model = TransformerBlock(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    # Initialize TT model
    tt_model = TtTransformerBlock(
        mesh_device=t3k_mesh_device,
        state_dict=state_dict,
        args=model_args,
        layer_num=0,
        dtype=dtype,
    )

    current_rot_mat, rot_matrix = get_single_rot_mat(
        model_args.head_dim,
        tt_model.mesh_device,
    )

    generation_length = 10
    all_tests_pass = True

    seqlen = 1

    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")

        pt_decode_input_bsh = (torch.rand(batch, seqlen, model_args.dim) * 2) - 1
        start_pos = generation_start_pos + i
        current_pos = start_pos

        decode_input_b1sh = prepare_inputs_ttnn(
            pt_decode_input_bsh,
            model_args.dim,
            start_pos,
            model_args,
            tt_model.mesh_device,
        )

        # Run TT model
        tt_out_b1sh = tt_model(decode_input_b1sh, start_pos, current_pos, None, current_rot_mat)

        tt_output_torch_b1h = (
            ttnn.to_torch(tt_out_b1sh, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=0))[0]
            .squeeze(1)
            .view(32, 1, -1)
        )[:batch, ...]

        # Reference model
        positions = torch.LongTensor([start_pos])
        freqs_cis_i = precompute_freqs_cis(model_args.head_dim, 128_000)[positions]
        ref_output_bsh = reference_model(pt_decode_input_bsh, freqs_cis_i, positions, mask=None)

        passing, pcc_message = comp_pcc(ref_output_bsh, tt_output_torch_b1h, pcc)

        logger.info(comp_allclose(ref_output_bsh, tt_output_torch_b1h))
        logger.info(pcc_message)

        if passing:
            logger.info("Mistral Decoder Block Passed!")
        else:
            logger.warning("Mistral Decoder Block Failed!")
            all_tests_pass = False

        current_rot_mat = ttnn.linear(rot_matrix, current_rot_mat)

    if all_tests_pass:
        logger.info(f"All {generation_length} Mistral decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Mistral decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
