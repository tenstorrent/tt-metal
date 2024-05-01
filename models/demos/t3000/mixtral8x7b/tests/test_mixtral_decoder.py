# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger
import ttnn
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import (
    prepare_inputs_ttnn,
)
from models.demos.t3000.mixtral8x7b.tt.mixtral_decoder import TtTransformerBlock
from models.demos.t3000.mixtral8x7b.reference.model import TransformerBlock, precompute_freqs_cis
from models.utility_functions import comp_pcc, comp_allclose, get_devices_for_t3000

# Set Mixtral flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["MIXTRAL_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"

from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs


def test_mixtral_decoder_inference(all_devices, use_program_cache, reset_seeds):
    """
    b: batch
    s: sequence length
    h: hidden size
    """
    pcc = 0.99
    dtype = ttnn.bfloat8_b

    devices = all_devices
    num_devices = len(devices)
    assert num_devices == 8, "This test requires a T3000 (8 devices)"
    devices = get_devices_for_t3000(devices, num_devices)  # [ttnn.open_device(device_id=i) for i in range(8)]

    model_args = TtModelArgs(devices[0])
    state_dict = torch.load(model_args.state_dict_path)
    partial_state_dict = {k[9:]: v for k, v in state_dict.items() if (k.startswith("layers.0."))}
    reference_model = TransformerBlock(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    # Initialize TT model
    tt_model = TtTransformerBlock(
        devices=devices,
        state_dict=state_dict,
        args=model_args,
        layer_num=0,
        dtype=dtype,
    )

    generation_start_pos = 0
    generation_length = 1
    all_tests_pass = True

    seqlen = 1
    batch = 32

    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")

        # input = torch.randn(1, 32, 4096)
        pt_decode_input_bsh = (torch.rand(batch, seqlen, model_args.dim) * 2) - 1
        start_pos = generation_start_pos + i
        current_pos = start_pos % model_args.sliding_window

        decode_input_b1sh, rot_mat = prepare_inputs_ttnn(
            pt_decode_input_bsh.clone(),
            tt_model.hidden_size,
            tt_model.head_dim,
            tt_model.max_seq_len,
            tt_model.devices,
        )
        # Run TT model
        tt_out_b1sh = tt_model(decode_input_b1sh, start_pos, current_pos, rot_mat)
        tt_output_torch_b1h = ttnn.to_torch(tt_out_b1sh[0]).squeeze(1).view(batch, 1, -1)

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

    if all_tests_pass:
        logger.info(f"All {generation_length} Mistral decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Mistral decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
