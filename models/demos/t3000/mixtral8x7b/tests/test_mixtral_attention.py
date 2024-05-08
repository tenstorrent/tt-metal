# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger

import ttnn
from models.demos.t3000.mixtral8x7b.tt.mixtral_attention import TtMixtralAttention
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import prepare_inputs_ttnn, prepare_rotation_mat_ttnn
from models.demos.t3000.mixtral8x7b.reference.model import Attention, precompute_freqs_cis
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import get_devices_for_t3000

# Set Mixtral flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["MIXTRAL_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"

from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs


def test_mixtral_attention_inference(all_devices, use_program_cache, reset_seeds):
    pcc = 0.99
    dtype = ttnn.bfloat8_b
    devices = all_devices
    num_devices = len(devices)
    assert num_devices == 8, f"This test requires a T3000 (8 devices), found {num_devices} devices."
    devices = get_devices_for_t3000(devices, num_devices)  # [ttnn.open_device(device_id=i) for i in range(8)]

    model_args = TtModelArgs(devices[0])
    state_dict = torch.load(model_args.consolidated_weights_path(0), map_location="cpu")

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[19:]: v for k, v in state_dict.items() if (k.startswith("layers.0.attention."))}

    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    batch = 32
    seq_len = 1  # length to generate

    tt_model = TtMixtralAttention(devices, state_dict, args=model_args, layer_num=0, dtype=dtype)

    rot_mat = prepare_rotation_mat_ttnn(
        model_args.head_dim,
        model_args.max_seq_len,
        tt_model.devices,
    )

    generation_start_pos = 0
    generation_length = 1
    all_tests_pass = True

    for i in range(generation_length):
        pt_attention_input = (torch.rand(batch, seq_len, model_args.dim) * 2) - 1
        tt_attention_input = pt_attention_input
        start_pos = generation_start_pos + i
        attention_input = prepare_inputs_ttnn(
            tt_attention_input,
            tt_model.hidden_size,
            tt_model.devices,
        )
        current_pos = start_pos % model_args.sliding_window
        tt_out = tt_model(
            attention_input,
            start_pos,
            current_pos,
            rot_mat,
        )
        assert isinstance(tt_out, list)  # tt_out should be replicated on N devices
        tt_out = tt_out[0]
        tt_output_torch = ttnn.to_torch(tt_out).squeeze(2).view(batch, 1, -1)  # [ batch, seq, hidden_dim]
        positions = torch.LongTensor([start_pos])
        freqs_cis_i = precompute_freqs_cis(model_args.head_dim, 128_000)[positions]
        reference_output = reference_model(pt_attention_input, freqs_cis_i, positions, mask=None)
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info(f"[start_pos={start_pos}] Mistral_Attention Passed!")
        else:
            logger.warning(f"[start_pos={start_pos}] Mistral_Attention Failed!")
            all_tests_pass = False

        # Check kv cache
        # PyTorch output --------------------------------------------------------------------
        pytorch_layer_present = [
            reference_model.cache_k.permute(2, 0, 1, 3),  # [n_kv_heads, batch, seq, head_dim]
            reference_model.cache_v.permute(2, 0, 1, 3),  # [n_kv_heads, batch, seq, head_dim]
        ]
        # TT hardware execution -------------------------------------------------------------
        tt_layer_present = []
        for layer_past in tt_model.layer_past_list:
            tt_layer_present.append([ttnn.to_torch(cache) for cache in layer_past])
        # concat the pasts by heads
        if len(devices) > 1:
            tt_layer_present = [
                torch.cat([tt_cache for tt_cache in tt_cache_head], dim=0) for tt_cache_head in zip(*tt_layer_present)
            ]
        else:
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
                if i == 0:
                    logger.info(f"K Cache Passed!")
                else:
                    logger.info(f"V Cache Passed!")
            else:
                if i == 0:
                    logger.warning(f"K Cache Failed! PCC value is lower than {pcc}")
                else:
                    logger.warning(f"V Cache Failed! PCC value is lower than {pcc}")
                all_tests_pass = False

    if all_tests_pass:
        logger.info("Mistral Attention output Passed!")
    else:
        logger.warning("Mistral Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
