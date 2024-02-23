# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import tt_lib
import pytest
from loguru import logger
import json
from pathlib import Path
from models.experimental.mistral.tt.mistral_common import precompute_freqs, generate_cos_sin_cache, prepare_inputs
from models.experimental.mistral.tt.mistral_model import TtTransformer
from models.experimental.mistral.tt.model_config import TtModelArgs, get_model_config
from models.experimental.mistral.reference.model import Transformer
from models.experimental.mistral.reference.tokenizer import Tokenizer
from models.utility_functions import tt2torch_tensor
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "n_layers",
    ((3,)),
)
@pytest.mark.parametrize(
    "model_config",
    ("BFLOAT16-DRAM", "BFLOAT8-DRAM"),
)
@pytest.mark.parametrize(
    "iterations",
    ((3),),
)
@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_mistral_model_inference(pcc, model_config, model_location_generator, device, iterations, n_layers):
    prompts = [
        "This is a sample text for single layer execution ",
    ]

    mistral_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")
    tokenizer = Tokenizer(str(Path(mistral_path) / "tokenizer.model"))
    base_address = f""
    with open(mistral_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))
    model_args.max_batch_size = 32
    model_args.n_layers = n_layers

    state_dict = torch.load(mistral_path / "consolidated.00.pth")
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
        )
    }

    reference_model = Transformer(args=model_args)
    reference_model.load_state_dict(state_dict)

    # TODO Scale the model (mixtral) to multiple devices when T3000 is available
    devices = [
        device,
    ]

    model_config = get_model_config(model_config)
    tt_cos_cached, tt_sin_cached = generate_cos_sin_cache(
        devices, model_args.head_dim, "", model_args.max_seq_len * 2, 10000, model_config
    )
    tt_model = TtTransformer(
        args=model_args,
        devices=devices,
        state_dict=state_dict,
        base_address=base_address,
        model_config=model_config,
        tt_cos_cached=tt_cos_cached,
        tt_sin_cached=tt_sin_cached,
    )

    generation_start_pos = 0
    generation_length = iterations
    all_tests_pass = True

    seqlen = 1
    batch = 32

    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    freqs_cis = torch.complex(cos, sin)

    # TODO Update start_pos (check llama test for reference)
    for i in range(generation_length):
        print(f"[Model] Generating token {i}")

        # input = torch.randn(1, 32, 4096)
        pt_decode_input = (torch.rand(batch, seqlen, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input.clone()
        start_pos = generation_start_pos + i

        decode_input, start_pos, attn_mask, current_pos = prepare_inputs(
            tt_decode_input,
            start_pos,
            model_args.dim,
            model_args.n_heads // len(devices),
            model_args.sliding_window,
            tt_model.devices,
            tt_model.num_devices,
        )
        # Run TT model
        tt_out = tt_model(decode_input, start_pos, current_pos, attn_mask)
        # tt_output = tt_model(tt_input, bcast_freq_xq, bcast_freq_xk, tt_position, mask, seqlen)

        tt_output_torch = tt2torch_tensor(tt_out).permute(2, 1, 0, 3).squeeze(1)  # [seq, batch, hidden_dim]

        freqs_cis_i = freqs_cis[start_pos, :].unsqueeze(0)
        positions = torch.tensor([start_pos])

        # Reference model
        # mask = tt2torch_tensor(attn_mask[0])
        ref_output = reference_model(pt_decode_input, freqs_cis_i, positions)  # mask)

        passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info("Mistral Model Block Passed!")
        else:
            logger.warning("Mistral Model Block Failed!")
            all_tests_pass = False

    if all_tests_pass:
        logger.info(f"All {generation_length} Mistral decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Mistral decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
