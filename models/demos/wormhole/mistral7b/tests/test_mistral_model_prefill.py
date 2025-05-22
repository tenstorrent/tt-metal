# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.wormhole.mistral7b.reference.model import Transformer, precompute_freqs_cis
from models.demos.wormhole.mistral7b.reference.tokenizer import Tokenizer
from models.demos.wormhole.mistral7b.tt.mistral_common import (
    get_prefill_rot_mat,
    get_rot_transformation_mat,
    prepare_inputs_ttnn_prefill,
)
from models.demos.wormhole.mistral7b.tt.mistral_model import TtTransformer
from models.demos.wormhole.mistral7b.tt.model_config import TtModelArgs
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull, skip_for_wormhole_b0


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


@skip_for_grayskull("Requires wormhole_b0 to run")
@skip_for_wormhole_b0("#12874: Flaky, seems to hang after verification, need to fix, otherwise seems fine")
@pytest.mark.timeout(900)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "seq_len",
    (4096,),
)
def test_mistral_model_inference(device, seq_len, use_program_cache, reset_seeds, is_ci_env):
    # Set additional Mistral flag for CI
    if is_ci_env:
        os.environ["MISTRAL_REF_OUTPUT_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/prefill/"

    run_ref_pt = True  # Flag to run reference PyTorch model and compare PCC
    cache_pcc = False  # Flag to measure KV cache PCC for all layers

    dtype = ttnn.bfloat8_b
    pcc = 0.93

    model_args = TtModelArgs(device)
    model_args.n_layers = 32  # Full model

    tokenizer = Tokenizer(model_args.tokenizer_path)

    logger.info("Loading weights...")
    state_dict = torch.load(model_args.consolidated_weights_path)
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
        )
    }
    logger.info("Finished loading weights...")

    prompt_file = os.environ["MISTRAL_REF_OUTPUT_PATH"] + "/tale-of-two-cities.txt"
    assert os.path.exists(
        prompt_file
    ), f"Expected prompt file not found: {prompt_file}. Please set the flag 'MISTRAL_REF_OUTPUT_PATH' correctly."

    with open(prompt_file, "r") as f:
        prompts = f.read()

    encoded_prompts = tokenizer.encode(prompts)[:seq_len]

    if run_ref_pt:
        reference_model = Transformer(args=model_args)
        reference_model.load_state_dict(state_dict)

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

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
    # Load TTNN model
    tt_model = TtTransformer(
        args=model_args,
        device=device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layers=list(range(model_args.n_layers)),
        rot_mat=None,
        start_pos=0,
    )

    if run_ref_pt:
        all_tests_pass = True

    batch = 1

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor).view(batch, seq_len, -1)

    tt_decode_input = pt_decode_input

    decode_input, attn_mask, attn_mask_torch = prepare_inputs_ttnn_prefill(
        tt_decode_input,
        tt_model.device,
    )
    for i in range(3):
        # Run TT model
        tt_out = tt_model(decode_input, 0, attn_mask, rot_mats, transformation_mats, user_id=i, mode="prefill")
        # Convert ttnn tensor to torch tensor
        tt_output_torch = ttnn.to_torch(tt_out).view(batch, seq_len, -1)  # [seq, batch, hidden_dim]

        if run_ref_pt:  # Run reference model
            positions = torch.LongTensor(range(seq_len))
            freqs_cis_i = precompute_freqs_cis(model_args.head_dim, seq_len)[positions]

            # mask = ttnn.to_torch(attn_mask[0])
            ref_output = reference_model(pt_decode_input, freqs_cis_i, positions, mode="prefill")

        # TODO Measure only PCC at the end, instead of at every iteration
        # Measure PCC if also running reference model
        if run_ref_pt:
            passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

            logger.info(comp_allclose(ref_output, tt_output_torch))
            logger.info(f"Model output: {pcc_message}")

            if passing:
                logger.info("Mistral Model Passed!")
            else:
                logger.warning("Mistral Model Failed!")
            if not passing:
                all_tests_pass = False

            # Compare KV caches
            if cache_pcc:
                for i in range(model_args.n_layers):
                    pytorch_layer_present = [
                        reference_model.layers[i]
                        .attention.cache_k.clone()
                        .permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
                        reference_model.layers[i]
                        .attention.cache_v.clone()
                        .permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
                    ]

                    tt_layer_present = []
                    for layer_past in tt_model.layers[i].attention.layer_past_list[0]:
                        tt_layer_present.append(ttnn.to_torch(layer_past))

                    for i, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
                        cache_length_to_check = model_args.sliding_window
                        cache_pt = cache_pt[:, :, 0:cache_length_to_check, :]
                        cache_tt = cache_tt[:, :, 0:cache_length_to_check, :]
                        does_pass, output_pcc = comp_pcc(cache_pt, cache_tt)
                        if i == 0:
                            logger.info(f"K cache output: {output_pcc}")
                        else:
                            logger.info(f"V cache output: {output_pcc}")

                        if does_pass:
                            logger.info(f"V Cache Passed!")
                        else:
                            logger.warning(f"V Cache Failed! PCC value is lower than {0.99}")
                        # if not does_pass:
                        # all_tests_pass = False

    if run_ref_pt:
        if all_tests_pass:
            logger.info(f"All  Mistral decode iterations Passed!")
        else:
            logger.warning("One or more iterations of Mistral decode had bad PCC")
            assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
