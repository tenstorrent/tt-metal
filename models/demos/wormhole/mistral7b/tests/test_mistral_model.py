# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.wormhole.mistral7b.tt.mistral_common import (
    precompute_freqs,
    freqs_to_rotation_matrix,
    prepare_inputs_ttnn,
    sample,
)
from models.demos.wormhole.mistral7b.tt.mistral_model import TtTransformer
from models.demos.wormhole.mistral7b.tt.model_config import TtModelArgs
from models.demos.wormhole.mistral7b.reference.model import Transformer
from models.demos.wormhole.mistral7b.reference.tokenizer import Tokenizer
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "iterations",
    (17,),
)
def test_mistral_model_inference(device, iterations, use_program_cache, reset_seeds):
    run_ref_pt = True  # Flag to run reference PyTorch model and compare PCC
    cache_pcc = False  # Flag to measure KV cache PCC for all layers

    dtype = ttnn.bfloat8_b
    pcc = 0.97

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

    prompts = ["This is a test"] * model_args.max_batch_size

    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    if run_ref_pt:
        reference_model = Transformer(args=model_args)
        reference_model.load_state_dict(state_dict)

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    generation_start_pos = 0
    generation_length = iterations

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

    # Load TTNN model
    tt_model = TtTransformer(
        args=model_args,
        device=device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layers=list(range(model_args.n_layers)),
        rot_mat=rot_emb_matrix_list,
        start_pos=generation_start_pos,
    )

    if run_ref_pt:
        all_tests_pass = True

    seqlen = 1  # Generating one token per user at a time
    batch = model_args.max_batch_size

    if run_ref_pt:
        cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
        freqs_cis = torch.complex(cos, sin)

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)

    tt_decode_input = pt_decode_input

    # Keep track of generated outputs to print out later
    all_outputs = []
    if run_ref_pt:
        all_outputs_ref = []

    for i in range(generation_length):
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
        # Convert ttnn tensor to torch tensor
        tt_output_torch = (
            ttnn.to_torch(tt_out).permute(2, 1, 0, 3).squeeze(1)[: model_args.max_batch_size, :, :]
        )  # [seq, batch, hidden_dim]

        if run_ref_pt:  # Run reference model
            freqs_cis_i = freqs_cis[current_pos, :].unsqueeze(0)
            positions = torch.tensor([current_pos])
            # mask = ttnn.to_torch(attn_mask[0])
            ref_output = reference_model(pt_decode_input, freqs_cis_i, positions)

        # While in "prefill" mode, use the prompt tokens as the output
        if i in range(len(encoded_prompts[0])):
            all_outputs.append(encoded_prompts[0][i])  # Update list of TT outputs
            if run_ref_pt:
                all_outputs_ref.append(encoded_prompts[0][i])  # Update list of ref outputs

            tt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
            if run_ref_pt:
                pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
        else:
            # Greedy decode (temperature = 0) the generated token and save it to print out later
            tt_out_tok = sample(tt_output_torch, temperature=0, top_p=0.8)
            tt_decode_input = embd(tt_out_tok)
            all_outputs.append(tt_out_tok.squeeze(1).tolist()[0])  # Update generated token to list of TT outputs
            if run_ref_pt:
                pt_out_tok = sample(ref_output, temperature=0, top_p=0.8)
                pt_decode_input = embd(pt_out_tok)
                all_outputs_ref.append(
                    pt_out_tok.squeeze(1).tolist()[0]
                )  # Update generated token to list of ref outputs

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
                        cache_length_to_check = min(
                            model_args.sliding_window, generation_start_pos + generation_length + 1
                        )
                        cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
                        cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
                        does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
                        if i == 0:
                            logger.info(f"K cache output: {output_pcc}")
                        else:
                            logger.info(f"V cache output: {output_pcc}")

                        if does_pass:
                            logger.info(f"V Cache Passed!")
                        else:
                            logger.warning(f"V Cache Failed! PCC value is lower than {pcc}")
                        # if not does_pass:
                        # all_tests_pass = False

        logger.trace("[ttnn generation User 0] ", "".join(tokenizer.decode(all_outputs)))
        if run_ref_pt:
            logger.trace("[Ref generation User 0] ", "".join(tokenizer.decode(all_outputs_ref)))

    if run_ref_pt:
        if all_tests_pass:
            logger.info(f"All {generation_length} Mistral decode iterations Passed!")
        else:
            logger.warning("One or more iterations of Mistral decode had bad PCC")
            assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
