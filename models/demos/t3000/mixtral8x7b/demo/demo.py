# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import json
import pytest
from loguru import logger
from time import time

import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import (
    load_inputs,
    preprocess_inputs,
    prepare_inputs_ttnn,
    get_single_rot_mat,
    sample,
    cache_attention,
)
from models.demos.t3000.mixtral8x7b.tt.mixtral_model import TtTransformer
from models.demos.t3000.mixtral8x7b.tt.mixtral_embedding import TtMixtralEmbedding
from models.demos.t3000.mixtral8x7b.reference.tokenizer import Tokenizer


from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


@torch.no_grad()
def run_mixtral_demo(user_input, batch_size, device_mesh, instruct_mode, is_ci_env):
    if batch_size == 32:
        max_seq_len = 16384
    elif batch_size in [4, 8, 16]:
        max_seq_len = 32768
    else:
        raise ValueError(f"Batch size {batch_size} not supported")

    dtype = ttnn.bfloat8_b

    embed_on_host = True  # embedding and argmax on host. TODO Seeing bad output when on device
    seqlen = 1  # Generating one token per user at a time

    logger.info(f"Reading inputs...")
    if len(user_input) == 1:
        input_prompts = user_input * batch_size  # Always process 32 users
    else:
        input_prompts = load_inputs(user_input, batch_size)

    # Load model args, weights, and tokenizer
    model_args = TtModelArgs(
        device_mesh.get_device(0), instruct=instruct_mode, max_seq_len=max_seq_len, max_batch_size=batch_size
    )
    tokenizer = Tokenizer(model_args.tokenizer_path)

    model_args.n_layers = 32  # Full model

    logger.info("Loading weights...")
    state_dict = torch.load(model_args.state_dict_path)
    # If not using the full model, remove the layers that are not used
    keys_dict = list(state_dict.keys())[:]
    remv = [f"layers.{i}" for i in range(model_args.n_layers, 32)]
    for k in keys_dict:
        if any([r in k for r in remv]):
            state_dict.pop(k)

    # Embedding on host
    if embed_on_host:
        embd = Emb()
        embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    logger.info("Loading weights finished!")

    # Preprocess initial prompt inputs
    input_tokens_tt, max_prompt_len, input_mask, input_tokens_pt, input_mask_pt = preprocess_inputs(
        input_prompts, tokenizer, model_args, dtype, instruct_mode, device_mesh
    )

    if instruct_mode:
        tokenizer._model.pad_id = tokenizer._model.eos_id

    # Load TTNN mixtral model
    logger.info("Loading weights to device...")
    tt_model = TtTransformer(
        device_mesh=device_mesh,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
        rotary_on_host=True,
    )

    if not embed_on_host:
        tt_embds = TtMixtralEmbedding(
            device_mesh=device_mesh,
            args=model_args,
            weight_cache_path=model_args.weight_cache_path(dtype),
            state_dict=state_dict,
            dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
        )

    logger.info("Finished loading weights to device.")
    # Prepare the first token embedding for each user
    if embed_on_host:
        pt_decode_input = embd(input_tokens_pt[:, 0]).view(batch_size, seqlen, -1)
    else:  # Embedding on device
        # Each device does its own embedding
        decode_input_11BH = tt_embds(input_tokens_tt[0])
        # Reshape and change row major to tile layout
        decode_input_11BH = ttnn.reshape(decode_input_11BH, ttnn.Shape([1, 1, batch_size, model_args.dim]))

        decode_input_11BH = ttnn.to_layout(decode_input_11BH, layout=ttnn.TILE_LAYOUT)
        # decode_input_11BH = [ttnn.tilize(decode_input_11BH[i]) for i in range(len(devices))]
        # decode_input_11BH = [ttnn.tilize_with_val_padding(decode_input_11BH[i], ) for i in range(len(devices))]")

    # Prepare inputs for decode mode (rotary embeddings, attention mask, padding)
    current_rot_mat, rot_matrix = get_single_rot_mat(
        model_args.head_dim,
        tt_model.device_mesh,
    )

    generation_start_pos = 0
    max_generated_tokens = 50  # max_seq_len-1

    cache_attention(
        device_mesh,
        state_dict,
        model_args,
        current_rot_mat,
        rot_matrix,
        dtype,
    )

    logger.info("Starting inference...")

    # Keep track of generated outputs to print out every iteration
    all_outputs = [[] for _ in range(batch_size)]

    # Keep track of users that are done generating and stop printing their outputs
    finished_generation = [False] * batch_size

    # Keep running inference as long as there is a user in the batch still decoding or max tokens per user are decoded
    for iteration in range(max_generated_tokens):
        # Check if all users have finished generating (reached EoS token). If so, stop decoding.
        if all(finished_generation):
            logger.info("All users have finished generating tokens")
            break

        iteration_time_start = time()
        start_pos = generation_start_pos + iteration
        current_pos = start_pos

        if embed_on_host:
            decode_input_11BH = prepare_inputs_ttnn(
                pt_decode_input,
                model_args.dim,
                start_pos,
                model_args,
                tt_model.device_mesh,
            )

        # Run ttnn mixtral model
        tt_out_11BH = tt_model(decode_input_11BH, start_pos, current_pos)

        if embed_on_host:
            # Convert ttnn tensor to torch tensor
            tt_output_torch = (
                ttnn.to_torch(tt_out_11BH, mesh_composer=ConcatMeshToTensor(device_mesh, dim=0))[0]
                .squeeze(1)
                .view(32, seqlen, -1)
                .detach()
                .float()
            )[:batch_size, ...]
            # tt_token_batch = tt_output_torch.squeeze().argmax(axis=-1)
            # Argmax on host to get the new generated tokens
            tt_token_batch = sample(tt_output_torch, temperature=0, top_p=0.8)
            # Update the users that are still in prefill and the ones generating new tokens
            if iteration < max_prompt_len:
                tt_token_batch = torch.where(
                    input_mask_pt[:, iteration], input_tokens_pt[:, iteration], tt_token_batch[:, 0]
                ).unsqueeze(1)
            # Next PT input embedding
            pt_decode_input = embd(tt_token_batch).view(batch_size, seqlen, -1)
        else:  # Embedding/argmax on device
            # TODO Debug (only device 0 is doing argmax, otherwise it throws an error)
            # Alternatively, send the output back to device: ttnn.Tensor.to()
            # ttnn.SetDefaultDevice(device_mesh.get_device(0))

            # TODO Update argmax to ttnn when OP becomes available
            tt_out_B11B = ttnn.argmax(tt_out_11BH, dim=-1)
            tt_out_1B = ttnn.reshape(tt_out_B11B[:1, :, :, :], ttnn.Shape([1, batch_size]))  # [1, 32] Bfloat16
            # Update the users that are still in prefill and the ones generating new tokens
            if iteration < max_prompt_len:
                decode_input_1B = ttnn.where(input_mask[iteration], input_tokens_tt[iteration], tt_out_1B)
            else:
                decode_input_1B = tt_out_1B

            # Next TT input embeddings
            decode_input_1BH = tt_embds(decode_input_1B)
            decode_input_11BH = ttnn.reshape(decode_input_1BH, ttnn.Shape([1, 1, batch_size, model_args.dim]))
            decode_input_11BH = ttnn.to_layout(decode_input_11BH, layout=ttnn.TILE_LAYOUT)

            # Convert ttnn tensor to torch tensor and print decoded output (from a single device)
            # tt_output_torch = ttnn.to_torch(decode_input_1B).transpose(0, 1)
            tt_token_batch = ttnn.to_torch(decode_input_1B).transpose(0, 1)

        # Get the generated tokens for each user for printing in the log
        for user in range(batch_size):
            user_tok = int(tt_token_batch[user].item())
            if user_tok == tokenizer.eos_id:  # Stop saving the ouput after hitting the EOS token
                finished_generation[user] = True
            if finished_generation[user] == False:
                all_outputs[user].append(user_tok)

        iteration_time = time() - iteration_time_start
        tokens_per_second_per_user = 1 / iteration_time
        # Print out generated outputs for each user at the end of every iteration
        if not is_ci_env:  # Avoid printing every iteration in CI
            if len(user_input) == 1:
                logger.info("[User 0] {}".format("".join(tokenizer.decode(all_outputs[0]))))
            else:
                for user in range(batch_size):
                    logger.info("[User {}] {}".format(user, "".join(tokenizer.decode(all_outputs[user]))))

        # Always print iteration perf
        logger.info(
            f"Iteration {iteration}: {1000*iteration_time:.2f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
        )

    # In CI only print the final generated output to avoid spamming the logs
    # FIXME Issue #11850: Token verification is disabled for now
    # if is_ci_env:
    #     if len(user_input) == 1:
    #         logger.info("[User 0] {}".format("".join(tokenizer.decode(all_outputs[0]))))
    #     else:
    #         for user in range(batch_size):
    #             logger.info("[User {}] {}".format(user, "".join(tokenizer.decode(all_outputs[user]))))

    #     # When running in CI, check the output against the expected output to avoid accuracy regressions
    #     expected_output = "models/demos/t3000/mixtral8x7b/demo/expected_outputs.json"
    #     with open(expected_output, "r") as f:
    #         expected_out = json.load(f)
    #     assert (
    #         len(expected_out) >= batch_size * 2
    #     ), f"expected_outputs.json should have 64 outputs: 32 for general weights and 32 for instruct weights!"

    #     for i in range(batch_size):
    #         user_output = "".join(tokenizer.decode(all_outputs[i]))
    #         if instruct_mode:  # The instruct outputs are at the end of the expected outputs file
    #             user_expect = expected_out[i + 32]["output_instruct"]
    #         else:
    #             user_expect = expected_out[i]["output_general"]

    #         assert user_output == user_expect, f"Output for user {i} does not match expected output!"
    #     logger.info("[CI-Only] Output token validation passed!")


@pytest.mark.parametrize(
    "input_prompts, instruct_weights",
    [
        ("models/demos/t3000/mixtral8x7b/demo/input_data.json", False),
        ("models/demos/t3000/mixtral8x7b/demo/input_data_questions.json", True),
    ],
    ids=["general", "instruct"],
)
def test_mixtral8x7b_demo(t3k_device_mesh, use_program_cache, input_prompts, instruct_weights, is_ci_env):
    if is_ci_env and instruct_weights == True:
        pytest.skip("CI demo test only runs general weights to reduce CI pipeline load (both are supported)")

    for device in t3k_device_mesh.get_device_ids():
        t3k_device_mesh.get_device(device).enable_async(True)

    return run_mixtral_demo(
        user_input=input_prompts,
        batch_size=32,
        device_mesh=t3k_device_mesh,
        instruct_mode=instruct_weights,
        is_ci_env=is_ci_env,
    )
