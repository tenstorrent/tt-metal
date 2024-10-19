# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import json
from time import time
from datetime import datetime
from loguru import logger
import os
import pytest
from models.demos.wormhole.qwen2_7b.tt.qwen2_common import (
    prepare_inputs_ttnn,
    sample,
    precompute_freqs,
    freqs_to_rotation_matrix,
    cache_attention,
    load_safetensor_weights,
)
from models.demos.wormhole.qwen2_7b.reference.tokenizer import Tokenizer
from models.demos.wormhole.qwen2_7b.reference.model import Transformer, Emb


# load from json, return as a list
def load_inputs(user_input, batch):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    in_prompt = []
    for i in range(batch):
        in_prompt.append(user_input[i]["prompt"])
    return in_prompt


def preprocess_inputs(input_prompts, tokenizer, model_args, embd, instruct):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    if instruct:
        # Pre append [INST] and post append [/INST] to the encoded prompts if instruct mode
        encoded_prompts = [tokenizer.encode("[INST] " + prompt + " [/INST]") for prompt in input_prompts]
    else:
        encoded_prompts = [tokenizer.encode(prompt) for prompt in input_prompts]

    prompt_lens = [len(x) for x in encoded_prompts]

    # Pad the inputs to the max length prompt
    max_prompt_len = max(prompt_lens)
    input_tokens = torch.full((len(input_prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.long)

    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = torch.tensor(encoded).to(input_tokens)
    input_mask = input_tokens != tokenizer.pad_id

    num_users = len(encoded_prompts)
    logger.info(f"# of users: {num_users}")

    seqlen = 1  # Generating one token per user at a time
    # Select the first token from the prompts for initial decoding
    pt_tokenized_inputs = torch.tensor(input_tokens)
    emb_inputs = embd(pt_tokenized_inputs[:, 0]).view(pt_tokenized_inputs.shape[0], seqlen, -1)

    return emb_inputs, pt_tokenized_inputs, input_mask


def run_qwen2_demo(user_input, batch_size, device, instruct_mode, is_ci_env, num_batches, print_to_file):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = "models/demos/wormhole/qwen2_7b/demo/output"
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o755)
    output_filename = f"{output_directory}/demo_user_output_{timestamp}.txt"
    # Set Qwen2 flags for CI
    if is_ci_env and instruct_mode:  # Update paths for instruct mode, otherwise use default paths for general weights
        os.environ["QWEN2_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Mistral/mistral-7B-v0.1/instruct/"
        os.environ["QWEN2_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/mistral-7B-v0.1/instruct/"
        os.environ["QWEN2_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/mistral-7B-v0.1/instruct/"

    # This module requires the env paths above for CI runs
    from models.demos.wormhole.qwen2_7b.tt.model_config import TtModelArgs

    # Load model args, weights, and tokenizer
    model_args = TtModelArgs(device, instruct=instruct_mode)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    logger.info(f"Reading inputs...")
    if len(user_input) == 1:
        input_prompts = user_input * batch_size  # Always process 32 users
    else:
        input_prompts = load_inputs(user_input, batch_size)
    model_args.n_layers = 28

    # Generate the batched prompts
    batch_prompts = []
    for i in range(num_batches):
        batch_prompts.append([input_prompts[(j + i) % len(input_prompts)] for j in range(len(input_prompts))])

    logger.info("Loading weights...")
    state_dict = load_safetensor_weights(model_args.consolidated_weights_path)
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
        )
    }
    logger.info("Loading weights finished!")

    reference_model = Transformer(args=model_args)

    partial_state_dict = {k[len("model.") :]: v for k, v in state_dict.items() if "layers." in k}
    partial_state_dict["norm.weight"] = state_dict["model.norm.weight"]
    partial_state_dict["lm_head.weight"] = state_dict["lm_head.weight"]
    reference_model.load_state_dict(partial_state_dict)

    # TODO Should we keep initial embedding on host?
    embd = Emb(model_args.vocab_size, model_args.dim, tokenizer.pad_id)
    embd.load_state_dict({"emb.weight": state_dict["model.embed_tokens.weight"]})

    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    freqs_cis = torch.complex(cos, sin)

    generation_start_pos = 0
    max_generated_tokens = 20
    users_decoding = True

    for batch_idx, input_prompts in enumerate(batch_prompts):
        # Preprocess initial prompt inputs
        tt_decode_input, pt_encoded_input, input_mask = preprocess_inputs(
            input_prompts, tokenizer, model_args, embd, instruct_mode
        )

        # Keep track of generated outputs to print out every iteration
        all_outputs = [[] for _ in range(batch_size)]
        user_done = [False] * batch_size  # Keeps track when a user reaches EoD token

        iteration = 0
        users_decoding = True  # reset to handle next batch
        # Keep running inference as long as there is a user in the batch still decoding or max tokens per user are decoded
        while users_decoding:
            iteration_time_start = time()
            current_pos = generation_start_pos + iteration

            # Run ttnn qwen2 model
            freqs_cis_i = freqs_cis[current_pos, :].unsqueeze(0)
            positions = torch.tensor([current_pos])
            # mask = ttnn.to_torch(attn_mask[0])
            ref_output = reference_model(tt_decode_input, freqs_cis_i, positions)

            # If temperature is 0, does greedy decoding (top-1)
            tt_out_tok = sample(ref_output, temperature=0, top_p=0.8)

            # TODO argmax on device
            # tt_out = ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)
            # tt_out = ttnn.permute(tt_out, (2, 1, 0, 3))
            # tt_out = ttnn.reshape(tt_out, (tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]))  # Squeeze(1)
            # tt_out_argmax = ttnn.argmax(tt_out, dim=-1)
            # Typecast from bf16 to uint32 for embedding
            # tt_out_tok = ttnn.clone(tt_out_argmax, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.uint32)
            # tt_out_tok = ttnn.experimental.tensor.typecast(tt_out_tok, dtype=ttnn.uint32)

            if iteration < input_mask.shape[1]:  # If prefill
                # If token is pad token, start generating new token, otherwise, push the next prompt token to the model
                tt_out_tok = torch.where(
                    input_mask[:, iteration], pt_encoded_input[:, iteration], tt_out_tok[:, 0]
                ).unsqueeze(1)

            # Save output token to print out later
            for user in range(batch_size):
                user_tok = tt_out_tok[user].tolist()
                if (
                    user_tok[0] != tokenizer.eos_id and user_done[user] == False
                ):  # Stop saving the ouput after hitting the EOS token
                    all_outputs[user].append(user_tok[0])
                else:
                    user_done[user] = True
                    if (
                        iteration < input_mask.shape[1]
                    ):  # Still in prefill, so ignore EOS token and save the generated token
                        # all_outputs[user].append(user_tok[0])
                        pass
                    else:
                        logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                        if all(user_done):
                            users_decoding = False
            tt_decode_input = embd(tt_out_tok)

            # Print out generated outputs for each user at the end of every iteration
            iteration_time = time() - iteration_time_start
            tokens_per_second_per_user = 1 / iteration_time
            if not is_ci_env:
                if len(user_input) == 1:
                    logger.info("[User 0] {}".format("".join(tokenizer.decode(all_outputs[0]))))
                else:
                    for user in range(batch_size):
                        text = "".join(tokenizer.decode(all_outputs[user]))
                        if len(text) > 100:
                            text = "..." + text[-97:]
                        text = text.replace("\n", " ")
                        logger.info("[User {}] {}".format(user, text))

            # Always print perf at every iteration
            logger.info(
                f"Iteration {iteration}: {1000*iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
            )

            iteration += 1

            # Upper limit of generated tokens for each user (to avoid infinite generation in case eos is not seen)
            if iteration >= max_generated_tokens:
                users_decoding = False

            if not users_decoding:
                with open(output_filename, "a") as f:
                    for i, (output, prompt) in enumerate(zip(all_outputs, input_prompts)):
                        text = tokenizer.decode(output)
                        if instruct_mode:
                            split_text = text.split("[/INST]", 1)
                        else:
                            split_text = text.split(prompt, 1)
                        if len(split_text) > 1:
                            text_after_prompt = split_text[1]
                        else:
                            text_after_prompt = text  # If prompt is not found, use the whole text
                        if print_to_file:
                            f.write(
                                f"\nbatch: {batch_idx} user: {i}\nprompt: {prompt} \noutput:\n{text_after_prompt}\n"
                            )
                        else:
                            print(f"\nbatch: {batch_idx} user: {i}\nprompt: {prompt} \noutput:\n{text_after_prompt}\n")


@pytest.mark.parametrize(
    "input_prompts, instruct_weights, num_batches",
    [
        # Combinations for general weights
        ("models/demos/wormhole/qwen2_7b/demo/input_data.json", False, 1),
        # FIXME(cthsieh): Will enable when ready.
        # ("models/demos/wormhole/mistral7b/demo/input_data.json", False, 2),
        # ("models/demos/wormhole/mistral7b/demo/input_data.json", False, 3),
        # ("models/demos/wormhole/mistral7b/demo/input_data.json", False, 4),
        # ("models/demos/wormhole/mistral7b/demo/input_data.json", False, 5),
        # Combinations for instruct weights
        # ("models/demos/wormhole/mistral7b/demo/input_data_questions.json", True, 1),
        # ("models/demos/wormhole/mistral7b/demo/input_data_questions.json", True, 2),
        # ("models/demos/wormhole/mistral7b/demo/input_data_questions.json", True, 3),
        # ("models/demos/wormhole/mistral7b/demo/input_data_questions.json", True, 4),
        # ("models/demos/wormhole/mistral7b/demo/input_data_questions.json", True, 5),
    ],
    ids=[
        "general_weights-1_batch",
        # FIXME(cthsieh): Will enable when ready.
        # "general_weights-2_batch",
        # "general_weights-3_batch",
        # "general_weights-4_batch",
        # "general_weights-5_batch",
        # "instruct_weights-1_batch",
        # "instruct_weights-2_batch",
        # "instruct_weights-3_batch",
        # "instruct_weights-4_batch",
        # "instruct_weights-5_batch",
    ],
)
def test_qwen2_7B_demo(device, use_program_cache, input_prompts, instruct_weights, is_ci_env, num_batches):
    if (is_ci_env and instruct_weights == False) or (is_ci_env and not (num_batches == 1 or num_batches == 3)):
        pytest.skip(
            "CI demo test only runs instruct weights (1 and 3 batches) to reduce CI pipeline load (both are supported)"
        )

    return run_qwen2_demo(
        user_input=input_prompts,
        batch_size=8,
        device=device,
        instruct_mode=instruct_weights,
        is_ci_env=is_ci_env,
        num_batches=num_batches,
        print_to_file=False,
    )
