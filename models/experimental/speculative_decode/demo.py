# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import json
from time import time
from loguru import logger
import os
import ttnn
import pytest
from pathlib import Path
from models.demos.wormhole.llama31_8b.tt.llama_common import (
    prepare_inputs_ttnn,
    sample,
    get_single_rot_mat,
    cache_attention_share_cache,
    encode_prompt_llama_instruct,
    HostEmbedding,
)
from models.demos.wormhole.llama31_8b.tt.llama_model import TtTransformer
from models.demos.wormhole.llama31_8b.tt.llama_embedding import TtLlamaEmbedding
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer

from models.experimental.speculative_decode.draft_models import NGramModel, load_data
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.console import Console


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


def preprocess_inputs(input_prompts, tokenizer, batch_size, dtype, instruct, device, embed_on_device=False):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    if instruct:
        encoded_prompts = [encode_prompt_llama_instruct(tokenizer, prompt) for prompt in input_prompts]
    else:
        encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in input_prompts]

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
    if embed_on_device:
        tt_out_tok = pt_tokenized_inputs[:, 0].view(1, batch_size)
    else:
        tt_out_tok = pt_tokenized_inputs[:, 0].view(batch_size, seqlen)

    return tt_out_tok, pt_tokenized_inputs, input_mask, None


# Load the draft model
def load_draft_model(tokenizer, n=5, laplace=0.01, data_type="sample-data"):
    if data_type == "sample-data":
        data_path = Path("models/experimental/speculative_decode/draft_models/n_gram/sample_data")
        train, _ = load_data(data_path)

    elif data_type == "wikitext-103":
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", ignore_verifications=True)
        train = ds["train"][:200000]["text"]
        # remove empty sentences ''
        train = [s for s in train if s != ""]
    else:
        raise ValueError("Invalid data choice")

    print("Loading {}-gram model...".format(n))
    lm = NGramModel(train, n, tokenizer, laplace=0.01)
    return lm


# Construct rotary mat for speculation
def get_rotary_mat_for_speculation(model_args, speculation_length, start_pos=0):
    current_rot_mat_torch, rot_matrix_torch = get_single_rot_mat(
        model_args.head_dim,
        "cpu",
        start_pos=start_pos,
    )
    # return current_rot_mat_torch, rot_matrix_torch
    rot_mats = [current_rot_mat_torch]
    for i in range(speculation_length):
        rot_mats.append(torch.matmul(rot_matrix_torch, rot_mats[-1]))
    current_rot_mat = torch.concat(rot_mats, dim=1)
    # # pad dim 1 to 32 to work around a hang in ttnn
    # current_rot_mat = torch.cat((current_rot_mat, torch.zeros(current_rot_mat.shape[0], 32 - current_rot_mat.shape[1], model_args.head_dim, model_args.head_dim)), dim=1)
    return current_rot_mat, rot_matrix_torch


def generate_prints(text, tok_per_s) -> Table:
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Generated Tokens", style="dim", width=120)
    table.add_column("Tokens/s", justify="right")

    table.add_row(text, f"{tok_per_s:.2f}")

    return table


def generate_scrollable_panel(text: Text, tok_per_s, max_lines=60, console: Console = None) -> Panel:
    # Wrap the text to simulate lines based on terminal width
    wrapped_text = text.wrap(console=console, width=120)  # Adjust width as needed

    # Limit the text to the last `max_lines` lines
    visible_text = Text()
    for line in wrapped_text[-max_lines:]:
        visible_text.append(line)

    # Create a panel for display with the token speed at the bottom
    panel = Panel(
        visible_text,
        title=f"Tokens/s: {tok_per_s:.2f}",
        border_style="magenta",
        padding=(1, 2),
    )
    return panel


def run_llama_demo(user_input, batch_size, device, instruct_mode, GENERATION_LENGTH=10, SPECULATION_LENGTH=3):
    # This module requires the env paths above for CI runs
    from models.demos.wormhole.llama31_8b.tt.model_config import TtModelArgs

    embed_on_device = True
    acceptance_criteria = "EXACT_MATCH"  # "PROBABILITY"
    dtype = ttnn.bfloat8_b

    # Load model args, weights, and tokenizer
    model_args = TtModelArgs(device, instruct=instruct_mode, max_batch_size=batch_size + SPECULATION_LENGTH)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    logger.info(f"Reading inputs...")
    input_prompts = load_inputs(user_input, batch_size)
    model_args.n_layers = 32
    model_args.share_cache = True  # enable share cache so that speculated tokens (in batch dim) can use the same cache

    logger.info("Loading weights...")
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
        )
    }
    logger.info("Loading weights finished!")

    # TODO Should we keep initial embedding on host?
    embd = HostEmbedding(model_args)
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    generation_start_pos = 0
    max_generated_tokens = GENERATION_LENGTH
    users_decoding = True

    # Preprocess initial prompt inputs
    tt_out_tok, pt_encoded_input, input_mask, rot_emb_matrix_list = preprocess_inputs(
        input_prompts, tokenizer, batch_size, dtype, instruct_mode, device, embed_on_device
    )
    # pre-compute the rotational embedding matrix and send to device
    current_rot_mat_torch, rot_matrix_torch = get_rotary_mat_for_speculation(
        model_args,
        SPECULATION_LENGTH,
        start_pos=0,
    )
    current_rot_mat = ttnn.from_torch(
        current_rot_mat_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=model_args.model_config["ROT_MAT_MEMCONFIG"],
    )
    logger.info("Caching attention ops...")
    cache_attention_share_cache(device, state_dict, model_args, current_rot_mat, dtype, [0, 200, 400])

    # if instruct_mode:
    #     tokenizer._model.pad_id = tokenizer._model.eos_id

    # Load TTNN llama model
    logger.info("Loading weights to device...")
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
    tt_embd = TtLlamaEmbedding(
        device=device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
    )
    logger.info("Finished loading weights to device. Starting inference...")

    # Load the draft model
    draft_model = load_draft_model(tokenizer, n=3, laplace=0.01, data_type="wikitext-103")
    num_correct_first_speculation = 0
    num_correct_second_speculation = 0
    num_correct_speculations = [0 for _ in range(SPECULATION_LENGTH)]

    # Keep track of generated outputs to print out every iteration
    all_outputs = []
    user_done = False  # Keeps track when a user reaches EoD token
    text = Text()

    # compile run
    dummy_input = torch.zeros(1, 1, 32, model_args.dim)
    dummy_input = ttnn.from_torch(dummy_input, device=tt_model.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_dummy_out = tt_model(dummy_input, list(range(0, SPECULATION_LENGTH + 1)), rot_mat=current_rot_mat)
    tt_dummy_out = ttnn.untilize(tt_dummy_out, use_multicore=False)
    tt_dummy_out.deallocate()
    if embed_on_device:
        dummy_out_tok = torch.zeros(1, 32)
        dummy_out_tok = ttnn.from_torch(
            dummy_out_tok,
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        tt_dummy_input = tt_embd(dummy_out_tok)
        tt_dummy_input.deallocate()

    breakpoint()

    # Initialize the console for use in wrapping text
    console = Console()

    iteration = 0
    num_calls_to_tt_model = 0
    total_start_time = time()
    with Live(generate_scrollable_panel(text, 0, console=console), refresh_per_second=10, console=console) as live:
        # Keep running inference as long as there is a user in the batch still decoding or max tokens per user are decoded
        while users_decoding:
            iteration_time_start = time()
            curr_pos = generation_start_pos + iteration

            # Get the speculated tokens from draft model
            speculated_tokens, _ = draft_model(all_outputs, generation_length=SPECULATION_LENGTH, sampling=False)

            # Construct model decode input based on speculated tokens
            # Prepare inputs for decode mode (rotary embeddings, attention mask, padding)
            if embed_on_device:
                # extend with the speculated tokens
                speculated_tok_torch = torch.tensor(speculated_tokens).view(1, SPECULATION_LENGTH)
                tt_out_tok = torch.cat((tt_out_tok, speculated_tok_torch), dim=1)

                # Pad tt_out_tok to batch size of 32
                padded_tt_out_tok = torch.zeros(1, 32, dtype=tt_out_tok.dtype, device=tt_out_tok.device)
                padded_tt_out_tok[:, : tt_out_tok.shape[1]] = tt_out_tok
                tt_out_tok = ttnn.from_torch(
                    padded_tt_out_tok,
                    device=device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                tt_decode_input = tt_embd(tt_out_tok)
                current_pos = curr_pos
                decode_input = tt_decode_input
            else:
                tt_decode_input = embd(tt_out_tok)
                tt_decode_speculated_input = embd(torch.tensor(speculated_tokens)).view(SPECULATION_LENGTH, 1, -1)
                tt_decode_input = torch.cat((tt_decode_input, tt_decode_speculated_input), dim=0)
                decode_input, current_pos = prepare_inputs_ttnn(
                    tt_decode_input,
                    curr_pos,
                    model_args.dim,
                    model_args.sliding_window,
                    tt_model.device,
                )
            current_pos = list(range(current_pos, current_pos + SPECULATION_LENGTH + 1))

            # Run ttnn llama model
            tt_out = tt_model(decode_input, current_pos, rot_mat=current_rot_mat)
            tt_out = ttnn.untilize(tt_out, use_multicore=False)
            tt_output_torch = (
                ttnn.to_torch(tt_out).permute(2, 1, 0, 3).squeeze(1)[: model_args.max_batch_size, :, :]
            )  # [batch, seq, hidden_dim]

            # Update rotation matrix for next iteration
            current_rot_mat_torch = torch.matmul(rot_matrix_torch, current_rot_mat_torch)
            current_rot_mat.deallocate()
            current_rot_mat = ttnn.from_torch(
                current_rot_mat_torch,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=model_args.model_config["ROT_MAT_MEMCONFIG"],
            )
            # If temperature is 0, does greedy decoding (top-1)
            tt_out_tok = sample(tt_output_torch, temperature=0, top_p=0.8)

            tok_index = 0
            if iteration < input_mask.shape[1]:  # If prefill
                # If token is pad token, start generating new token, otherwise, push the next prompt token to the model
                tt_out_tok = torch.where(
                    input_mask[:, iteration], pt_encoded_input[:, iteration], tt_out_tok[:, 0]
                ).unsqueeze(1)

                # Save output token to print out later
                user_tok = tt_out_tok[0].tolist()  # take the first, none speculated token
                if user_tok[0] != 28803 and user_done == False:  # Stop saving the ouput after hitting the EOS token
                    all_outputs.append(user_tok[0])

            else:
                if acceptance_criteria == "EXACT_MATCH":
                    while (
                        tok_index < len(speculated_tokens)
                        and int(tt_out_tok[tok_index]) == speculated_tokens[tok_index]
                    ):
                        num_correct_speculations[tok_index] += 1
                        tok_index += 1
                elif acceptance_criteria == "PROBABILITY":
                    raise NotImplementedError("PROBABILITY acceptance criteria not implemented yet")

                # Save output token to print out later
                user_tok = tt_out_tok.squeeze().tolist()
                if (
                    all([user_tok[idx] != 28803 for idx in range(tok_index + 1)]) and user_done == False
                ):  # Stop saving the ouput after hitting the EOS token
                    all_outputs.extend(user_tok[: tok_index + 1])  # TODO: highlight speculated tokens
                else:
                    user_done = True
                    users_decoding = False

            # take the next input as the last corrected speculated token
            tt_out_tok = tt_out_tok[tok_index : tok_index + 1]

            num_generated_tokens = tok_index + 1

            # Print out generated outputs for each user at the end of every iteration
            iteration_time = time() - iteration_time_start
            tokens_per_second_per_user = num_generated_tokens / iteration_time
            new_tokens = all_outputs[-num_generated_tokens:]
            decoded_tokens_non_spec = tokenizer.decode(new_tokens[: len(new_tokens) - tok_index])
            if tok_index > 0:
                # highlight the speculated tokens
                decoded_tokens_spec = tokenizer.decode(new_tokens[-tok_index:])
                text.append(decoded_tokens_non_spec)
                text.append(decoded_tokens_spec, style="green")
                # decoded_tokens = decoded_tokens_non_spec + "[green]" + decoded_tokens_spec + "[/green]"
            else:
                text.append(decoded_tokens_non_spec)
                # decoded_tokens = decoded_tokens_non_spec
            # text = text + "".join(decoded_tokens)
            live.update(generate_scrollable_panel(text, tokens_per_second_per_user, console=console))

            iteration += num_generated_tokens
            num_calls_to_tt_model += 1

            # Upper limit of generated tokens for each user (to avoid infinite generation in case eos is not seen)
            if iteration >= max_generated_tokens:
                users_decoding = False

    total_time = time() - total_start_time
    logger.info(f"Total time taken: {total_time:.2f} seconds")
    logger.info(f"Average tokens per second: {iteration / total_time:.2f}")
    logger.info(f"Average tokens per second (w/o speculative tokens): {num_calls_to_tt_model / total_time:.2f}")
    logger.info(f"Number of correct speculations: {num_correct_speculations}")


@pytest.mark.parametrize(
    "input_prompts, instruct_weights, generation_length, speculation_length",
    [
        ("models/experimental/speculative_decode/input_data_questions.json", True, 1000, 3),
    ],
    ids=["llama3.1_8b"],
)
def test_llama_demo(device, use_program_cache, input_prompts, instruct_weights, generation_length, speculation_length):
    return run_llama_demo(
        user_input=input_prompts,
        batch_size=1,
        device=device,
        instruct_mode=instruct_weights,
        GENERATION_LENGTH=generation_length,
        SPECULATION_LENGTH=speculation_length,
    )
