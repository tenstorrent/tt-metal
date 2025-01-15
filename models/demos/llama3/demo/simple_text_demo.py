# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional
from loguru import logger

from time import time
from datetime import datetime
import hashlib
import requests
import json

import math
from termcolor import cprint

# import llama_models.llama3.reference_impl.generation as llama_reference_generation
# from llama_models.llama3.api.tokenizer import Tokenizer
# TODO, use the tokenizer above or below?
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer

# from llama_models.llama3.api.chat_format import ChatFormat
# from llama_models.llama3.api.datatypes import ImageMedia, UserMessage

from pkg_resources import resource_filename

from models.demos.llama3.tt.llama_common import (
    get_prefill_rot_mat,
    get_rot_transformation_mat,
    # HostEmbedding,
    encode_prompt_llama_instruct,
    PagedAttentionConfig,
    sample_host,
)

from models.perf.benchmarking_utils import BenchmarkProfiler
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.demos.llama3.tt.model_config import LlamaOptimizations


import torch
import pytest
import os
import ttnn

from models.demos.llama3.tt.generator import LlamaGenerator


def load_and_cache_context(context_url, cache_dir, max_length=None):
    cache_file = cache_dir / hashlib.md5(context_url.encode()).hexdigest()

    if cache_file.exists():
        with open(cache_file, "r") as f:
            context_text = f.read()
        logger.info(f"Loaded context from cache: {context_url}")
    else:
        try:
            response = requests.get(context_url)
            if response.status_code == 200:
                context_text = response.text
                with open(cache_file, "w") as f:
                    f.write(context_text)
                logger.info(f"Downloaded and cached context: {context_url}")
            else:
                logger.warning(f"Failed to fetch context from URL: {context_url}. Status code: {response.status_code}")
                context_text = ""
        except Exception as e:
            logger.error(f"Error fetching context from URL: {context_url}. Error: {str(e)}")
            context_text = ""


# load from json, return as a list
def load_inputs(user_input, batch, instruct):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    in_prompt = []
    cache_dir = Path("models/demos/llama3/demo/context_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    for i in range(batch):
        prompt = user_input[i]["prompt"]
        if "context" in user_input[i]:
            if "max_length" in user_input[i]:  # Clip the context to the max length provided
                context_text = load_and_cache_context(
                    user_input[i]["context"], cache_dir, max_length=user_input[i]["max_length"]
                )
            else:
                context_text = load_and_cache_context(user_input[i]["context"], cache_dir)
            if instruct:
                prompt = (
                    "```" + context_text + "```\n\n" + prompt
                )  # Add the markdown block to the context to comply with the prompt
            else:
                prompt = context_text
        in_prompt.append(prompt)
    return in_prompt


def preprocess_inputs_prefill(
    input_prompts,
    tokenizer,
    model_args,
    instruct,
    max_generated_tokens,
    max_prefill_len=128 * 1024,
):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    # The maximum KV-cache len supported is 32k. To avoid going out of memory, clip the max prefill length by the maximum number of tokens that will be generated
    if max_prefill_len == 128 * 1024:
        max_prefill_len = 128 * 1024 - max_generated_tokens

    if instruct:
        encoded_prompts = [encode_prompt_llama_instruct(tokenizer, prompt) for prompt in input_prompts]
    else:
        encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in input_prompts]

    # Print the length of encoded prompts
    logger.info("Encoded prompt lengths:" + ", ".join(str(len(prompt)) for prompt in encoded_prompts))

    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    # The large input demo we provide contains more tokens than the maximum (32k tokens)
    # To avoid running out of memory, clip to max_prefill_len
    if min_prompt_len > max_prefill_len:
        logger.info(f"Clipping prompts to {max_prefill_len}")
        if instruct:  # When clipping, make sure to add the ` 】 token at the end (4 tokens)
            encoded_prompts = [encod[: max_prefill_len - 4] for encod in encoded_prompts]
            dec_prompts = [tokenizer.decode(encod) + " 】" for encod in encoded_prompts]
            encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in dec_prompts]
        else:
            encoded_prompts = [encod[:max_prefill_len] for encod in encoded_prompts]

        # Update prompt lengths
        prompt_lens = [len(x) for x in encoded_prompts]
        min_prompt_len = min(prompt_lens)
        max_prompt_len = max(prompt_lens)

    assert (
        max_prompt_len <= model_args.max_seq_len
    ), f"Max prompt length {max_prompt_len} exceeds model max seq len {model_args.max_seq_len}"
    assert min_prompt_len > 0, "Minimum prompt length must be greater than 0"
    assert min_prompt_len <= max_prompt_len, f"Minimum prompt length {min_prompt_len} exceeds max len {max_prompt_len}"

    logger.info(f"# of users: {len(encoded_prompts)}")
    input_tokens_prefill = []
    decoding_pos = []
    prefill_lens = []

    # Always prefill the nearest power of 2 for each user. This means that the majority of cases we will prefill more tokens than needed.
    # To avoid issues, we keep track of the decoding position to decode correctly the user's prompt
    for i, encoded in enumerate(encoded_prompts):
        # Prefill size is nearest power of 2
        prefill_seq_len = max(2 ** math.ceil(math.log(len(encoded), 2)), 128)

        # Initial prefill tensors full of pad tokens
        input_tokens_prefill_i = torch.full((1, prefill_seq_len), 0, dtype=torch.int32)
        input_tokens_prefill_i[0, : len(encoded[:])] = torch.tensor(encoded[:]).to(input_tokens_prefill_i)
        input_tokens_prefill.append(input_tokens_prefill_i)

        # Keep the correct decoding position of each user
        decoding_pos.append(len(encoded))
        prefill_lens.append(prefill_seq_len)

    return (
        input_tokens_prefill,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    )


def create_tt_model(
    mesh_device,
    instruct,
    max_batch_size,
    optimizations,
    max_seq_len,
    page_params,
    dtype=ttnn.bfloat8_b,
    use_paged_kv_cache=False,
):
    from models.demos.llama3.tt.llama_model import TtTransformer
    from models.demos.llama3.tt.model_config import TtModelArgs

    tt_model_args = TtModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
    )
    state_dict = tt_model_args.load_state_dict()

    # page_table_tt = None
    page_table = None
    paged_attention_config = None
    if use_paged_kv_cache:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            tt_model_args.max_batch_size, paged_attention_config.max_num_blocks // tt_model_args.max_batch_size
        )
        # page_table_tt = ttnn.from_torch(
        #     page_table,
        #     device=mesh_device,
        #     dtype=ttnn.int32,
        #     layout=ttnn.ROW_MAJOR_LAYOUT,
        #     mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=tt_model_args.cluster_shape),
        # )
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )

    model = TtTransformer(
        args=tt_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )
    # TODO Miguel send page table to generator!
    return tt_model_args, model, page_table


# List of supported Parameters for demo.py
#
# input_prompts (string): input json file with prompts to process. See models/demos/llama3/demo/*.json for list of input files
# instruct (bool): Whether to use instruct weights or general weights
# repeat_batches (int): Number of consecutive batches of users to run (default: 1)
# max_seq_len (int): Maximum context length supported by the model (Llama3.1 and Llama3.2 models have a maximum context length of 128k, i.e., 128 * 1024)
# batch_size (int): Number of users in a batch (Supports 1/2/4/8/16/32 batches)
# max_generated_tokens (int): Maximum number of tokens to generate for each user (Note that the users will stop generation before this limit if they reach a EoS token)
# paged_attention (bool): Whether to use paged attention or default attention (vLLM requires paged attention)
# page_params (dict): Page parameters for paged attention (block_size, max_num_blocks) For smaller context lengths use block_size=32 and max_num_blocks=1024, for larger context use block_size=64 and max_num_blocks=2048
# sampling_params (dict): Sampling parameters for decoding (temperature, top_p). If temperature is set to 0, argmax (greedy decode) is used.
#
# optimization (LlamaOptimizations): Optimization level to use for the model (performance or accuracy)
# FAKE_DEVICE (str): Fake device to use for testing (N150, N300, T3K, TG). Usage: `export FAKE_DEVICE=N150`, will enable running a single-chip demo on a multi-chip system.
@pytest.mark.parametrize(
    "input_prompts, instruct, repeat_batches, max_seq_len, batch_size, max_generated_tokens, paged_attention, page_params, sampling_params",
    [
        (  # Batch-1 run (Latency) - single user, small prompt
            "models/demos/llama3/demo/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            1,  # batch_size
            200,  # max_generated_tokens
            False,  # paged_attention # Miguel, set to True after adding to generator!
            {"page_block_size": 32, "page_max_num_blocks": 1024},  # page_params  # TODO This will be serviced by vLLM
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
        ),
        (  # Batch-32 run (Throughput) - 32 users, small prompt
            "models/demos/llama3/demo/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            32,  # batch_size
            200,  # max_generated_tokens
            False,  # paged_attention  # Miguel
            {"page_block_size": 32, "page_max_num_blocks": 1024},  # page_params  # TODO This will be serviced by vLLM
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
        ),
        (  # Long-context run - Single user, long prompt (adapted to the model being used and architecture)
            "models/demos/llama3/demo/input_data_long_64k.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            64 * 1024,  # max_seq_len
            1,  # batch_size
            200,  # max_generated_tokens
            False,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params  # TODO This will be serviced by vLLM
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
        ),
    ],
    ids=[
        "batch-1",  # latency
        "batch-32",  # throughput
        "long-context",  # max-length
    ],
)
@pytest.mark.parametrize(
    "optimizations",
    [
        LlamaOptimizations.performance,
        LlamaOptimizations.accuracy,
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_llama_demo_text(
    input_prompts,
    instruct,
    repeat_batches,
    max_seq_len,
    batch_size,
    max_generated_tokens,
    paged_attention,
    page_params,
    sampling_params,
    optimizations,
    mesh_device,
    use_program_cache,
    is_ci_env,
    reset_seeds,
):
    """
    Simple Llama demo with limited dependence on reference code.
    """
    mesh_device.enable_async(True)  # Miguel fix with blocking calls!

    print_to_file = False

    # Creat batch output file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = "models/demos/llama3/demo/output"
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o755)
    output_filename = f"{output_directory}/llama_text_demo_output_{timestamp}.txt"

    # We disregard any warmup iteration for profiling, in favour of just measuring compile time on the first iteration
    N_warmup_iter = {"inference_prefill": 0, "inference_decode": 0}

    # Start profiler
    logger.info(f"Start profiler")
    profiler = BenchmarkProfiler()
    profiler.start("run")

    logger.info(f"Reading inputs...")
    profiler.start("loading_inputs")
    if len(input_prompts) == 1:
        input_prompts = input_prompts * batch_size
    else:
        input_prompts = load_inputs(input_prompts, batch_size, input_prompts)
    profiler.end("loading_inputs")

    # Generate the batched prompts (rotate the inputs between the users, for each batch)
    # If batch_size == 1, the same prompt is repeated for each batch
    batch_prompts = []
    for i in range(repeat_batches):
        batch_prompts.append([input_prompts[(j + i) % len(input_prompts)] for j in range(len(input_prompts))])

    model_args, model, page_table = create_tt_model(
        mesh_device,
        instruct=instruct,
        max_batch_size=batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        page_params=page_params,
        dtype=ttnn.bfloat8_b,
        use_paged_kv_cache=paged_attention,
    )

    generator = LlamaGenerator(model, model_args, mesh_device)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    num_tokens_generated_decode = []

    logger.info("Starting inference...")
    for batch_idx, input_prompts in enumerate(batch_prompts):
        logger.info(f"Processing batch {batch_idx}")
        profiler.start(f"preprocess_prefill_inputs", iteration=batch_idx)
        # Preprocess initial prompt inputs
        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prompt_lens,  # Miguel change name to prefill lens!
        ) = preprocess_inputs_prefill(
            input_prompts,
            tokenizer,
            model_args,
            instruct,
            max_generated_tokens,
        )

        max_encoded_prompt_len = max(len(p) for p in encoded_prompts)
        assert (
            max_generated_tokens + max_encoded_prompt_len <= max_seq_len
        ), f"Prompt prefill tokens ({max_encoded_prompt_len}) + maximum number of decoded iterations ({max_generated_tokens}) needs to be <= than max_seq_len ({max_seq_len})"

        profiler.end(f"preprocess_prefill_inputs", iteration=batch_idx)

        # set kv cache to zeros if not first batch, to avoid context leaking when doing multiple batches
        if batch_idx != 0:
            for layer in model.layers:
                k_cache, v_cache = layer.attention.layer_past
                k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
                v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)

        logger.info(f"Starting prefill...")

        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)

        tt_kv_cache = [l.attention.layer_past for l in model.layers]
        # Miguel fix this (not pass the page table to model config! in the demo parameters)
        # logits = generator.prefill_forward_text(input_tokens_prefill_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
        logits = generator.prefill_forward_text(input_tokens_prefill_pt, prompt_lens=decoding_pos)
        logger.info(f"Prefill finished")

        # Preparing first decode token
        profiler.start(f"prepare_first_decode_token_{batch_idx}")
        # pt_out_batched = logits.squeeze()
        # pt_out_batched = torch.argmax(pt_out_batched, dim=-1)
        pt_out_batched = torch.argmax(logits, dim=-1)

        profiler.end(f"prepare_first_decode_token_{batch_idx}")

        # Keep track of generated outputs to print out every iteration
        all_outputs = [encoded_prompts[b][: prompt_lens[b]] for b in range(batch_size)]
        for user in range(batch_size):
            user_tok = int(pt_out_batched[user].item())
            all_outputs[user].append(user_tok)

        user_done = [False] * batch_size  # Keeps track when a user reaches EoD token

        logger.info("Starting decode...")

        # Shard the page table for TG decode
        # if paged_attention and model_args.is_galaxy and batch_size > 1:
        #     page_table_tt = ttnn.from_torch(
        #         page_table,
        #         device=mesh_device,
        #         dtype=ttnn.int32,
        #         layout=ttnn.ROW_MAJOR_LAYOUT,
        #         mesh_mapper=ttnn.ShardTensor2dMesh(
        #             mesh_device,
        #             dims=(None, -2) if batch_size > 1 else (None, None),
        #             mesh_shape=model_args.cluster_shape,
        #         ),
        #     )

        # Set sampling mode
        argmax_on_device = False if (batch_size > 1 or sampling_params["temperature"] != 0) else True

        # Initial positions
        current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)])

        # Start decoding
        iteration = 0
        users_decoding = True  # reset to handle next batch
        total_decoding_time = 0  # Track total decoding time
        total_tokens_generated = 0  # Track total tokens generated

        logger.info(f"Starting decode loop...")
        profiler.start(f"inference_decode", iteration=batch_idx)

        out_tok = pt_out_batched

        while users_decoding:
            if iteration == 0:  # First iteration also accounts for compile time
                profiler.start(f"compile_decode", iteration=batch_idx)
            iteration_time_start = time()

            # Necessary padding to be full tile sized
            padding_time = time()
            out_tok = torch.nn.functional.pad(out_tok, (0, 32 - len(out_tok)), "constant", 0)
            logger.info(f"Miguel: padding time: {1000*(time() - padding_time)}ms")

            # logits = generator.decode_forward_text(pt_out_batched, current_pos, page_table=page_table, kv_cache=tt_kv_cache)
            logits = generator.decode_forward_text(out_tok, current_pos, enable_trace=True)

            argmax_time = time()
            # TODO Miguel - Re-add argmax on device
            _, out_tok = sample_host(
                logits,
                None,
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                on_host=True,
            )

            logger.info(f"Miguel: Argmax time: {1000*(time() - argmax_time)}ms")

            current_pos += 1

            # Print out generated outputs for each user at the end of every iteration
            iteration_time = time() - iteration_time_start

            # Save output token to print out later
            for user in range(batch_size):
                # user_tok = tt_output_torch[user].tolist()
                user_tok = out_tok[user].item()
                if (
                    user_tok != 128009 and user_done[user] == False
                ):  # Stop saving the ouput after hitting the eos token (<|eot_id|>) (128009)
                    all_outputs[user].append(user_tok)
                else:
                    user_done[user] = True
                    logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                    if all(user_done):
                        users_decoding = False

            # Ignore the first iteration for average speed calculation
            if iteration > 0:
                total_decoding_time += iteration_time
                total_tokens_generated += 1

            tokens_per_second_per_user = 1 / iteration_time

            profiler.start(f"log_printing_iter_{iteration}", iteration=batch_idx)
            # Print out generated outputs for each user at the end of every iteration
            if not is_ci_env:
                if len(input_prompts) == 1:
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
            profiler.end(f"log_printing_iter_{iteration}", iteration=batch_idx)

            if iteration == 0:  # First iteration also accounts for compile time
                profiler.end(f"compile_decode", iteration=batch_idx)

            iteration += 1

            # Upper limit of generated tokens for each user (to avoid infinite generation in case eos is not seen)
            if iteration >= max_generated_tokens:
                users_decoding = False

            if not users_decoding:
                profiler.start(f"log_saving_file", iteration=batch_idx)
                for i, (output, prompt) in enumerate(zip(all_outputs, input_prompts)):
                    text = tokenizer.decode(output)
                    if instruct:
                        split_text = text.split("<|start_header_id|>assistant<|end_header_id|>", 1)
                    else:
                        split_text = text.split(prompt, 1)
                    if len(split_text) > 1:
                        text_after_prompt = split_text[1]
                    else:
                        text_after_prompt = text  # If prompt is not found, use the whole text
                    if print_to_file:
                        with open(output_filename, "a") as f:
                            f.write(
                                f"\nbatch: {batch_idx} user: {i}\nprompt: {prompt} \noutput:\n{text_after_prompt}\n"
                            )
                    else:
                        # Strip leading newlines from output when sent to terminal
                        short_prompt = (
                            (prompt[:100] + "\n<long prompt not printed in full>\n" + prompt[-100:])
                            if len(prompt) > 200
                            else prompt
                        )
                        logger.info(
                            f"\nbatch: {batch_idx} user: {i}\nprompt: {short_prompt} \noutput:\n{text_after_prompt.strip()}\n"
                        )
                profiler.end(f"log_saving_file", iteration=batch_idx)

        num_tokens_generated_decode.append(
            total_tokens_generated
        )  # Save the number of tokens generated for each batch (excluding the first token)

        profiler.end(f"inference_decode", iteration=batch_idx)

    # Finish profiling at the end of all batches inference
    profiler.end("run")

    logger.info("Miguel: Re-add the benchmark profiling")
    # # Prepare profile benchmark metrics for batch 0
    # compile_prefill_time = profiler.get_duration("compile_prefill")
    # compile_decode_time = profiler.get_duration("compile_decode")
    # inference_prefill_time = profiler.get_duration("inference_prefill")
    # inference_decode_time = profiler.get_duration("inference_decode")
    # log_printing_time = sum(profiler.get_duration(f"log_printing_iter_{i}") for i in range(total_tokens_generated))
    # log_saving_file_time = profiler.get_duration(f"log_saving_file")

    # # Correct the inference decode time to remove the time spent on compile (1st iteration) and log_printing (at the end of every iteration)
    # inference_decode_time = inference_decode_time - compile_decode_time - log_printing_time - log_saving_file_time
    # # Correct the inference prefill time to remove the time spent on compile (1st iteration)
    # inference_prefill_time = inference_prefill_time - compile_prefill_time
    # # Average prefill time for each user
    # prefill_time_to_first = inference_prefill_time / num_users_generated_prefill

    # measurements = {
    #     # Required measurements
    #     "compile_prefill": compile_prefill_time,
    #     "compile_decode": compile_decode_time,
    #     "inference_prefill": inference_prefill_time,
    #     "inference_decode": inference_decode_time,
    #     "prefill_time_to_token": prefill_time_to_first,
    #     "prefill_t/s": num_users_generated_prefill / inference_prefill_time * prefill_seq_len,  # tokens/s
    #     "decode_t/s/u": num_tokens_generated_decode[0] / inference_decode_time,  # tokens/s/u
    #     "decode_t/s": num_tokens_generated_decode[0] / inference_decode_time * batch_size,  # tokens/s
    #     # Optional measurements
    #     "loading_inputs": profiler.get_duration("loading_inputs"),
    #     "weight_loading": profiler.get_duration("weight_loading"),
    #     "prepare_first_decode_token": profiler.get_duration("prepare_first_decode_token_0"),
    #     "preprocess_prefill_inputs": profiler.get_duration("preprocess_prefill_inputs"),
    #     "loading_weights_to_device": profiler.get_duration("loading_weights_to_device"),
    #     "Total compile time": compile_prefill_time + compile_decode_time,
    #     "Full demo runtime": profiler.get_duration("run"),
    # }

    # # Print some of the perf metrics
    # logger.info("")
    # logger.info(f"Performance metrics for batch 0")
    # logger.info(f"Prefill compile time: {round(measurements['compile_prefill'], 4)}s")
    # logger.info(f"Decode compile time: {round(measurements['compile_decode'], 4)}s")
    # logger.info(f"Prefill inference time per user: {round(inference_prefill_time/num_users_generated_prefill, 4)}s")
    # logger.info(
    #     f"Total Decode inference time ({total_tokens_generated-1} iterations): {round(measurements['inference_decode'], 4)}s"
    # )
    # logger.info("")
    # logger.info(f"Time to first token: {round(measurements['prefill_time_to_token']* 1000, 2)}ms")
    # logger.info(
    #     f"Average speed: {round(inference_decode_time / num_tokens_generated_decode[0] * 1000, 2)}ms @ {round(measurements['decode_t/s/u'], 2)} tok/s/user ({round(measurements['decode_t/s'], 2)} tok/s throughput)"
    # )
    # logger.info("")

    # supported_models = ["3.2-1B", "3.2-3B", "3.1-8B", "3.2-11B", "3.1-70B"]
    # supported_devices = ["N150", "N300", "T3K", "TG"]

    # # TODO update targets based on the llama3 model and the target device
    # llama_model_name = model_args.model_name
    # tt_device_name = model_args.device_name

    # assert llama_model_name in supported_models, f"Model {llama_model_name} not supported"
    # assert tt_device_name in supported_devices, f"Device {tt_device_name} not supported"

    # # Set the target times to first token for every combination of device and model
    # target_prefill_tok_s = {
    #     "N150_3.2-1B": 1050,  # TODO Update target
    #     "N300_3.2-1B": 1050,  # TODO Update target
    #     "T3K_3.2-1B": 1050,  # TODO Update target
    #     "TG_3.2-1B": 1050,  # TODO Update target
    #     #
    #     "N150_3.2-3B": 1050,  # TODO Update target
    #     "N300_3.2-3B": 1050,  # TODO Update target
    #     "T3K_3.2-3B": 1050,  # TODO Update target
    #     "TG_3.2-3B": 1050,  # TODO Update target
    #     #
    #     "N150_3.1-8B": 1050,
    #     "N300_3.1-8B": 1050,
    #     "T3K_3.1-8B": 1050,
    #     "TG_3.1-8B": 1050,
    #     #
    #     "N150_3.2-11B": 1050,  # TODO Update target
    #     "N300_3.2-11B": 1050,  # TODO Update target
    #     "T3K_3.2-11B": 1050,  # TODO Update target
    #     "TG_3.2-11B": 1050,  # TODO Update target
    #     #
    #     "N150_3.1-70B": 1050,  # TODO Update target
    #     "N300_3.1-70B": 1050,  # TODO Update target
    #     "T3K_3.1-70B": 1050,  # TODO Update target
    #     "TG_3.1-70B": 1050,  # TODO Update target
    # }[f"{tt_device_name}_{llama_model_name}"]

    # # Set the target decode timesfor every combination of device and model
    # target_decode_tok_s_u = {
    #     "N150_3.2-1B": 160,  # TODO Update target
    #     "N300_3.2-1B": 250,  # TODO Update target
    #     "T3K_3.2-1B": 300,  # TODO Update target
    #     "TG_3.2-1B": 300,  # TODO Update target
    #     #
    #     "N150_3.2-3B": 60,  # TODO Update target
    #     "N300_3.2-3B": 100,  # TODO Update target
    #     "T3K_3.2-3B": 150,  # TODO Update target
    #     "TG_3.2-3B": 150,  # TODO Update target
    #     #
    #     "N150_3.1-8B": 23,  # TODO Update target
    #     "N300_3.1-8B": 38,
    #     "T3K_3.1-8B": 45,
    #     "TG_3.1-8B": 45,  # TODO Update target
    #     #
    #     "N150_3.2-11B": 23,
    #     "N300_3.2-11B": 38,  # TODO Update target
    #     "T3K_3.2-11B": 45,  # TODO Update target
    #     "TG_3.2-11B": 45,  # TODO Update target
    #     #
    #     "T3K_3.1-70B": 20,  # TODO Update target
    #     "TG_3.1-70B": 20,  # TODO Update target
    # }[f"{tt_device_name}_{llama_model_name}"]

    # target_decode_tok_s = target_decode_tok_s_u * batch_size
    # targets = {
    #     "prefill_t/s": target_prefill_tok_s,
    #     "decode_t/s": target_decode_tok_s,
    #     "decode_t/s/u": target_decode_tok_s_u,
    # }

    # # Save benchmark data for CI dashboard
    # if is_ci_env:
    #     benchmark_data = create_benchmark_data(profiler, measurements, N_warmup_iter, targets)
    #     benchmark_data.prep_csvs(
    #         profiler,
    #         run_type=f"{tt_device_name}-demo",
    #         ml_model_name=llama_model_name,
    #         ml_model_type="llm",
    #         num_layers=model_args.n_layers,
    #         batch_size=batch_size,
    #         input_sequence_length=prefill_seq_len,
    #         output_sequence_length=1,
    #     )
