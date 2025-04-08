# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from loguru import logger
from datetime import datetime
import hashlib
import requests
import json

import torch
import pytest
import os
import ttnn

from models.demos.llama3_subdevices.tt.generator import Generator
from models.demos.llama3_subdevices.tt.model_config import LlamaOptimizations
from models.tt_transformers.tt.common import (
    preprocess_inputs_prefill,
    PagedAttentionConfig,
    sample_host,
)
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.demos.utils.llm_demo_utils import create_benchmark_data


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

    # Clip the context to the max length provided
    if max_length:
        context_text = context_text[:max_length]
        logger.info(f"Clipped the context text to {max_length} characters")

    return context_text


# load input prompts from json, return as a list
def load_inputs(user_input, batch, instruct):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)

    if len(user_input) < batch:
        logger.warning(
            f"Number of users in the file is less than the provided batch={batch}. Repeating the prompts to match the batch size."
        )
        user_input = user_input * batch

    in_prompt = []
    cache_dir = Path("models/tt_transformers/demo/context_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # The demo supports a custom prompt file, where the context is provided by a link to a book from the gutenberg project
    # It clips the excerpt to the max length provided to allow testing different long context lengthts
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
    from models.demos.llama3_subdevices.tt.llama_model import TtTransformer
    from models.demos.llama3_subdevices.tt.model_config import TtModelArgs

    tt_model_args = TtModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
    )
    # tt_model_args.n_layers = 1
    state_dict = tt_model_args.load_state_dict()

    page_table = None
    paged_attention_config = None
    tt_kv_cache = None

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
        mode="prefill",
    )

    if use_paged_kv_cache:
        tt_kv_cache = [l.attention.layer_past for l in model.layers]

    return tt_model_args, model, page_table, [tt_kv_cache]


# List of supported Parameters for demo.py
#
# input_prompts (string): input json file with prompts to process. See models/tt_transformers/demo/*.json for list of input files
# instruct (bool): Whether to use instruct weights or general weights
# repeat_batches (int): Number of consecutive batches of users to run (default: 1)
# max_seq_len (int): Maximum context length supported by the model (Llama3.1 and Llama3.2 models have a maximum context length of 128k, i.e., 128 * 1024)
# batch_size (int): Number of users in a batch (Supports 1/2/4/8/16/32 batches)
# max_generated_tokens (int): Maximum number of tokens to generate for each user (Note that the users will stop generation before this limit if they reach a EoS token)
# paged_attention (bool): Whether to use paged attention or default attention (vLLM requires paged attention)
# page_params (dict): Page parameters for paged attention (block_size, max_num_blocks) For smaller context lengths use block_size=32 and max_num_blocks=1024, for larger context use block_size=64 and max_num_blocks=2048
# sampling_params (dict): Sampling parameters for decoding (temperature, top_p). If temperature is set to 0, argmax (greedy decode) is used.
# stop_at_eos (bool): Whether to stop decoding when the model generates an EoS token
#
# optimization (LlamaOptimizations): Optimization level to use for the model (performance or accuracy)
# MESH_DEVICE (str): Fake device to use for testing (N150, N300, T3K, TG). Usage: `export MESH_DEVICE=N150`, will enable running a single-chip demo on a multi-chip system.
@pytest.mark.parametrize(
    "input_prompts, instruct, repeat_batches, max_seq_len, batch_size, max_generated_tokens, paged_attention, page_params, sampling_params, stop_at_eos, ci_only",
    [
        (  # Batch-1 run (Latency) - single user, small prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            1,  # batch_size
            150,  # max_generated_tokens
            False,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
        ),
        (  # Repeat-5 Batch-32 run (Throughput) - 32 users, small prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            32,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
        ),
        (  # Batch-32 run (Throughput) - 32 users, small prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            5,  # repeat_batches
            1024,  # max_seq_len
            32,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
        ),
        (  # Long-context run - Single user, long prompt (adapted to the model being used and architecture)
            "models/tt_transformers/demo/sample_prompts/input_data_long_64k.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
        ),
        (  # Batch-1 run (Reasoning) - single user, small prompt, long thinking time
            "models/tt_transformers/demo/input_data_questions_reasoning.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            16 * 1024,  # max_seq_len
            1,  # batch_size
            15000,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks": 1024},  # page_params  # TODO This will be serviced by vLLM
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # ci_only
        ),
        (  # CI Batch-1 run - Measures the performance of a single user over 4096 iterations
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            8192,  # max_seq_len
            1,  # batch_size
            4096,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            True,  # ci_only
        ),
        (  # CI Batch-32 run - Measures the performance of a 32 users over 4096 iterations
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            2000,  # max_seq_len
            32,  # batch_size
            1024,  # max_generated_tokens  # TODO Update this to 4096, and make sure it fits in DRAM with correct page_params
            True,  # paged_attention  # TODO Find the correct paged_attn params to avoid hangs in this config with long context generation
            {"page_block_size": 64, "page_max_num_blocks": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            True,  # ci_only
        ),
    ],
    ids=[
        "batch-1",  # latency
        "batch-32",  # throughput
        "repeat5-batch-32",  # throughput with 5 repeat batches
        "long-context",  # max-length
        "reasoning-1",  # reasoning
        "ci-1",  # CI batch 1
        "ci-32",  # CI batch 32
    ],
)
@pytest.mark.parametrize(
    "optimizations",
    [
        LlamaOptimizations.performance,
        LlamaOptimizations.accuracy,
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 23887872, "num_command_queues": 1, "dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_demo_text(
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
    stop_at_eos,
    mesh_device,
    use_program_cache,
    is_ci_env,
    ci_only,
    reset_seeds,
    request,
):
    """
    Simple demo with limited dependence on reference code.
    """

    if is_ci_env and (optimizations == LlamaOptimizations.accuracy or not ci_only):
        pytest.skip("CI only runs the CI-only tests")

    # TODO: Remove this once all batch sizes are supported on TG
    if os.environ.get("MESH_DEVICE") == "TG" and batch_size not in [1, 32]:
        pytest.skip("TG only supports batch 1 and 32")

    mesh_device.enable_async(True)
    enable_trace = True  # Use tracing for better perf
    print_to_file = False  # Enable this flag to print the output of all users to a file

    # Override parameters from command line if they are provided
    # input_prompts = request.config.getoption("--input_prompts") or input_prompts
    # if request.config.getoption("--instruct") in [
    #     0,
    #     1,
    # ]:  # If the flag is provided, use it. Take an int instead of bool due to parser limitations
    #     instruct = request.config.getoption("--instruct")
    # repeat_batches = request.config.getoption("--repeat_batches") or repeat_batches
    # max_seq_len = request.config.getoption("--max_seq_len") or max_seq_len
    # batch_size = request.config.getoption("--batch_size") or batch_size
    # max_generated_tokens = request.config.getoption("--max_generated_tokens") or max_generated_tokens
    # paged_attention = request.config.getoption("--paged_attention") or paged_attention
    # page_params = request.config.getoption("--page_params") or page_params
    # sampling_params = request.config.getoption("--sampling_params") or sampling_params
    # if request.config.getoption("--stop_at_eos") in [
    #     0,
    #     1,
    # ]:  # If the flag is provided, use it. Take an int instead of bool due to parser limitations
    #     stop_at_eos = request.config.getoption("--stop_at_eos")
    stop_at_eos = False
    if not stop_at_eos:
        logger.info(f"The decode generation will only stop at the max_generated_tokens limit == {max_generated_tokens}")

    if print_to_file:
        # Creat batch output file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_directory = "models/tt_transformers/demo/output"
        os.makedirs(output_directory, exist_ok=True)
        os.chmod(output_directory, 0o755)
        output_filename = f"{output_directory}/llama_text_demo_output_{timestamp}.txt"

    # Start profiler
    logger.info(f"Start profiler")
    profiler = BenchmarkProfiler()
    profiler.start("run")

    logger.info(f"Reading inputs...")
    profiler.start("loading_inputs")
    if len(input_prompts) == 1:  # Manual input
        input_prompts = input_prompts * batch_size
    else:  # Inputs from file
        input_prompts = load_inputs(input_prompts, batch_size, input_prompts)
    profiler.end("loading_inputs")

    # To simulate a deployment environment, the demo supports repeating batched prompts.
    # This loop will rotate the prompts between the users for each batch, to simulate users sending different requests
    # If batch_size=1, the same prompt is repeated for each batch
    repeat_batch_prompts = []
    for i in range(repeat_batches):
        repeat_batch_prompts.append([input_prompts[(j + i) % len(input_prompts)] for j in range(len(input_prompts))])

    model_args, model, page_table, tt_kv_cache = create_tt_model(
        mesh_device,
        instruct=instruct,
        max_batch_size=batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        page_params=page_params,
        dtype=ttnn.bfloat8_b,
        use_paged_kv_cache=paged_attention,
    )

    # model_args.tokenizer = Tokenizer(model_args.tokenizer_path)
    tokenizer = model_args.tokenizer
    generator = Generator(model, model_args, mesh_device, tokenizer=tokenizer)

    num_tokens_generated_decode = []

    logger.info("Starting inference...")
    for batch_idx, input_prompts in enumerate(repeat_batch_prompts):
        logger.info(f"Processing batch {batch_idx}")
        profiler.start(f"preprocess_prefill_inputs", iteration=batch_idx)
        # Preprocess initial prompt inputs
        try:
            (
                input_tokens_prefill_pt,
                encoded_prompts,
                decoding_pos,
                prefill_lens,
            ) = preprocess_inputs_prefill(
                input_prompts,
                tokenizer,
                [model_args],
                instruct,
                max_generated_tokens,
            )
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")

        max_encoded_prompt_len = max(len(p) for p in encoded_prompts)
        assert (
            max_generated_tokens + max_encoded_prompt_len <= max_seq_len
        ), f"Prompt prefill tokens ({max_encoded_prompt_len}) + maximum number of decoded iterations ({max_generated_tokens}) needs to be <= than max_seq_len ({max_seq_len})"

        profiler.end(f"preprocess_prefill_inputs", iteration=batch_idx)

        # when doing repeating batches, set kv-caches to zero, to avoid context leaking
        if batch_idx != 0:
            model.switch_mode("prefill")
            for layer in model.layers:
                k_cache, v_cache = layer.attention.layer_past
                k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
                v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)
        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)

        logger.info("Starting prefill warmup...")
        profiler.start(f"compile_prefill", iteration=batch_idx)

        logits = generator.prefill_forward_text(
            input_tokens_prefill_pt[0].unsqueeze(0),  # Just warmup prefill for 1 user
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
        )
        profiler.end(f"compile_prefill", iteration=batch_idx)
        logger.info("Finished prefill warmup")

        logger.info(f"Starting prefill...")
        profiler.start(f"inference_prefill", iteration=batch_idx)
        logits = generator.prefill_forward_text(
            input_tokens_prefill_pt[0].unsqueeze(0),
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
        )
        prefilled_token = torch.argmax(logits, dim=-1)
        profiler.end(f"inference_prefill", iteration=batch_idx)
        logger.info(f"Prefill finished")

        if prefilled_token.shape[0] != 32:
            prefilled_token = prefilled_token.repeat(batch_size, 1)

        # Keep track of generated outputs to print out every iteration
        all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(batch_size)]
        for user in range(batch_size):
            user_tok = int(prefilled_token[0].item())
            all_outputs[user].append(user_tok)
        # print("Prefill outputs:", [tokenizer.decode(output) for output in all_outputs])
        # model.tt_ccl.close()
        # return True
        user_done = [False] * batch_size  # Keeps track when a user reaches EoD token

        # TODO Argmax on device is only supported for batch_size=1
        argmax_on_device = True  # False if (batch_size > 1 or sampling_params["temperature"] != 0) else True

        # Initial positions
        current_pos = torch.tensor([decoding_pos[0] for b in range(batch_size)])

        # Start decoding
        iteration = 0
        users_decoding = True

        out_tok = prefilled_token  # .repeat(batch_size, 1)
        try:
            model.switch_mode("decode")
        except Exception as e:
            logger.error(f"Error switching to decode mode: {str(e)}")
            model.tt_ccl.close()
        logger.info(f"Starting decode loop...")

        # Log total inference (accounting for compile_decode as well)
        profiler.start(f"inference_decode", iteration=batch_idx)
        while users_decoding:
            if iteration == 0:  # First iteration also accounts for compile time
                profiler.start(f"compile_decode", iteration=batch_idx)
            else:
                profiler.start(f"inference_decode_time_{iteration}", iteration=batch_idx)

            # Run decode forward
            try:
                logits = generator.decode_forward_text(
                    out_tok,
                    current_pos,
                    enable_trace=enable_trace,
                    page_table=page_table,
                    kv_cache=tt_kv_cache,
                    argmax_on_device=argmax_on_device,
                )
            except Exception as e:
                logger.error(f"Error during decoding: {str(e)}")
                break

            # Get the next token
            if argmax_on_device:
                out_tok = logits.unsqueeze(1)
                for b in range(1, 32):
                    out_tok[b][0] = 0
            else:
                # TODO Fix use case with temperature > 0
                _, out_tok = sample_host(
                    logits,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    on_host=True,
                )

            if iteration == 0:  # First iteration will account the compile time
                profiler.end(f"compile_decode", iteration=batch_idx)
                decode_iteration_time = profiler.get_duration("compile_decode", iteration=batch_idx)
            else:
                profiler.end(f"inference_decode_time_{iteration}", iteration=batch_idx)
                decode_iteration_time = profiler.get_duration(f"inference_decode_time_{iteration}", iteration=batch_idx)

            # Always print perf after every iteration
            tokens_per_second_per_user = 1 / decode_iteration_time
            logger.info(
                f"Iteration {iteration}: {1000*decode_iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
            )

            current_pos += 1

            # Save output token to print out later
            for user in range(batch_size):
                user_tok = out_tok[user].item()
                if (
                    user_tok not in tokenizer.stop_tokens and user_done[user] == False
                ):  # Read until an eos token (e.g. <|eot_id|>); create_tokenizer adds stop_tokens to HF tokenizers
                    all_outputs[user].append(user_tok)
                else:
                    if (
                        stop_at_eos
                    ):  # For performance gathering in CI, we want to sometimes force decoding for a fixed number of iterations
                        user_done[user] = True
                        logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                        if all(user_done):
                            users_decoding = False

            # Print out generated outputs for each user at the end of every iteration
            if not is_ci_env:
                for user in range(batch_size):
                    text = "".join(tokenizer.decode(all_outputs[user]))
                    if len(text) > 100:
                        text = "..." + text[-97:]
                    text = text.replace("\n", " ")
                    logger.info("[User {}] {}".format(user, text))

            iteration += 1

            # Upper limit of generated tokens for each user
            if iteration >= max_generated_tokens:
                users_decoding = False

            # Final print
            if not users_decoding:
                profiler.start(f"log_saving_file", iteration=batch_idx)
                logger.info("Finished decoding, printing the final outputs...\n")
                for i, (output, prompt) in enumerate(zip(all_outputs, input_prompts)):
                    text = tokenizer.decode(output)
                    prompt_including_assistant_tags = tokenizer.decode(
                        model_args.encode_prompt(prompt, instruct=instruct)
                    )
                    text_after_prompt = text.replace(prompt_including_assistant_tags, "", 1)
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
                            f"\n==REPEAT BATCH {batch_idx}\n==USER {i} - PROMPT\n{short_prompt} \n==USER {i} - OUTPUT\n{text_after_prompt.strip()}\n"
                        )
                    break
                profiler.end(f"log_saving_file", iteration=batch_idx)

        num_tokens_generated_decode.append(iteration)  # Save the number of tokens generated for each repeat batch

    profiler.end(f"inference_decode", iteration=batch_idx)

    # Finish profiling at the end of inference for all repeated batches
    profiler.end("run")

    # Prepare profile benchmark metrics for the first repeat batch only
    compile_prefill_time = profiler.get_duration("compile_prefill")
    compile_decode_time = profiler.get_duration("compile_decode")

    total_inference_prefill_time = profiler.get_duration("inference_prefill")
    total_inference_decode_time = 0
    for i in range(1, iteration):  # Iteration 0 is the compile time
        total_inference_decode_time += profiler.get_duration(f"inference_decode_time_{i}")

    # Average prefill time for each user
    avg_time_to_first_token = total_inference_prefill_time / batch_size
    # Average decode time per batch iteration
    avg_decode_iteration_time = total_inference_decode_time / (iteration - 1)

    prefill_tok_s = prefill_lens[0] / total_inference_prefill_time * batch_size
    decode_tok_s_user = (num_tokens_generated_decode[0] - 1) / total_inference_decode_time  # Remove the compile time
    decode_tok_s = (
        (num_tokens_generated_decode[0] - 1) / total_inference_decode_time * batch_size
    )  # Remove the compile time

    measurements = {
        # Required measurements
        "compile_prefill": compile_prefill_time,
        "compile_decode": compile_decode_time,
        "inference_prefill": total_inference_prefill_time,
        "inference_decode": total_inference_decode_time,
        "prefill_time_to_token": avg_time_to_first_token,
        "prefill_t/s": prefill_tok_s,  # tokens/s
        "decode_t/s/u": decode_tok_s_user,  # tokens/s/u
        "decode_t/s": decode_tok_s,  # tokens/s
        # Optional measurements
        "Total compile time": compile_prefill_time + compile_decode_time,
        "Full demo runtime": profiler.get_duration("run"),
    }

    # Decode performance for some specific tokens
    tok_1_perf = profiler.get_duration(f"inference_decode_time_{1}")  # Iteration 0 is compile time
    tok_128_perf = profiler.get_duration(f"inference_decode_time_{127}") if 127 < iteration else 0
    tok_1024_perf = profiler.get_duration(f"inference_decode_time_{1023}") if 1023 < iteration else 0
    tok_4096_perf = profiler.get_duration(f"inference_decode_time_{4095}") if 4095 < iteration else 0

    if not stop_at_eos:
        logger.info(f"Please note that 'stop_at_eos' is disabled. Output repetition is expected.")

    logger.info("")
    logger.info(f"=== Performance metrics ===")
    logger.info(
        f"1st token decode time: {tok_1_perf*1000:.2f}ms [{round(1/tok_1_perf, 2)} t/s/u, {round((1/tok_1_perf)*batch_size, 2)} t/s]"
    )
    if tok_128_perf > 0:
        logger.info(
            f"128th token decode time: {tok_128_perf*1000:.2f}ms [{round(1/tok_128_perf, 2)} t/s/u, {round((1/tok_128_perf)*batch_size, 2)} t/s]"
        )
    if tok_1024_perf > 0:
        logger.info(
            f"1024th token decode time: {tok_1024_perf*1000:.2f}ms [{round(1/tok_1024_perf, 2)} t/s/u, {round((1/tok_1024_perf)*batch_size, 2)} t/s]"
        )
    if tok_4096_perf > 0:
        logger.info(
            f"4096th token decode time: {tok_4096_perf*1000:.2f}ms [{round(1/tok_4096_perf, 2)} t/s/u, {round((1/tok_4096_perf)*batch_size, 2)} t/s]"
        )

    # Print some of the perf metrics
    logger.info("==")
    logger.info(f"Prefill compile time: {round(compile_prefill_time, 2)}s")
    logger.info(f"Decode compile time: {round(compile_decode_time, 2)}s")
    logger.info("")
    logger.info(f"Average Time to First Token (TTFT): {round(avg_time_to_first_token*1000, 2)}ms")
    logger.info(
        f"Average speed: {round(avg_decode_iteration_time * 1000, 2)}ms @ {round(decode_tok_s_user, 2)} tok/s/user ({round(decode_tok_s, 2)} tok/s throughput)"
    )

    # Benchmark targets
    supported_models = ["Llama3.1-70B", "Deepseek-R1-Distill-70B"]
    # model_args.base_model_name = "Llama3.1-70B"
    supported_devices = ["TG"]

    tt_device_name = model_args.device_name

    # Set the target times to first token for every combination of device and model
    target_prefill_tok_s = {
        "TG_Llama3.1-70B": 1050,  # TODO Update target
        "TG_Deepseek-R1-Distill-70B": 1050,  # TODO Update target
    }[f"{tt_device_name}_{model_args.base_model_name}"]

    # Set the target decode timesfor every combination of device and model
    target_decode_tok_s_u = {
        "TG_Llama3.1-70B": 20,  # TODO Update target
        "TG_Deepseek-R1-Distill-70B": 20,  # TODO Update target
    }[f"{tt_device_name}_{model_args.base_model_name}"]

    target_decode_tok_s = target_decode_tok_s_u * batch_size
    targets = {
        "prefill_t/s": target_prefill_tok_s,
        "decode_t/s": target_decode_tok_s,
        "decode_t/s/u": target_decode_tok_s_u,
    }

    # Save benchmark data for CI dashboard
    if is_ci_env:
        # Instead of running warmup iterations, the demo profiles the initial compile iteration
        bench_n_warmup_iter = {"inference_prefill": 0, "inference_decode": 1}
        benchmark_data = create_benchmark_data(profiler, measurements, bench_n_warmup_iter, targets)

        # Save the decode performance of every iteration for plotting in superset
        for i in range(1, iteration):
            benchmark_data.add_measurement(
                profiler,
                0,
                "inference_decode",
                f"time_to_token_{i}",
                profiler.get_duration(f"inference_decode_time_{i}") * 1000,
                step_warm_up_num_iterations=None,
                target=None,
            )

        # Also save the avg decode performance for the 128 iterations (excluding the compile time)
        inference_decode_time_first_128 = sum(
            profiler.get_duration(f"inference_decode_time_{i}") for i in range(1, 128)
        )
        benchmark_data.add_measurement(
            profiler,
            0,
            "inference_decode",
            "avg_decode_time_first_128",
            inference_decode_time_first_128 * 1000 / 127,
            step_warm_up_num_iterations=None,
            target=None,
        )

        benchmark_data.save_partial_run_json(
            profiler,
            run_type=f"{tt_device_name}-demo",
            ml_model_name=model_args.base_model_name,
            ml_model_type="llm",
            num_layers=model_args.n_layers,
            batch_size=batch_size,
            input_sequence_length=max(prefill_lens),
            output_sequence_length=num_tokens_generated_decode[0],
        )
