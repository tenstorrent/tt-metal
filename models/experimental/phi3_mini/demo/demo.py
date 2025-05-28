# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from datetime import datetime

import torch
import pytest
import os
import ttnn

from models.experimental.phi3_mini.tt.phi3_mini_generator import Phi3MiniGenerator
from models.tt_transformers.tt.generator import SamplingParams
from models.tt_transformers.tt.model_config import DecodersPrecision, parse_decoder_json

from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    sample_host,
)

from models.experimental.phi3_mini.tt.phi3_mini_common import (
    create_tt_model,
    preprocess_inputs_prefill,
)

from models.perf.benchmarking_utils import BenchmarkProfiler
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.tt_transformers.demo.simple_text_demo import load_inputs, create_tt_page_table


def prepare_generator_args(
    num_devices,
    data_parallel,
    mesh_device,
    instruct,
    global_batch_size,
    optimizations,
    max_seq_len,
    page_params,
    paged_attention,
):
    # Partition the mesh, singular model implemented for TP on 1xN mesh
    submesh_devices = (
        mesh_device.create_submeshes(ttnn.MeshShape(1, num_devices // data_parallel))
        if isinstance(mesh_device, ttnn.MeshDevice) and data_parallel > 1
        else [mesh_device]
    )
    state_dict = None

    # Hybrid requires a model per submesh
    model_args = []
    model = []
    tt_kv_cache = []

    paged_attention_config = (
        PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks_per_dp"],
        )
        if paged_attention
        else None
    )

    for submesh in submesh_devices:
        model_args_i, model_i, tt_kv_cache_i, state_dict = create_tt_model(
            submesh,
            instruct=instruct,
            max_batch_size=global_batch_size // data_parallel,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
        )
        model_args.append(model_args_i)
        model.append(model_i)
        tt_kv_cache.append(tt_kv_cache_i)

    page_table = create_tt_page_table(
        global_batch_size=global_batch_size,
        data_parallel=data_parallel,
        paged_attention_config=paged_attention_config,
    )
    # Host code, safe to reuse tokenizer from the 1st model
    tokenizer = model_args[0].tokenizer
    return model_args, model, page_table, tt_kv_cache, tokenizer


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
# optimization (ModelOptimizations): Optimization level to use for the model (performance or accuracy)
# MESH_DEVICE (str): Fake device to use for testing (N150, N300, T3K, TG). Usage: `export MESH_DEVICE=N150`, will enable running a single-chip demo on a multi-chip system.
@pytest.mark.parametrize(
    "input_prompts, instruct, repeat_batches, max_seq_len, batch_size, max_generated_tokens, paged_attention, page_params, sampling_params, stop_at_eos, ci_only, data_parallel",
    [
        (  # Batch-1 run (Latency) - single user, small prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            1,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
            1,
        ),
        (  # Batch-32 run (Throughput) - 32 users, small prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            32,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
            1,  # data_parallel
        ),
        (  # Long-context run - Single user, long prompt (adapted to the model being used and architecture)
            "models/tt_transformers/demo/sample_prompts/input_data_long_32k.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
            1,  # data_parallel
        ),
        (  # Batch-1 run (Reasoning) - single user, small prompt, long thinking time
            "models/tt_transformers/demo/input_data_questions_reasoning.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            16 * 1024,  # max_seq_len
            1,  # batch_size
            15000,  # max_generated_tokens
            True,  # paged_attention
            {
                "page_block_size": 32,
                "page_max_num_blocks_per_dp": 1024,
            },  # page_params  # TODO This will be serviced by vLLM
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # ci_only
            1,  # data_parallel
        ),
        (  # CI Batch-1 run - Measures the performance of a single user over 4096 iterations
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            8192,  # max_seq_len
            1,  # batch_size
            4096,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            True,  # ci_only
            1,  # data_parallel
        ),
        (  # CI Batch-32 run - Measures the performance of a 32 users over 4096 iterations
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            2000,  # max_seq_len
            32,  # batch_size
            1024,  # max_generated_tokens  # TODO Update this to 4096, and make sure it fits in DRAM with correct page_params
            True,  # paged_attention  # TODO Find the correct paged_attn params to avoid hangs in this config with long context generation
            {"page_block_size": 64, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            True,  # ci_only
            1,  # data_parallel
        ),
    ],
    ids=[
        "batch-1",  # latency
        "batch-32",  # throughput
        "long-context",  # max-length
        "reasoning-1",  # reasoning
        "ci-1",  # CI batch 1
        "ci-32",  # CI batch 32
    ],
)
@pytest.mark.parametrize(
    "optimizations",
    [
        lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
        lambda model_args: DecodersPrecision.accuracy(model_args.n_layers, model_args.model_name),
    ],
    ids=["performance", "accuracy"],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
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
    data_parallel,
    reset_seeds,
    request,
):
    """
    Simple demo with limited dependence on reference code.
    """
    test_id = request.node.callspec.id
    if is_ci_env and (("accuracy" in test_id) or not ci_only):
        pytest.skip("CI only runs the CI-only tests")

    # TODO: Remove this once all batch sizes are supported on TG
    if os.environ.get("MESH_DEVICE") == "TG" and batch_size not in [1, 32]:
        pytest.skip("TG only supports batch 1 and 32")

    if ("long" in test_id) and os.environ.get("MESH_DEVICE") == "N150":
        pytest.skip("N150 is not able run long context test")

    enable_trace = True  # Use tracing for better perf
    print_to_file = False  # Enable this flag to print the output of all users to a file

    input_prompts = request.config.getoption("--input_prompts") or input_prompts
    if request.config.getoption("--instruct") in [
        0,
        1,
    ]:  # If the flag is provided, use it. Take an int instead of bool due to parser limitations
        instruct = request.config.getoption("--instruct")
    repeat_batches = request.config.getoption("--repeat_batches") or repeat_batches
    max_seq_len = request.config.getoption("--max_seq_len") or max_seq_len
    batch_size = request.config.getoption("--batch_size") or batch_size
    max_generated_tokens = request.config.getoption("--max_generated_tokens") or max_generated_tokens
    paged_attention = request.config.getoption("--paged_attention") or paged_attention
    page_params = request.config.getoption("--page_params") or page_params
    sampling_params = request.config.getoption("--sampling_params") or sampling_params
    json_config_file = request.config.getoption("--decoder_config_file")
    if json_config_file:
        optimizations = parse_decoder_json(json_config_file)
    else:
        optimizations = request.config.getoption("--optimizations") or optimizations

    if request.config.getoption("--stop_at_eos") in [
        0,
        1,
    ]:  # If the flag is provided, use it. Take an int instead of bool due to parser limitations
        stop_at_eos = request.config.getoption("--stop_at_eos")
    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1
    global_batch_size = batch_size * data_parallel  # input batch_size is interpreted as size per DP group

    # uneven split of devices per DP group not supported
    if data_parallel > num_devices or num_devices % data_parallel != 0:
        pytest.skip(f"Invalid number of DP groups: {data_parallel}, for {num_devices} devices")

    if is_ci_env and data_parallel > 1 and batch_size > 1:
        pytest.skip("CI runs only hybrid with batch 1 per submesh")

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
        input_prompts = input_prompts * global_batch_size
    else:  # Inputs from file
        input_prompts = load_inputs(input_prompts, global_batch_size, input_prompts)
    profiler.end("loading_inputs")

    # To simulate a deployment environment, the demo supports repeating batched prompts.
    # This loop will rotate the prompts between the users for each batch, to simulate users sending different requests
    # If batch_size=1, the same prompt is repeated for each batch
    repeat_batch_prompts = []
    for i in range(repeat_batches):
        repeat_batch_prompts.append([input_prompts[(j + i) % len(input_prompts)] for j in range(len(input_prompts))])

    model_args, model, page_table, tt_kv_cache, tokenizer = prepare_generator_args(
        num_devices=num_devices,
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        instruct=instruct,
        global_batch_size=global_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=paged_attention,
    )
    generator = Phi3MiniGenerator(model, model_args, mesh_device, tokenizer=tokenizer)

    num_tokens_generated_decode = []

    logger.info("Starting inference...")
    for batch_idx, input_prompts in enumerate(repeat_batch_prompts):
        logger.info(f"Processing batch {batch_idx}")
        profiler.start(f"preprocess_prefill_inputs", iteration=batch_idx)
        # Preprocess initial prompt inputs
        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
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

        # when doing repeating batches, set kv-caches to zero, to avoid context leaking
        if batch_idx != 0:
            for i in range(len(model)):
                for layer in model[i].layers:
                    k_cache, v_cache = layer.attention.layer_past
                    k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
                    v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)

        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)

        logger.info("Starting prefill warmup...")
        profiler.start(f"compile_prefill", iteration=batch_idx)
        logits = generator.prefill_forward_text(
            input_tokens_prefill_pt[::batch_size, :],  # Warmup prefill for each device
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
        )
        profiler.end(f"compile_prefill", iteration=batch_idx)
        logger.info("Finished prefill warmup")

        logger.info(f"Starting prefill...")
        profiler.start(f"inference_prefill", iteration=batch_idx)
        logits = generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
        )
        prefilled_token = torch.argmax(logits, dim=-1)
        profiler.end(f"inference_prefill", iteration=batch_idx)
        logger.info(f"Prefill finished")

        # Keep track of generated outputs to print out every iteration
        all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(global_batch_size)]
        for user in range(global_batch_size):
            user_tok = int(prefilled_token[user].item())
            all_outputs[user].append(user_tok)

        user_done = [False] * global_batch_size  # Keeps track when a user reaches EoD token

        # TODO Argmax on device is only supported for batch_size=1 (per submesh)
        argmax_on_device = (
            False if (global_batch_size // data_parallel > 1 or sampling_params["temperature"] != 0) else True
        )
        if argmax_on_device:
            device_sampling_params = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0)
        else:
            device_sampling_params = None

        # Initial positions
        current_pos = torch.tensor([decoding_pos[b] for b in range(global_batch_size)])

        # Start decoding
        iteration = 0
        users_decoding = True

        out_tok = prefilled_token

        logger.info(f"Starting decode loop...")

        # Log total inference (accounting for compile_decode as well)
        profiler.start(f"inference_decode", iteration=batch_idx)
        while users_decoding:
            if iteration == 0:  # First iteration also accounts for compile time
                profiler.start(f"compile_decode", iteration=batch_idx)
            else:
                profiler.start(f"inference_decode_time_{iteration}", iteration=batch_idx)

            # Run decode forward
            logits = generator.decode_forward_text(
                out_tok,
                current_pos,
                enable_trace=enable_trace,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                sampling_params=device_sampling_params,
            )

            # Get the next token
            if device_sampling_params is not None:
                out_tok = logits.unsqueeze(1)
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
                f"Iteration {iteration}: {1000*decode_iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({global_batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
            )

            current_pos += 1

            # Save output token to print out later
            for user in range(global_batch_size):
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
                for user in range(global_batch_size):
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
                        model_args[0].encode_prompt(prompt, instruct=instruct)
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
    avg_time_to_first_token = total_inference_prefill_time / global_batch_size
    # Average decode time per batch iteration
    avg_decode_iteration_time = total_inference_decode_time / (iteration - 1)

    prefill_tok_s = prefill_lens[0] / total_inference_prefill_time * global_batch_size
    decode_tok_s_user = (num_tokens_generated_decode[0] - 1) / total_inference_decode_time  # Remove the compile time
    decode_tok_s = (
        (num_tokens_generated_decode[0] - 1) / total_inference_decode_time * global_batch_size
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
        f"1st token decode time: {tok_1_perf*1000:.2f}ms [{round(1/tok_1_perf, 2)} t/s/u, {round((1/tok_1_perf)*global_batch_size, 2)} t/s]"
    )
    if tok_128_perf > 0:
        logger.info(
            f"128th token decode time: {tok_128_perf*1000:.2f}ms [{round(1/tok_128_perf, 2)} t/s/u, {round((1/tok_128_perf)*global_batch_size, 2)} t/s]"
        )
    if tok_1024_perf > 0:
        logger.info(
            f"1024th token decode time: {tok_1024_perf*1000:.2f}ms [{round(1/tok_1024_perf, 2)} t/s/u, {round((1/tok_1024_perf)*global_batch_size, 2)} t/s]"
        )
    if tok_4096_perf > 0:
        logger.info(
            f"4096th token decode time: {tok_4096_perf*1000:.2f}ms [{round(1/tok_4096_perf, 2)} t/s/u, {round((1/tok_4096_perf)*global_batch_size, 2)} t/s]"
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

    tt_device_name = model_args[0].device_name

    logger.warning(f"Model {model_args[0].base_model_name} not does not have performance targets set")
    targets = {}

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
            ml_model_name=model_args[0].base_model_name,
            ml_model_type="llm",
            num_layers=model_args[0].n_layers,
            batch_size=global_batch_size,
            input_sequence_length=max(prefill_lens),
            output_sequence_length=num_tokens_generated_decode[0],
        )
