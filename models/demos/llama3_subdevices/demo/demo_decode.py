# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import json
from time import time
from datetime import datetime
from loguru import logger
import os
import ttnn
import pytest
import requests
from pathlib import Path
import hashlib

from models.demos.llama3_subdevices.tt.llama_common import (
    PagedAttentionConfig,
)
from models.demos.llama3_subdevices.tt.llama_model import TtTransformer
from models.demos.llama3_subdevices.tt.llama_embedding import TtLlamaEmbedding
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs
from models.demos.llama3_subdevices.tt.sampling import TTSampling

from models.perf.benchmarking_utils import BenchmarkProfiler
from models.demos.llama3_subdevices.tt.model_config import LlamaOptimizations


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


# load from json, return as a list
def load_inputs(user_input, batch, instruct_mode):
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
            # if instruct_mode:
            #     prompt = (
            #         "```" + context_text + "```\n\n" + prompt
            #     )  # Add the markdown block to the context to comply with the prompt
            # else:
            prompt = context_text
        in_prompt.append(prompt)
    return in_prompt


def run_llama3_demo(
    user_input,
    mesh_device,
    max_seq_len,
    batch_size,
    num_batches,
    paged_attention,
    paged_attention_config,
    max_generated_tokens,
    optimizations,
    sampling_params,
    instruct_mode,
    is_ci_env,
    print_to_file,
    weights,
    layers,
    stress_test,
    start_pos,
):
    # Creat batch output file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = "models/demos/llama3/demo/output"
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o755)
    output_filename = f"{output_directory}/demo_user_output_{timestamp}.txt"

    dtype = ttnn.bfloat8_b
    assert batch_size <= 32, "Max batch size currently supported is 32"
    assert max_seq_len <= 128 * 1024, "Max sequence length must be less than 128k tokens"

    dummy_weights = weights == "random"

    # We disregard any warmup iteration for profiling, in favour of just measuring compile time on the first iteration
    N_warmup_iter = {"inference_prefill": 0, "inference_decode": 0}

    # Start profiler
    logger.info(f"Start profiler")
    profiler = BenchmarkProfiler()
    profiler.start("run")

    logger.info(f"Reading inputs...")
    profiler.start("loading_inputs")
    if len(user_input) == 1:
        input_prompts = user_input * batch_size
    else:
        input_prompts = load_inputs(user_input, batch_size, instruct_mode)
    profiler.end("loading_inputs")

    # Generate the batched prompts (rotate the inputs between the users, for each batch)
    # If batch_size == 1, the same prompt is repeated for each batch
    batch_prompts = []
    for i in range(num_batches):
        batch_prompts.append([input_prompts[(j + i) % len(input_prompts)] for j in range(len(input_prompts))])

    # Load model args, weights, and tokenizer
    model_args = TtModelArgs(
        mesh_device,
        instruct=instruct_mode,
        max_batch_size=batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        dummy_weights=dummy_weights,
    )
    model_args.n_layers = layers

    tokenizer = Tokenizer(model_args.tokenizer_path)

    # Check max sequence length compatibility with model and architecture. Refer to README for more information
    llama_model_name = model_args.model_name  # ["3.2-1B", "3.2-3B", "3.1-8B", "3.2-11B", "3.1-70B"]
    tt_device_name = model_args.device_name  # ["N150", "N300", "T3K", "TG"]

    if llama_model_name == "3.1-70B":
        assert tt_device_name in ["TG"], "Llama3.1-70B is only supported on TG"
        assert max_seq_len <= 128 * 1024, "TG supports the official max context length of 128k tokens for Llama3.1-70B"

    logger.info("Loading weights...")
    profiler.start("weight_loading")
    state_dict = model_args.load_state_dict()
    profiler.end("weight_loading")

    page_table_tt = None
    if paged_attention:
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.batch_size_per_device_group,
            paged_attention_config.max_num_blocks // model_args.batch_size_per_device_group,
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        )
        logger.info("Page table tensor done")

    # Load TTNN Llama3.1 model
    logger.info("Loading weights to device...")
    profiler.start("loading_weights_to_device")
    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )
    tt_embd = TtLlamaEmbedding(
        mesh_device=mesh_device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
    )
    tt_sampling = TTSampling(
        args=model_args,
        mesh_device=mesh_device,
        sampling_params=sampling_params,
        tt_ccl=tt_model.tt_ccl,
    )
    profiler.end("loading_weights_to_device")
    logger.info("Finished loading weights to device.")

    # Keep track of generated outputs to print out every iteration
    if dummy_weights:
        encoded_prompts = [
            [128000, 2028, 374, 264, 1296]
        ] * model_args.max_batch_size  # "This is a test" encoded prompt
    else:
        # if instruct_mode:
        #     encoded_prompts = [encode_prompt_llama_instruct(tokenizer, prompt) for prompt in input_prompts]
        # else:
        encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in input_prompts]

    # Prefill by decode: start at first token; pad to 32 (tile size)
    max_prompt_length = max([len(prompt) for prompt in encoded_prompts])
    padded_token_prompts = [prompt + [128009] * (max_prompt_length - len(prompt)) for prompt in encoded_prompts]
    encoded_prompts_tensor_whole_sequence = torch.tensor([padded_token_prompts[b] for b in range(batch_size)])

    user_done = [False] * batch_size  # Keeps track when a user reaches EoD token

    logger.info("Starting decode...")
    # Initial positions
    decoding_pos = [start_pos] * batch_size
    current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)])

    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    logger.info("Current pos tensor done")

    # Get cos/sin matrices for the current position of each user
    rot_mats, rot_mat_idxs = tt_model.rope_setup.get_rot_mats(current_pos, return_rot_idxs=True)

    logger.info("Rot mats done")

    # Prepare the encoded prompts for the decode input
    tt_out_tok = ttnn.from_torch(
        encoded_prompts_tensor_whole_sequence[:, :1].reshape(1, 1, 1, batch_size),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )

    # Compile
    logger.info(f"Compiling model trace...")
    if layers == 1:
        num_compile_iters = 10
    elif layers == 5:
        num_compile_iters = 2
    else:
        num_compile_iters = 1
    for i in range(num_compile_iters):
        tt_decode_input = tt_embd(tt_out_tok)
        logger.info(f"tt_decode_input done")

        tt_out = tt_model(
            tt_decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        logger.info(f"tt_out done")

        # Sampling
        # tt_out_tok_reset = tt_sampling(tt_out[0], tt_out_tok)

        # Note: Persistent output buffer used, do not deallocate output!
        tt_out_gathered = tt_model.tt_ccl.line_all_gather(
            tt_out[0], dim=3, num_links=2, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG, buffer_key="SAMPLING"
        )
        tt_out_rm = ttnn.untilize(tt_out_gathered, use_multicore=True, sub_core_grids=sub_core_grids)
        tt_out_tok = ttnn.argmax(  # FIXME When ttnn.argmax supports multicore, avoid falling back to host
            tt_out_rm, dim=3, use_multicore=True, output_tensor=tt_out_tok, sub_core_grids=sub_core_grids
        )
        logger.info(f"sampling done")

    if not stress_test:
        ttnn.plus_one(
            current_pos_tensor,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        )
    # profiler.end(f"plus one position done")

    # Capture Trace
    logger.info(f"Capturing model trace...")
    profiler.start(f"capture_trace")

    tt_model.tt_ccl.reset_gather_and_buffer_idx()

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)

    # Get cos/sin matrices for the current position of each user
    rot_mats = tt_model.rope_setup.get_rot_mats(rot_mat_idxs)
    tt_decode_input = tt_embd(tt_out_tok)
    tt_out = tt_model(
        tt_decode_input,
        current_pos_tensor,
        rot_mats=rot_mats,
        mode="decode",
        page_table=page_table_tt,
    )

    # Note: Persistent output buffer used, do not deallocate output!
    tt_out_gathered = tt_model.tt_ccl.line_all_gather(
        tt_out[0], dim=3, num_links=2, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG, buffer_key="SAMPLING"
    )
    tt_out_rm = ttnn.untilize(tt_out_gathered, use_multicore=True, sub_core_grids=sub_core_grids)
    tt_out_tok = ttnn.argmax(  # FIXME When ttnn.argmax supports multicore, avoid falling back to host
        tt_out_rm, dim=3, use_multicore=True, output_tensor=tt_out_tok, sub_core_grids=sub_core_grids
    )

    if not stress_test:
        ttnn.plus_one(
            current_pos_tensor,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        )
    # ttnn.plus_one(rot_mat_idxs)  # FIXME <- This won't work since embedding requires uint32 and plus_one only works for int32

    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Reset the decoding position for the proper run of the model
    current_pos_reset = ttnn.from_torch(
        current_pos,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    tt_out_tok_reset = ttnn.from_torch(
        encoded_prompts_tensor_whole_sequence[:, :1].reshape(1, 1, 1, batch_size),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
    )

    # Reset the current position and output token tensors for the real decode run
    ttnn.copy_host_to_device_tensor(current_pos_reset, current_pos_tensor)
    ttnn.copy_host_to_device_tensor(tt_out_tok_reset, tt_out_tok)
    rot_mat_idxs_reset = tt_model.rope_setup.get_rot_idxs(current_pos, on_host=True)
    ttnn.copy_host_to_device_tensor(rot_mat_idxs_reset, rot_mat_idxs)

    profiler.end(f"capture_trace")

    ttnn.synchronize_device(mesh_device)

    # Start decoding
    iteration = 0
    users_decoding = True  # reset to handle next batch
    total_decoding_time = 0  # Track total decoding time
    total_tokens_generated = 0  # Track total tokens generated

    all_outputs = []

    logger.info(f"Starting decode loop...")
    profiler.start(f"inference_decode", iteration=iteration)

    while users_decoding:
        if iteration == 0:  # First iteration also accounts for compile time
            profiler.start(f"compile_decode", iteration=iteration)
        iteration_time_start = time()

        # Execute trace
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        tt_out_tok_cpu = tt_out_tok.cpu(blocking=True, cq_id=0)
        iteration_time = time() - iteration_time_start

        # Update current pos and mat idxs on host and send to device
        # TODO This is required for now since we cannot ttnn.plus_one(rot_mat_idxs) while it being uint32.
        # If this tensor is int32, it won't be supported by ttnn.embedding
        if not stress_test:
            current_pos += 1
        rot_mat_idxs_updated = tt_model.rope_setup.get_rot_idxs(current_pos, on_host=True)
        ttnn.copy_host_to_device_tensor(rot_mat_idxs_updated, rot_mat_idxs)
        # ttnn.synchronize_device(mesh_device)
        # Write to host
        tt_output_torch = ttnn.to_torch(
            tt_out_tok_cpu,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device,
                dims=(3, 1),
                mesh_shape=model_args.cluster_shape,
            ),
        )[0, 0, 0, :batch_size]
        # Append the generated token to the list of outputs
        if iteration in range(len(encoded_prompts[0])):
            all_outputs.append(encoded_prompts[0][iteration])  # Update list of TT outputs
            tt_out_tok_reset = ttnn.from_torch(
                encoded_prompts_tensor_whole_sequence[:, iteration].reshape(1, 1, 1, batch_size),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
            )
            ttnn.copy_host_to_device_tensor(tt_out_tok_reset, tt_out_tok)
        else:
            all_outputs.append(tt_output_torch.tolist()[0])  # Update generated token to list of TT outputs
            if all_outputs[-1] in [128001, 128009] and not stress_test:  # EoT tokens
                users_decoding = False

        # Ignore the first iteration for average speed calculation
        if iteration > 0:
            total_decoding_time += iteration_time
            total_tokens_generated += 1

        tokens_per_second_per_user = 1 / iteration_time

        profiler.start(f"log_printing_iter_{iteration}", iteration=iteration)
        # Print out generated outputs for each user at the end of every iteration
        if not is_ci_env:
            # if len(user_input) == 1:
            logger.info("[User 0] {}".format("".join(tokenizer.decode(all_outputs))))
            # else:
            #     for user in range(batch_size):
            #         text = "".join(tokenizer.decode(all_outputs[user]))
            #         if len(text) > 100:
            #             text = "..." + text[-97:]
            #         text = text.replace("\n", " ")
            #         logger.info("[User {}] {}".format(user, text))

        # Always print perf at every iteration
        logger.info(
            f"Iteration {iteration}: {1000*iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
        )
        tsu_threshold = 128 if layers == 1 else 28
        if not stress_test:
            assert tokens_per_second_per_user > tsu_threshold, "Throughput is less than 28 tokens per second per user"
        profiler.end(f"log_printing_iter_{iteration}", iteration=iteration)

        if iteration == 0:  # First iteration also accounts for compile time
            profiler.end(f"compile_decode", iteration=iteration)

        iteration += 1

        # Upper limit of generated tokens for each user (to avoid infinite generation in case eos is not seen)
        if iteration >= max_generated_tokens:
            users_decoding = False

    # Release trace
    ttnn.release_trace(mesh_device, trace_id)

    # Finish profiling at the end of all batches inference
    profiler.end("run")


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
    "weights, layers, input_prompts, instruct, repeat_batches, max_seq_len, batch_size, max_generated_tokens, paged_attention, page_params, sampling_params, stress_test, start_pos",
    [
        (  # full demo, batch 32
            "instruct",
            80,
            "models/demos/llama3_subdevices/demo/input_data_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            32,  # batch_size
            200,  # max_generated_tokens
            False,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks": 1024},  # page_params  # TODO This will be serviced by vLLM
            {"top_k": 32, "top_p": 0.08, "seed": 42},  # sampling_params (argmax)
            False,  # stress_test
            0,  # start_pos
        ),
        (  # quick 1L demo
            "random",
            1,
            "models/demos/llama3_subdevices/demo/input_data_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            32,  # batch_size
            200,  # max_generated_tokens
            False,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks": 1024},  # page_params  # TODO This will be serviced by vLLM
            {"top_k": 32, "top_p": 0.08, "seed": 42},  # sampling_params (argmax)
            False,  # stress_test
            0,  # start_pos
        ),
        (  # Stress test: batch-32 very long generations but at same token index
            "instruct",
            80,
            "models/demos/llama3_subdevices/demo/input_data_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            32,  # batch_size
            4 * 128 * 1024,  # max_generated_tokens (same index for stress test)
            False,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks": 1024},  # page_params  # TODO This will be serviced by vLLM
            {"top_k": 32, "top_p": 0.08, "seed": 42},  # sampling_params (argmax)
            True,  # stress_test
            0,  # start_pos
        ),
        (  # 10 layers for devive perf measurements
            "random",
            10,
            "models/demos/llama3_subdevices/demo/input_data_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            32,  # batch_size
            1,  # max_generated_tokens
            False,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks": 1024},  # page_params  # TODO This will be serviced by vLLM
            {"top_k": 32, "top_p": 0.08, "seed": 42},  # sampling_params (argmax)
            False,  # stress_test
            127,  # start_pos
        ),
    ],
    ids=[
        "full",  # full demo
        "quick",  # 1L demo
        "stress-test",  # stress test with many iterations and same token index, full model
        "measure-device-perf",  # 10L demo for device performance measurements
    ],
)
@pytest.mark.parametrize(
    "optimizations",
    [
        LlamaOptimizations.performance,
        # LlamaOptimizations.accuracy,
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}], indirect=True
)
def test_llama_demo(
    weights,
    layers,
    input_prompts,
    instruct,
    repeat_batches,
    max_seq_len,
    batch_size,
    max_generated_tokens,
    paged_attention,
    page_params,
    sampling_params,
    stress_test,
    start_pos,
    optimizations,
    mesh_device,
    use_program_cache,
    is_ci_env,
    reset_seeds,
):
    if is_ci_env and ("long" in input_prompts or optimizations == LlamaOptimizations.accuracy):
        pytest.skip("Do not run the 'long-context' or accuracy tests on CI to reduce load")

    assert_msg = (
        "Ring buffer size not set. Needs to be set env TT_METAL_WORKER_RINGBUFFER_SIZE=122880 (120KB) for best perf"
    )
    assert os.environ.get("TT_METAL_WORKER_RINGBUFFER_SIZE") is not None, assert_msg

    # TODO: Remove this once all batch sizes are supported on TG
    if os.environ.get("FAKE_DEVICE") == "TG" and batch_size not in [1, 32]:
        pytest.skip("TG only supports batch 1 and 32")

    mesh_device.enable_async(True)

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
    else:
        paged_attention_config = None

    return run_llama3_demo(
        user_input=input_prompts,
        mesh_device=mesh_device,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_batches=repeat_batches,
        paged_attention=paged_attention,
        paged_attention_config=paged_attention_config,
        max_generated_tokens=max_generated_tokens,
        optimizations=optimizations,
        sampling_params=sampling_params,
        instruct_mode=instruct,
        is_ci_env=is_ci_env,
        print_to_file=False,
        weights=weights,
        layers=layers,
        stress_test=stress_test,
        start_pos=start_pos,
    )
