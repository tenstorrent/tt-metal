# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS demo using tt_transformers generation pipeline

Integrates GPT-OSS with tt_transformers infrastructure for:
- Paged attention support
- Sophisticated generation loop with sampling
- Performance profiling and benchmarking
- Multi-user batch generation capability

Updated to use refactored TestFactory and MeshConfig patterns:
- Uses parametrize_mesh_with_fabric() for consistent mesh setup
- Uses TestFactory.setup_test() for unified configuration
- Passes mesh_config to create_tt_model for proper sharding
"""

import json
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.gpt_oss.tests.test_factory import TestFactory, parametrize_mesh_with_fabric

# Import GPT-OSS components using our refactored patterns
from models.demos.gpt_oss.tt.common import create_tt_model
from models.demos.utils.llm_demo_utils import create_benchmark_data, verify_perf
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.demo.simple_text_demo import create_tt_page_table, load_inputs
from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    get_padded_prefill_len,
    preprocess_inputs_prefill,
    sample_host,
)

# Import specific utilities from tt_transformers
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.model_config import determine_device_name


class GPTOSSGenerator(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prefill_forward_text(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        model_id_warmup=None,
        **kwargs,
    ):
        if page_table is not None:
            assert isinstance(page_table, torch.Tensor), "page_table mush be torch.Tensor"
        else:
            # Only paged attention is supported for prefill
            enable_trace = False

        # self.warmup_prefill_traces(
        #     page_table,
        #     kv_cache,
        #     enable_trace,
        # )

        batch_size, batch_seq_len = tokens.shape
        max_batch_size_per_model = self.model_args[0].max_batch_size
        max_batch_per_mesh_row = max_batch_size_per_model // self.mesh_device.shape[0]

        # Each model expected to run the same model, safe to use 1st vocab size
        output_logits = torch.zeros(batch_size, 1, self.model_args[0].vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch_size)

        if empty_slots is None:
            empty_slots = list(range(batch_size))

        out_list = []
        for idx, user_id in enumerate(empty_slots):
            # if model_id is not None, it means that prefill is called from warmup_prefill_traces
            model_id = user_id // max_batch_size_per_model if model_id_warmup is None else model_id_warmup
            group_user_id = user_id % max_batch_size_per_model if page_table is None else 0
            seq_len = int(prompt_lens[idx])
            last_token_idx = seq_len - 1
            prefill_seq_len = get_padded_prefill_len(seq_len)
            local_kwargs = kwargs.copy()  # Avoid modifying original kwargs

            logger.info(f"Prefilling User {user_id + 1} up to {seq_len} tokens")

            # Extracting data for the current user
            # If page_table is not provided, we keep track of the relative/model user_id through group_user_id
            prefill_ids = torch.cat(
                [tokens[idx : idx + 1, :seq_len], torch.zeros(1, prefill_seq_len - seq_len).long()], dim=-1
            )

            enable_trace_current_prompt = enable_trace and self.model_args[model_id].can_enable_trace(prefill_seq_len)

            logger.info(
                f"Prefill seq len: {prefill_seq_len}, max_prefill_chunk_size: {self.model_args[0].max_prefill_chunk_size}, trace: {enable_trace_current_prompt}"
            )

            page_table_user = (
                self._get_prefill_user_page_table(
                    page_table,
                    kv_cache[model_id],
                    seq_len,
                    trace_enabled=enable_trace_current_prompt,
                    prefill_seq_len=prefill_seq_len,
                )
                if page_table is not None
                else None
            )
            new_page_table_user = -1 * torch.ones_like(page_table_user)
            new_page_table_user[user_id] = page_table_user[user_id]

            model_kv_cache = kv_cache[model_id] if kv_cache is not None else None

            # Check if 'pixel_values' exists and index it safely
            if local_kwargs.get("pixel_values", None) is not None:
                local_kwargs["pixel_values"] = local_kwargs["pixel_values"][idx]
                if "image_grid_thw" in local_kwargs:
                    local_kwargs["image_grid_thw"] = local_kwargs["image_grid_thw"][idx]

            if enable_trace_current_prompt:
                logits = self._easy_trace_prefill(
                    prefill_ids,
                    page_table=new_page_table_user,
                    user_id=user_id % max_batch_per_mesh_row,
                    last_token_idx=last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    prefill_seq_len=prefill_seq_len,
                    **local_kwargs,
                )
            else:
                logits = self.prefill_forward_single_user_text(
                    prefill_ids,
                    page_table=new_page_table_user,
                    user_id=user_id % max_batch_per_mesh_row,
                    last_token_idx=last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    **local_kwargs,
                )
            if enable_trace_current_prompt:
                # Slicing the tensor to the nearest ceiling/floor multiples of 32 for the prefill_len, to get the last token
                # We need to do this here, because we can't do this part in forward() if we have trace enabled
                # The reason we can't do it in trace is because we can't pass the correct get_last_token to trace
                logits = self.model[model_id].process_logits_after_prefill_trace(logits, last_token_idx)

            # if data parallel is greater than 1, we need to add logits to out_list and do the processing after all the prefill are done
            # otherwise, we can process the logits after prefill immediately
            if self.data_parallel > 1:
                out_list.append(logits)
            else:
                output_logits[idx] = self.model[model_id].process_output_prefill(
                    logits, last_token_idx=(last_token_idx % 32)
                )
                del logits

        # Process the logits after all the prefill are done in data parallel mode
        if self.data_parallel > 1:
            for idx, out in enumerate(out_list):
                seq_len = int(prompt_lens[idx])
                last_token_idx = seq_len - 1
                user_id = empty_slots[idx]
                model_id = user_id // max_batch_size_per_model if model_id_warmup is None else model_id_warmup

                # Since we give unpadded_seq_len, only the tile containing the last token is returned
                output_logits[idx] = self.model[model_id].process_output_prefill(
                    out, last_token_idx=(last_token_idx % 32)
                )

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")
        return output_logits


def prepare_gpt_oss_generator_args(
    num_devices,
    data_parallel,
    mesh_device,
    global_batch_size,
    optimizations,
    max_seq_len,
    page_params,
    paged_attention,
    mesh_config=None,
    state_dict=None,
):
    """Prepare generator args using GPT-OSS create_tt_model (clean version)"""
    submesh_devices = create_submeshes(mesh_device, data_parallel)

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
        # Use GPT-OSS create_tt_model directly!
        model_args_i, model_i, tt_kv_cache_i, state_dict = create_tt_model(
            submesh,
            max_batch_size=global_batch_size // data_parallel,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
            mesh_config=mesh_config,  # Pass mesh config for proper sharding
        )
        model_args.append(model_args_i)
        model.append(model_i)
        tt_kv_cache.append(tt_kv_cache_i)

    # Page table will be created using tt-transformers infrastructure after input preprocessing
    page_tables = (
        [
            create_tt_page_table(
                global_batch_size // mesh_device.shape[0],
                data_parallel,
                paged_attention_config,
            )
            for _ in range(mesh_device.shape[0])
        ]
        if paged_attention
        else None
    )
    page_table = torch.concat(page_tables, dim=0) if page_tables else None

    # Host code, safe to reuse tokenizer from the 1st model
    tokenizer = model_args[0].tokenizer
    processor = model_args[0].processor
    return model_args, model, page_table, tt_kv_cache, tokenizer, processor, paged_attention_config


@pytest.mark.parametrize(
    "mesh_shape",
    [
        # LoudBox (1×8) - Single device, low latency
        (1, 8),
        # Galaxy (4×8) - Multi-device mesh, higher throughput
        (4, 8),
    ],
    ids=["mesh_1x8", "mesh_4x8"],
)
@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "input_prompts, data_parallel, batch_size, repeat_batches, max_seq_len, max_generated_tokens, page_params, sampling_params, enable_decode_trace, enable_prefill_trace",
    [
        (
            "models/demos/gpt_oss/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            1,  # data_parallel
            1,  # batch_size
            1,  # repeat_batches
            4 * 1024,  # max_seq_len
            200,  # max_generated_tokens
            {"page_block_size": 64, "page_max_num_blocks_per_dp": 4 * 1024 // 64},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (greedy decoding),
            True,  # enable_decode_trace
            True,  # enable_prefill_trace
        ),
        (
            "models/tt_transformers/demo/sample_prompts/input_data_long_1k.json",  # input_prompts
            1,  # data_parallel
            1,  # batch_size
            1,  # repeat_batches
            4 * 1024,  # max_seq_len
            200,  # max_generated_tokens
            {"page_block_size": 64, "page_max_num_blocks_per_dp": 4 * 1024 // 64},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (greedy decoding)
            True,  # enable_decode_trace
            True,  # enable_prefill_trace
        ),
        (
            "models/tt_transformers/demo/sample_prompts/input_data_long_4k.json",  # input_prompts
            1,  # data_parallel
            1,  # batch_size
            1,  # repeat_batches
            4 * 1024,  # max_seq_len
            200,  # max_generated_tokens
            {"page_block_size": 64, "page_max_num_blocks_per_dp": 4 * 1024 // 64},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (greedy decoding)
            True,  # enable_decode_trace
            True,  # enable_prefill_trace
        ),
        (
            "models/demos/gpt_oss/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            1,  # data_parallel
            128,  # batch_size
            1,  # repeat_batches
            8 * 1024,  # max_seq_len
            200,  # max_generated_tokens
            {"page_block_size": 64, "page_max_num_blocks_per_dp": 128 * 1024 // 64},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (greedy decoding)
            True,
            False,
        ),
        # (
        #     "models/tt_transformers/demo/sample_prompts/input_data_long_8k.json",  # input_prompts
        #     1,  # data_parallel
        #     1,  # batch_size
        #     1,  # repeat_batches
        #     8 * 1024,  # max_seq_len
        #     200,  # max_generated_tokens
        #     {"page_block_size": 64, "page_max_num_blocks_per_dp": 4 * 1024 // 64},  # page_params
        #     {"temperature": 0, "top_p": 0.08},  # sampling_params (greedy decoding)
        # ),
        # (
        #     "models/tt_transformers/demo/sample_prompts/input_data_long_16k.json",  # input_prompts
        #     1,  # data_parallel
        #     1,  # batch_size
        #     1,  # repeat_batches
        #     16 * 1024,  # max_seq_len
        #     200,  # max_generated_tokens
        #     {"page_block_size": 64, "page_max_num_blocks_per_dp": 4 * 1024 // 64},  # page_params
        #     {"temperature": 0, "top_p": 0.08},  # sampling_params (greedy decoding)
        # ),
        # (
        #     "models/tt_transformers/demo/sample_prompts/input_data_long_32k.json",  # input_prompts
        #     1,  # data_parallel
        #     1,  # batch_size
        #     1,  # repeat_batches
        #     32 * 1024,  # max_seq_len
        #     200,  # max_generated_tokens
        #     {"page_block_size": 64, "page_max_num_blocks_per_dp": 4 * 1024 // 64},  # page_params
        #     {"temperature": 0, "top_p": 0.08},  # sampling_params (greedy decoding)
        # ),
        # (
        #     "models/tt_transformers/demo/sample_prompts/input_data_long_64k.json",  # input_prompts
        #     1,  # data_parallel
        #     1,  # batch_size
        #     1,  # repeat_batches
        #     64 * 1024,  # max_seq_len
        #     200,  # max_generated_tokens
        #     {"page_block_size": 64, "page_max_num_blocks_per_dp": 4 * 1024 // 64},  # page_params
        #     {"temperature": 0, "top_p": 0.08},  # sampling_params (greedy decoding)
        # ),
        # (
        #     "models/tt_transformers/demo/sample_prompts/input_data_long_128k.json",  # input_prompts
        #     1,  # data_parallel
        #     1,  # batch_size
        #     1,  # repeat_batches
        #     128 * 1024,  # max_seq_len
        #     200,  # max_generated_tokens
        #     {"page_block_size": 64, "page_max_num_blocks_per_dp": 4 * 1024 // 64},  # page_params
        #     {"temperature": 0, "top_p": 0.08},  # sampling_params (greedy decoding)
        # ),
    ],
    ids=[
        "prefill_128",
        "prefill_1k",
        "prefill_4k",
        "batch128"
        # "prefill_8k",
        # "prefill_16k",
        # "prefill_32k",
        # "prefill_64k",
        # "prefill_128k",
    ],
)
@parametrize_mesh_with_fabric()
def test_gpt_oss_demo(
    mesh_device,
    device_params,
    mesh_shape,
    input_prompts,
    data_parallel,
    batch_size,
    repeat_batches,
    max_seq_len,
    max_generated_tokens,
    page_params,
    sampling_params,
    enable_decode_trace,
    enable_prefill_trace,
    is_ci_env,
    state_dict,
):
    """GPT-OSS demo using full tt_transformers generation pipeline"""
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(mesh_shape))

    # Use our refactored TestFactory for consistent setup
    setup = TestFactory.setup_test(mesh_device, use_real_weights=True)
    config = setup["config"]
    mesh_config = setup["mesh_config"]

    logger.info(f"Using mesh config: {mesh_config}, model config: {config}")

    # Configuration matching tt_transformers defaults
    num_devices = mesh_device.get_num_devices()
    paged_attention = True  # Always use paged attention for GPT-OSS
    global_batch_size = batch_size * data_parallel  # Total batch across all devices

    # Validate data parallel configuration (like tt-transformers)
    if data_parallel > num_devices or num_devices % data_parallel != 0:
        raise ValueError(f"Invalid number of DP groups: {data_parallel}, for {num_devices} devices")

    logger.info(f"Running GPT-OSS demo with tt_transformers generation pipeline")

    # Setup profiler like tt_transformers
    profiler = BenchmarkProfiler()
    profiler.start("run")
    batch_idx = 0

    # GPT-OSS doesn't support any performance optimizations
    optimizations = None

    # Prepare GPT-OSS with tt_transformers infrastructure
    profiler.start(f"generator_setup", iteration=batch_idx)
    (
        model_args,
        model,
        page_table,
        tt_kv_cache,
        tokenizer,
        processor,
        paged_attention_config,
    ) = prepare_gpt_oss_generator_args(
        num_devices=num_devices,
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        global_batch_size=global_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=paged_attention,
        mesh_config=mesh_config,  # Pass our refactored mesh config
        state_dict=state_dict,
    )

    # Create generator (match tt-transformers pattern)
    # generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)
    generator = GPTOSSGenerator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    profiler.end(f"generator_setup", iteration=batch_idx)

    # Prepare input prompts
    logger.info(f"Reading inputs...")
    profiler.start("loading_inputs")
    if isinstance(input_prompts, list) and len(input_prompts) == 1:  # Manual input
        input_prompts = input_prompts * global_batch_size
    elif isinstance(input_prompts, str):  # Inputs from file
        input_prompts, _ = load_inputs(input_prompts, global_batch_size, instruct=False)
    else:
        raise ValueError(
            f"Invalid input prompts: {input_prompts}. Expected a list of prompts or a string path to a json file."
        )
    profiler.end("loading_inputs")

    # Create repeat batches (like tt-transformers)
    repeat_batch_prompts = []
    for i in range(repeat_batches):
        repeat_batch_prompts.append([input_prompts[(j + i) % len(input_prompts)] for j in range(len(input_prompts))])

    num_tokens_generated_decode = []

    logger.info("Starting inference...")
    logger.info(f"Page table: {page_table}")

    # Main inference loop for repeat batches (like tt-transformers)
    for batch_idx, input_prompts_batch in enumerate(repeat_batch_prompts):
        logger.info(f"Processing batch {batch_idx}")

        # Preprocess inputs (reusing tt_transformers function)
        profiler.start(f"preprocess_prefill_inputs", iteration=batch_idx)
        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            input_prompts_batch,
            tokenizer,
            model_args,
            instruct=False,
            max_generated_tokens=max_generated_tokens,
            max_prefill_len=max_seq_len,
        )

        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)
        profiler.end(f"preprocess_prefill_inputs", iteration=batch_idx)

        logger.info(f"Input prompt: {input_prompts_batch[0]}")
        logger.info(f"Encoded length: {prefill_lens[0]} tokens")

        # Clear KV caches for repeat batches (like tt-transformers)
        if batch_idx != 0:
            # Fix for ND hangs with multiple repeat batches
            generator.prev_page_table = None

            for i in range(len(model)):
                for layer in model[i].layers:
                    k_cache, v_cache = layer.self_attn.layer_past
                    k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
                    v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)

        # Prefill phase (matching tt_transformers)
        logger.info("Starting prefill warmup...")
        profiler.start(f"compile_prefill", iteration=batch_idx)
        logits = generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
            enable_trace=enable_prefill_trace,
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
            enable_trace=enable_prefill_trace,
        )
        prefilled_token = torch.argmax(logits, dim=-1)
        profiler.end(f"inference_prefill", iteration=batch_idx)
        logger.info(f"Prefill finished")
        logger.info(f"First generated token: '{tokenizer.decode(prefilled_token[0])}'")

        # Initialize generation state like tt_transformers
        all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(global_batch_size)]
        for user in range(global_batch_size):
            user_tok = int(prefilled_token[user].item())
            all_outputs[user].append(user_tok)

        user_done = [False] * global_batch_size
        current_pos = torch.tensor([decoding_pos[b] for b in range(global_batch_size)])
        out_tok = prefilled_token

        # Generation loop (matching tt_transformers structure)
        logger.info(f"Starting decode loop...")
        iteration = 0
        users_decoding = True

        profiler.start(f"inference_decode", iteration=batch_idx)
        while users_decoding and iteration < max_generated_tokens:
            if iteration == 0:
                profiler.start(f"compile_decode", iteration=batch_idx)
            else:
                profiler.start(f"inference_decode_time_{iteration}", iteration=batch_idx)

            # Decode forward (matching tt_transformers call)
            logits, _ = generator.decode_forward_text(
                out_tok,
                current_pos,
                enable_trace=enable_decode_trace,
                page_table=page_table,
                kv_cache=tt_kv_cache,
            )

            # Sample next token (reusing tt_transformers sampling)
            _, out_tok = sample_host(
                logits,
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                on_host=True,
            )

            if iteration == 0:
                profiler.end(f"compile_decode", iteration=batch_idx)
                decode_iteration_time = profiler.get_duration("compile_decode", iteration=batch_idx)
            else:
                profiler.end(f"inference_decode_time_{iteration}", iteration=batch_idx)
                decode_iteration_time = profiler.get_duration(f"inference_decode_time_{iteration}", iteration=batch_idx)

            # Print perf after every iteration
            tokens_per_second_per_user = 1 / decode_iteration_time
            logger.debug(
                f"Iteration {iteration}: {1000*decode_iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({global_batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
            )

            current_pos += 1

            # Save output token
            for user in range(global_batch_size):
                user_tok = out_tok[user].item()
                if user_tok != tokenizer.eos_token_id and user_done[user] == False:
                    all_outputs[user].append(user_tok)
                else:
                    user_done[user] = True
                    logger.debug(f"User {user} finished decoding at iteration {iteration}")
                    if all(user_done):
                        users_decoding = False

            iteration += 1

        profiler.end(f"inference_decode", iteration=batch_idx)

        # Final output for this batch (like tt_transformers)
        logger.info("Finished decoding, printing the final outputs...\n")
        for i, (output, prompt) in enumerate(zip(all_outputs, input_prompts_batch)):
            text = tokenizer.decode(output)
            prompt_including_assistant_tags = tokenizer.decode(model_args[0].encode_prompt(prompt))
            text_after_prompt = text.replace(prompt_including_assistant_tags, "", 1)
            short_prompt = (
                (prompt[:100] + "\n<long prompt not printed in full>\n" + prompt[-100:])
                if len(prompt) > 200
                else prompt
            )
            logger.info(
                f"\n==REPEAT BATCH {batch_idx}\n==USER {i} - PROMPT\n{short_prompt} \n==USER {i} - OUTPUT\n{text_after_prompt.strip()}\n"
            )

        num_tokens_generated_decode.append(iteration)  # Save the number of tokens generated for each repeat batch

    # Performance metrics calculation (like tt-transformers)
    profiler.end("run")

    # Calculate performance metrics for the first batch only
    compile_prefill_time = profiler.get_duration("compile_prefill")
    compile_decode_time = profiler.get_duration("compile_decode")

    total_inference_prefill_time = profiler.get_duration("inference_prefill")
    total_inference_decode_time = 0
    for i in range(1, num_tokens_generated_decode[0]):  # Iteration 0 is the compile time
        total_inference_decode_time += profiler.get_duration(f"inference_decode_time_{i}")

    # Calculate TTFT and t/s/u metrics (like tt-transformers)
    avg_time_to_first_token = total_inference_prefill_time / global_batch_size  # TTFT per user
    avg_decode_iteration_time = total_inference_decode_time / (num_tokens_generated_decode[0] - 1)

    prefill_tok_s = prefill_lens[0] / total_inference_prefill_time * global_batch_size
    decode_tok_s_user = (num_tokens_generated_decode[0] - 1) / total_inference_decode_time  # t/s/u
    decode_tok_s = (num_tokens_generated_decode[0] - 1) / total_inference_decode_time * global_batch_size  # total t/s

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

    # Performance logging (like tt-transformers)
    logger.info("")
    logger.info(f"=== Performance metrics ===")
    logger.info(f"Prefill compile time: {round(compile_prefill_time, 2)}s")
    logger.info(f"Decode compile time: {round(compile_decode_time, 2)}s")
    logger.info("")
    logger.info(f"Average Time to First Token (TTFT): {round(avg_time_to_first_token * 1000, 2)}ms")
    logger.info(
        f"Average decode speed: {round(avg_decode_iteration_time * 1000, 2)}ms @ {round(decode_tok_s_user, 2)} tok/s/user ({round(decode_tok_s, 2)} tok/s throughput)"
    )
    logger.info(f"Data parallel: {data_parallel}, Global batch size: {global_batch_size}")

    logger.info("GPT-OSS demo completed successfully!")

    if is_ci_env:
        tt_device_name = determine_device_name(mesh_device)  # submesh device should not decide performance target
        tt_device_name = "GLX" if tt_device_name == "TG" else tt_device_name  # TG is old nomenclature of 4U galaxy.
        model_name = model_args[0].model_name
        model_device_key = f"{tt_device_name}_{model_name}"

        with open(Path(__file__).parent.parent.joinpath("perf_targets.json"), "r") as f:
            perf_targets = json.load(f)
        prefill_pad_length = 1 << max(prefill_lens).bit_length()  # round up to the next power of 2
        if (
            f"prefill_{prefill_pad_length}" in perf_targets["targets"]
            and model_device_key in perf_targets["targets"][f"prefill_{prefill_pad_length}"]
        ):
            targets = {
                "prefill_t/s": perf_targets["targets"][f"prefill_{prefill_pad_length}"][model_device_key]["TTFT"],
                "decode_t/s": perf_targets["targets"][f"prefill_{prefill_pad_length}"][model_device_key][
                    "decode_tok_s"
                ],
                "decode_t/s/u": perf_targets["targets"][f"prefill_{prefill_pad_length}"][model_device_key][
                    "decode_tok_s_u"
                ],
            }
        else:
            targets = {}
        # Instead of running warmup iterations, the demo profiles the initial compile iteration
        bench_n_warmup_iter = {"inference_prefill": 0, "inference_decode": 1}
        benchmark_data = create_benchmark_data(profiler, measurements, bench_n_warmup_iter, targets)

        # Save the decode performance of every iteration for plotting in superset
        for i in range(1, num_tokens_generated_decode[0]):
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
        num_iterations_for_avg = min(128, num_tokens_generated_decode[0])
        inference_decode_time_first_128 = sum(
            profiler.get_duration(f"inference_decode_time_{i}") for i in range(1, num_iterations_for_avg)
        )
        benchmark_data.add_measurement(
            profiler,
            0,
            "inference_decode",
            "avg_decode_time_first_128",
            inference_decode_time_first_128 * 1000 / max(1, num_iterations_for_avg - 1),
            step_warm_up_num_iterations=None,
            target=None,
        )
        benchmark_data.save_partial_run_json(
            profiler,
            run_type=f"{tt_device_name}-demo",
            ml_model_name=model_name,
            ml_model_type="llm",
            num_layers=model_args[0].n_layers,
            batch_size=global_batch_size,
            config_params={"data_parallel": data_parallel, "tensor_parallel": num_devices // data_parallel},
            input_sequence_length=max(prefill_lens),
            output_sequence_length=num_tokens_generated_decode[0],
        )

        # check measurements against CI performance targets
        logger.info(
            f"Checking measurements against CI performance targets for {model_name} on {tt_device_name} for padded prefill length {prefill_pad_length}"
        )
        # Only call verify_perf if the model_device_key exists in the targets
        if f"prefill_{prefill_pad_length}" in perf_targets["ci"]:
            if model_device_key in perf_targets["ci"][f"prefill_{prefill_pad_length}"]:
                current_ttft_target = perf_targets["ci"][f"prefill_{prefill_pad_length}"][model_device_key]["TTFT"]
                if isinstance(current_ttft_target, list):
                    high_tol_percentage = current_ttft_target[1]
                    current_ttft_target = current_ttft_target[0]
                else:
                    high_tol_percentage = 1.15
                ci_targets = {
                    "prefill_time_to_token": current_ttft_target / 1000,  # convert to seconds
                    "decode_t/s/u": perf_targets["ci"][f"prefill_{prefill_pad_length}"][model_device_key][
                        "decode_tok_s_u"
                    ],
                    "decode_t/s": perf_targets["ci"][f"prefill_{prefill_pad_length}"][model_device_key][
                        "decode_tok_s_u"
                    ]
                    * global_batch_size,  # calculate from per-user rate
                }
                verify_perf(
                    measurements,
                    ci_targets,
                    high_tol_percentage=high_tol_percentage,
                    expected_measurements={k: True for k in ci_targets.keys()},
                )
            else:
                logger.warning(
                    f"No CI performance targets found for model {model_name} on device {tt_device_name} for prefill length {prefill_pad_length}. Skipping performance verification."
                )
        else:
            logger.warning(
                f"No CI performance targets found for prefill length {prefill_pad_length}. Skipping performance verification."
            )
