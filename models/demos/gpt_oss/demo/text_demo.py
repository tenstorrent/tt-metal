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

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.gpt_oss.tests.test_factory import TestFactory, parametrize_mesh_with_fabric

# Import GPT-OSS components using our refactored patterns
from models.demos.gpt_oss.tt.common import create_tt_model
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.demo.simple_text_demo import create_tt_page_table
from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill, sample_host

# Import specific utilities from tt_transformers
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.model_config import DecodersPrecision


def prepare_gpt_oss_generator_args(
    num_devices,
    data_parallel,
    mesh_device,
    instruct,
    global_batch_size,
    optimizations,
    max_seq_len,
    page_params,
    paged_attention,
    mesh_config=None,
):
    """Prepare generator args using GPT-OSS create_tt_model (clean version)"""
    submesh_devices = create_submeshes(mesh_device, data_parallel)
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
        # Use GPT-OSS create_tt_model directly!
        model_args_i, model_i, tt_kv_cache_i, state_dict = create_tt_model(
            submesh,
            instruct=instruct,
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
    page_table = (
        create_tt_page_table(
            global_batch_size,
            data_parallel,
            paged_attention_config,
        )
        if paged_attention
        else None
    )

    # Host code, safe to reuse tokenizer from the 1st model
    tokenizer = model_args[0].tokenizer
    processor = model_args[0].processor
    return model_args, model, page_table, tt_kv_cache, tokenizer, processor, paged_attention_config


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "mesh_shape, data_parallel, batch_size, repeat_batches, max_seq_len, max_generated_tokens, instruct, page_params, sampling_params",
    [
        (  # LoudBox (1×8) - Single device, low latency
            (1, 8),  # mesh_shape
            1,  # data_parallel
            1,  # batch_size
            1,  # repeat_batches
            1024,  # max_seq_len
            200,  # max_generated_tokens
            True,  # instruct (set to False for base model, True for instruct model)
            {"page_block_size": 64, "page_max_num_blocks_per_dp": 1024 // 64},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (greedy decoding)
        ),
        (  # Galaxy (4×8) - Multi-device mesh, higher throughput
            (4, 8),  # mesh_shape
            1,  # data_parallel
            1,  # batch_size
            1,  # repeat_batches
            1024,  # max_seq_len
            200,  # max_generated_tokens
            True,  # instruct
            {"page_block_size": 64, "page_max_num_blocks_per_dp": 1024 // 64},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params
        ),
    ],
    ids=["mesh_1x8", "mesh_4x8"],
)
@parametrize_mesh_with_fabric()
def test_gpt_oss_demo(
    mesh_device,
    device_params,
    mesh_shape,
    data_parallel,
    batch_size,
    repeat_batches,
    max_seq_len,
    max_generated_tokens,
    instruct,
    page_params,
    sampling_params,
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

    enable_trace = False if mesh_config.ep > 1 else True  # ep > 1 currently has a fallback

    logger.info(f"Running GPT-OSS demo with tt_transformers generation pipeline")

    # Setup profiler like tt_transformers
    profiler = BenchmarkProfiler()
    profiler.start("run")
    batch_idx = 0

    # Use performance optimizations
    optimizations = lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)

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
        instruct=instruct,
        global_batch_size=global_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=paged_attention,
        mesh_config=mesh_config,  # Pass our refactored mesh config
    )

    # Create generator (match tt-transformers pattern)
    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    profiler.end(f"generator_setup", iteration=batch_idx)

    # Prepare input prompts like tt_transformers does
    input_prompts = ["What are the prime factors of 1?"]
    if len(input_prompts) == 1:  # Manual input - repeat for global batch size
        input_prompts = input_prompts * global_batch_size

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
            input_prompts_batch, tokenizer, model_args, instruct, max_generated_tokens, max_prefill_len=max_seq_len
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
            logits = generator.decode_forward_text(
                out_tok,
                current_pos,
                enable_trace=enable_trace,
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
            prompt_including_assistant_tags = tokenizer.decode(model_args[0].encode_prompt(prompt, instruct=instruct))
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
