# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek-V3 demo using tt_transformers generation pipeline

Integrates DeepSeek-V3 with tt_transformers infrastructure for:
- Paged attention support
- Sophisticated generation loop with sampling
- Performance profiling and benchmarking
- Multi-user batch generation capability
"""

import os
from pathlib import Path

import torch
from loguru import logger

import ttnn

# Import DeepSeek-V3 specific modules
from models.demos.deepseek_v3.tt.tt_transformers_model import create_deepseek_v3_tt_transformers_model

# from models.utility_functions import run_for_wormhole_b0
from models.demos.deepseek_v3.utils.config_helpers import MAX_BATCH_SIZE
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import preprocess_inputs_prefill, sample_host
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision


def prepare_deepseek_v3_generator_args(
    num_devices,
    data_parallel,
    mesh_device,
    instruct,
    global_batch_size,
    optimizations,
    max_seq_len,
    model_path,
    cache_dir,
):
    """Prepare generator args using DeepSeek-V3 with tt_transformers Generator"""

    # Use DeepSeek-V3 tt_transformers model
    model_args, model, tt_kv_cache = create_deepseek_v3_tt_transformers_model(
        mesh_device,
        max_batch_size=MAX_BATCH_SIZE,
        max_seq_len=max_seq_len,
        model_path=model_path,
        cache_dir=cache_dir,
    )

    # Host code, safe to reuse tokenizer from the 1st model
    tokenizer = model_args.tokenizer
    processor = model_args.processor
    return model_args, model, tt_kv_cache, tokenizer, processor


def _default_mesh_shape() -> ttnn.MeshShape:
    device_ids = ttnn.get_device_ids()
    if len(device_ids) == 32:
        return ttnn.MeshShape(4, 8)
    return ttnn.MeshShape(1, max(1, len(device_ids)))


def run_demo(input_prompts=["How many r's in the word 'strawberry'?"]):
    """DeepSeek-V3 demo using tt_transformers Generator"""
    mesh_shape = _default_mesh_shape()
    logger.info("Setting fabric config to FABRIC_1D for demo run")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    logger.info(f"Opening mesh device with shape {mesh_shape}")
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    try:
        # Configuration matching tt_transformers defaults
        num_devices = mesh_device.get_num_devices()

        # Data parallel configuration (can be adjusted for testing)
        data_parallel = 1  # Set to > 1 to test data parallel (e.g., 2, 4, 8)
        batch_size = MAX_BATCH_SIZE  # Batch size per data parallel group
        repeat_batches = 1  # Number of consecutive batches to run
        paged_attention = True
        global_batch_size = batch_size * data_parallel  # Total batch across all devices

        # Validate data parallel configuration (like tt-transformers)
        if data_parallel > num_devices or num_devices % data_parallel != 0:
            raise ValueError(f"Invalid number of DP groups: {data_parallel}, for {num_devices} devices")
        max_seq_len = 1024
        max_generated_tokens = 200  # Reasonable limit for testing
        instruct = True
        enable_trace = True  # Start with trace disabled

        # DeepSeek-V3 specific configuration
        model_path = Path(os.getenv("DEEPSEEK_V3_HF_MODEL", "models/demos/deepseek_v3/reference"))
        cache_dir = Path(os.getenv("DEEPSEEK_V3_CACHE", "generated/deepseek_v3"))

        page_params = {
            "page_block_size": 64,
            "page_max_num_blocks_per_dp": max_seq_len // 64,  # Total blocks available per data parallel unit
        }

        sampling_params = {
            "temperature": 0,  # Greedy decoding for deterministic results
            "top_p": 0.08,
        }

        logger.info(f"Running DeepSeek-V3 demo with tt_transformers generation pipeline")

        # Setup profiler like tt_transformers
        profiler = BenchmarkProfiler()
        profiler.start("run")
        batch_idx = 0

        # Use performance optimizations
        optimizations = lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)

        # Prepare DeepSeek-V3 with tt_transformers infrastructure
        profiler.start(f"generator_setup", iteration=batch_idx)
        (
            model_args,
            model,
            tt_kv_cache,
            tokenizer,
            processor,
        ) = prepare_deepseek_v3_generator_args(
            num_devices=num_devices,
            data_parallel=data_parallel,
            mesh_device=mesh_device,
            instruct=instruct,
            global_batch_size=global_batch_size,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            model_path=model_path,
            cache_dir=cache_dir,
        )
        page_table = None
        # Create generator (match tt-transformers pattern)
        generator = Generator([model], [model_args], mesh_device, processor=processor, tokenizer=tokenizer)

        profiler.end(f"generator_setup", iteration=batch_idx)

        # Prepare input prompts like tt_transformers does

        if len(input_prompts) == 1:  # Manual input - repeat for global batch size
            input_prompts = input_prompts * global_batch_size

        # Create repeat batches (like tt_transformers)
        repeat_batch_prompts = []
        for i in range(repeat_batches):
            repeat_batch_prompts.append(
                [input_prompts[(j + i) % len(input_prompts)] for j in range(len(input_prompts))]
            )

        num_tokens_generated_decode = []

        logger.info("Starting inference...")

        # Main inference loop for repeat batches (like tt_transformers)
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
                input_prompts_batch, tokenizer, [model_args], instruct, max_generated_tokens, max_prefill_len=128
            )
            # TODO max_prefill_len=max_seq_len in max_prefill_len=max_seq_len

            input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)
            profiler.end(f"preprocess_prefill_inputs", iteration=batch_idx)

            logger.info(f"Input prompt: {input_prompts_batch[0]}")
            logger.info(f"Encoded length: {prefill_lens[0]} tokens")

            # Clear KV caches for repeat batches (like tt_transformers)
            if batch_idx != 0:
                for i in range(len(model)):
                    for layer in model[i].layers:
                        # DeepSeek-V3 doesn't use traditional KV cache, skip clearing
                        pass

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
                    decode_iteration_time = profiler.get_duration(
                        f"inference_decode_time_{iteration}", iteration=batch_idx
                    )

                # Print perf after every iteration
                tokens_per_second_per_user = 1 / decode_iteration_time
                logger.debug(
                    f"Iteration {iteration}: {1000*decode_iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({global_batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
                )

                current_pos += 1

                # Save output token
                for user in range(global_batch_size):
                    user_tok = out_tok[user].item()
                    if user_tok not in tokenizer.stop_tokens and user_done[user] == False:
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
                prompt_including_assistant_tags = tokenizer.decode(
                    model_args[0].encode_prompt(prompt, instruct=instruct)
                )
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
        decode_tok_s = (
            (num_tokens_generated_decode[0] - 1) / total_inference_decode_time * global_batch_size
        )  # total t/s

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

        logger.info("DeepSeek-V3 demo completed successfully!")
    finally:
        # Clean up mesh device(s)
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
        # Reset fabric config back to disabled after the run
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main() -> None:
    input_prompts = ["How many r's in the word 'strawberry'?"]

    run_demo(input_prompts=input_prompts)


if __name__ == "__main__":
    main()
