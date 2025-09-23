# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS demo using tt_transformers generation pipeline

Integrates GPT-OSS with tt_transformers infrastructure for:
- Paged attention support
- Sophisticated generation loop with sampling
- Performance profiling and benchmarking
- Multi-user batch generation capability
"""

import math

import pytest
import torch
from loguru import logger

import ttnn

# Import GPT-OSS create_tt_model
from models.demos.gpt_oss.tt.common import create_tt_model
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill, sample_host

# Import specific utilities from tt_transformers
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.model_config import DecodersPrecision
from models.utility_functions import run_for_wormhole_b0


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
        )
        model_args.append(model_args_i)
        model.append(model_i)
        tt_kv_cache.append(tt_kv_cache_i)

    # NOTE: We'll create the page table later when we know the actual sequence length
    # This is because the page table should be sized for the actual sequence, not max possible
    page_table = None

    # Host code, safe to reuse tokenizer from the 1st model
    tokenizer = model_args[0].tokenizer
    processor = model_args[0].processor
    return model_args, model, page_table, tt_kv_cache, tokenizer, processor, paged_attention_config


@run_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 42087296}], indirect=True
)
def test_gpt_oss_demo(mesh_device):
    """GPT-OSS demo using full tt_transformers generation pipeline"""
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 8)))

    # Configuration matching tt_transformers defaults
    num_devices = mesh_device.get_num_devices()
    data_parallel = 1
    paged_attention = True
    global_batch_size = 1
    max_seq_len = 1024
    max_generated_tokens = 200  # Reasonable limit for testing
    instruct = True
    enable_trace = True  # Start with trace disabled

    page_params = {
        "page_block_size": 64,  # User says block_size should be 64
        "page_max_num_blocks_per_dp": 1024 // 64,
    }

    sampling_params = {
        "temperature": 0,  # Greedy decoding for deterministic results
        "top_p": 0.08,
    }

    logger.info(f"Running GPT-OSS demo with tt_transformers generation pipeline")

    # Setup profiler like tt_transformers
    profiler = BenchmarkProfiler()
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
    )

    # Create generator
    generator = Generator(model=model, model_args=model_args, mesh_device=mesh_device)

    profiler.end(f"generator_setup", iteration=batch_idx)

    # Prepare input like tt_transformers does
    input_prompts = ["How many r's in the word 'strawberry'?"]

    # Preprocess inputs (reusing tt_transformers function)
    profiler.start(f"preprocess_prefill_inputs", iteration=batch_idx)
    (
        input_tokens_prefill_pt,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    ) = preprocess_inputs_prefill(
        input_prompts, tokenizer, model_args, instruct, max_generated_tokens, max_prefill_len=max_seq_len
    )

    logger.info(
        f"Input tokens prefill pt: {encoded_prompts}, {decoding_pos}, {prefill_lens}, {input_tokens_prefill_pt}"
    )
    print(tokenizer.decode(encoded_prompts[0]))
    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)
    profiler.end(f"preprocess_prefill_inputs", iteration=batch_idx)

    logger.info(f"Input prompt: {input_prompts[0]}")
    logger.info(f"Encoded length: {prefill_lens[0]} tokens")

    # Create page table sized for actual sequence length
    if paged_attention:
        actual_seq_len = prefill_lens[0]
        expected_blocks = math.ceil(actual_seq_len / page_params["page_block_size"])
        # Create page table with exactly the right number of blocks
        # Use first 'expected_blocks' from a permutation of available blocks
        permutation = torch.arange(paged_attention_config.max_num_blocks)
        reverse_permutation = torch.argsort(permutation)

        # Take only the blocks we need for this sequence
        blocks_needed = expected_blocks
        page_table = reverse_permutation[:].unsqueeze(0)  # Shape: [1, blocks_needed]

        print(
            f"✅ Created page table: shape={page_table.shape}, blocks_needed={blocks_needed} for seq_len={actual_seq_len}"
        )

    else:
        page_table = None
    print("page_table", page_table)

    # Prefill phase (matching tt_transformers)
    logger.info("Starting prefill warmup...")
    profiler.start(f"compile_prefill", iteration=batch_idx)
    logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,  # [:, :expected_blocks],
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
    )
    profiler.end(f"compile_prefill", iteration=batch_idx)
    logger.info("Finished prefill warmup")

    logger.info(f"Starting prefill...")
    profiler.start(f"inference_prefill", iteration=batch_idx)
    logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,  # [:, :expected_blocks],
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
    )
    print("logits", logits)
    prefilled_token = torch.argmax(logits, dim=-1)
    profiler.end(f"inference_prefill", iteration=batch_idx)
    logger.info(f"Prefill finished")
    print(tokenizer.decode(prefilled_token[0]))
    # return True

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

        current_pos += 1

        # Save output token
        for user in range(global_batch_size):
            user_tok = out_tok[user].item()
            if user_tok not in tokenizer.stop_tokens and user_done[user] == False:
                all_outputs[user].append(user_tok)
            else:
                user_done[user] = True
                logger.info(f"User {user} finished decoding at iteration {iteration}")
                if all(user_done):
                    users_decoding = False

        iteration += 1

    profiler.end(f"inference_decode", iteration=batch_idx)

    # Final output (like tt_transformers)
    logger.info("Finished decoding, printing the final outputs...\n")
    for i, (output, prompt) in enumerate(zip(all_outputs, input_prompts)):
        text = tokenizer.decode(output)
        logger.info(f"User {i}:")
        logger.info(f"  Input: {prompt}")
        logger.info(f"  Output: {text}")
        logger.info("")

    logger.info("✅ GPT-OSS demo completed successfully!")
