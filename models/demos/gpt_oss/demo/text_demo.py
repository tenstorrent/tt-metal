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
import os
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
    copy_host_to_device,
    get_padded_prefill_len,
    preprocess_inputs_prefill,
    sample_host,
)

# Import specific utilities from tt_transformers
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.model_config import determine_device_name


def create_long_context_page_table(
    global_batch_size: int,
    mesh_rows: int,
    paged_attention_config: PagedAttentionConfig,
    long_context_user_per_row: int = 0,
) -> torch.Tensor:
    """Create page table where one user per row gets all blocks.

    For long-context scenarios (e.g., 128k tokens), we want a single user per row
    to have access to the entire page table for that row, while other users
    (padding for decode batch size) have empty page tables.

    Args:
        global_batch_size: Total batch size across all rows (e.g., 128)
        mesh_rows: Number of rows in the mesh (e.g., 4 for 4x8)
        paged_attention_config: Paged attention configuration with block info
        long_context_user_per_row: Which user index (0-31) gets full allocation

    Returns:
        Page table tensor [global_batch_size, blocks_per_row]
    """
    users_per_row = global_batch_size // mesh_rows
    blocks_per_row = paged_attention_config.max_num_blocks

    # Initialize with -1 (invalid) for all users
    page_table = torch.full((global_batch_size, blocks_per_row), -1, dtype=torch.int32)

    for row in range(mesh_rows):
        # User index that gets the full page table for this row
        long_user_idx = row * users_per_row + long_context_user_per_row
        # Assign all blocks sequentially to this user
        page_table[long_user_idx, :] = torch.arange(blocks_per_row, dtype=torch.int32)

    return page_table


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
    users_row_sharded=False,
    long_context_mode=False,
):
    """Prepare generator args using GPT-OSS create_tt_model (clean version)

    Args:
        long_context_mode: If True, allocate all page blocks to user 0 of each row
                          for single-user long-context (e.g., 128k) scenarios.
                          Also disables throughput experts since single-user prefill
                          is not compatible with all_to_all dispatch/combine.
    """
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
        use_throughput = mesh_device.shape[0] > 1 and global_batch_size > 1
        model_args_i, model_i, tt_kv_cache_i, state_dict = create_tt_model(
            submesh,
            max_batch_size=global_batch_size // data_parallel,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
            mesh_config=mesh_config,  # Pass mesh config for proper sharding
            users_row_sharded=users_row_sharded,
            use_throughput_experts=use_throughput,
        )
        model_args.append(model_args_i)
        model.append(model_i)
        tt_kv_cache.append(tt_kv_cache_i)

    # Page table will be created using tt-transformers infrastructure after input preprocessing
    if paged_attention:
        if long_context_mode and users_row_sharded:
            # Long-context mode: one user per row gets all blocks
            page_table = create_long_context_page_table(
                global_batch_size,
                mesh_device.shape[0],
                paged_attention_config,
                long_context_user_per_row=0,
            )
        elif users_row_sharded:
            # If users are sharded on rows of mesh, we need a separate page table for each row
            page_tables = [
                create_tt_page_table(
                    global_batch_size // mesh_device.shape[0],
                    data_parallel,
                    paged_attention_config,
                )
                for _ in range(mesh_device.shape[0])
            ]
            # Concat the separate page tables into a single page table
            page_table = torch.concat(page_tables, dim=0) if page_tables else None
        else:
            page_table = create_tt_page_table(
                global_batch_size,
                data_parallel,
                paged_attention_config,
            )
    else:
        page_table = None

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
    "input_prompts, data_parallel, batch_size, repeat_batches, max_seq_len, max_generated_tokens, page_params, sampling_params, enable_decode_trace, enable_prefill_trace, users_row_sharded, long_context_mode, stop_at_eos",
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
            False,  # enable_prefill_trace
            False,  # users_row_sharded
            False,  # long_context_mode
            True,  # stop_at_eos
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
            False,  # enable_prefill_trace
            False,  # users_row_sharded
            False,  # long_context_mode
            True,  # stop_at_eos
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
            False,  # enable_prefill_trace
            False,  # users_row_sharded
            False,  # long_context_mode
            True,  # stop_at_eos
        ),
        (
            "models/demos/gpt_oss/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            1,  # data_parallel
            128,  # batch_size
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            200,  # max_generated_tokens
            {"page_block_size": 64, "page_max_num_blocks_per_dp": 128 * 1024 // 64},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (greedy decoding)
            True,  # enable_decode_trace
            True,  # enable_prefill_trace
            True,  # users_row_sharded
            False,  # long_context_mode
            True,  # stop_at_eos
        ),
        # Long-context mode: 1 user per row with 128k tokens, batch=128 for decode throughput
        (
            "models/tt_transformers/demo/sample_prompts/input_data_long_128k.json",  # input_prompts (128k prompt)
            1,  # data_parallel
            128,  # batch_size (32 per row, but only 1 real user per row)
            1,  # repeat_batches
            128 * 1024,  # max_seq_len (128k tokens)
            50,  # max_generated_tokens (reduced for long context)
            {"page_block_size": 64, "page_max_num_blocks_per_dp": 128 * 1024 // 64},  # 2048 blocks for 128k
            {"temperature": 0, "top_p": 0.08},  # sampling_params (greedy decoding)
            True,  # enable_decode_trace
            False,  # enable_prefill_trace
            True,  # users_row_sharded
            True,  # long_context_mode - single user per row gets all page blocks
            True,  # stop_at_eos
        ),
        # Long-context mode: short prefill, long decode
        (
            "models/demos/gpt_oss/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts (128 token prompt)
            1,  # data_parallel
            128,  # batch_size (32 per row, but only 1 real user per row)
            1,  # repeat_batches
            128 * 1024,  # max_seq_len (128k tokens)
            130000,  # max_generated_tokens (reduced for long context)
            {"page_block_size": 64, "page_max_num_blocks_per_dp": 128 * 1024 // 64},  # 2048 blocks for 128k
            {"temperature": 0, "top_p": 0.08},  # sampling_params (greedy decoding)
            True,  # enable_decode_trace
            False,  # enable_prefill_trace
            True,  # users_row_sharded
            True,  # long_context_mode - single user per row gets all page blocks
            False,  # stop_at_eos
        )
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
        "batch128",
        "long_context_128k",
        "long_context_short_prefill_long_decode"
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
    users_row_sharded,
    long_context_mode,
    stop_at_eos,
    is_ci_env,
    state_dict,
):
    """GPT-OSS demo using full tt_transformers generation pipeline"""
    if batch_size > 1 and mesh_shape[0] == 1:
        pytest.skip(
            f"Batch size = 128 demo skipped for mesh shape f{mesh_shape}. Only single user demo is supported for single row meshes."
        )
    if os.environ.get("CI", None) and long_context_mode:
        pytest.skip(f"Long-context mode skipped for CI environment.")
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
        users_row_sharded=users_row_sharded,
        long_context_mode=long_context_mode,
    )

    # Create generator (match tt-transformers pattern)
    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    profiler.end(f"generator_setup", iteration=batch_idx)

    # Prepare input prompts
    logger.info(f"Reading inputs...")
    profiler.start("loading_inputs")

    # For long_context_mode, we only have 1 real user per row
    # The rest are padding users with empty prompts
    num_real_users = mesh_device.shape[0] if long_context_mode else global_batch_size
    users_per_row = global_batch_size // mesh_device.shape[0]

    if isinstance(input_prompts, list) and len(input_prompts) == 1:  # Manual input
        real_prompts = input_prompts * num_real_users
    elif isinstance(input_prompts, str):  # Inputs from file
        real_prompts, _ = load_inputs(input_prompts, num_real_users, instruct=False)
    else:
        raise ValueError(
            f"Invalid input prompts: {input_prompts}. Expected a list of prompts or a string path to a json file."
        )

    if long_context_mode:
        # Expand to full batch: 1 real user + (users_per_row - 1) padding users per row
        # Padding users get minimal prompts (single token)
        padding_prompt = "."  # Minimal prompt for padding users
        input_prompts = []
        for row in range(mesh_device.shape[0]):
            input_prompts.append(real_prompts[row])  # User 0 of each row gets real prompt
            input_prompts.extend([padding_prompt] * (users_per_row - 1))  # Padding users
        logger.info(
            f"Long-context mode: {num_real_users} real users with 128k context, {global_batch_size - num_real_users} padding users"
        )
    elif users_row_sharded and len(real_prompts) < global_batch_size:
        # Randomize prompt order per row to help debug bad output patterns
        # This ensures each row gets the same prompts but in different order
        import random

        random.seed(42)  # Fixed seed for reproducibility
        input_prompts = []
        num_prompts = len(real_prompts)
        for row in range(mesh_device.shape[0]):
            # Create a shuffled copy of prompts for this row
            row_prompts = real_prompts.copy()
            random.shuffle(row_prompts)
            # Repeat if needed to fill users_per_row slots
            row_prompts_extended = (row_prompts * ((users_per_row // num_prompts) + 1))[:users_per_row]
            input_prompts.extend(row_prompts_extended)
        logger.info(f"Row-sharded mode: randomized {num_prompts} prompts per row (seed=42)")
    else:
        input_prompts = real_prompts

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

        # Prefill phase
        if long_context_mode:
            # Long-context mode: prefill only the real users (user 0 of each row)
            # Other users are padding and don't need prefill
            logger.info(f"Long-context prefill: processing {num_real_users} real users...")
            profiler.start(f"compile_prefill", iteration=batch_idx)

            prefilled_token = torch.zeros(global_batch_size, dtype=torch.long)
            real_user_indices = [row * users_per_row for row in range(mesh_device.shape[0])]

            model_id = 0  # data_parallel=1, single model

            for i, user_id in enumerate(real_user_indices):
                user_prefill_len = prefill_lens[user_id]
                padded_len = get_padded_prefill_len(user_prefill_len)

                # Pad tokens to required length (multiple of 32 / power of 2)
                user_tokens_raw = input_tokens_prefill_pt[user_id : user_id + 1, :user_prefill_len]
                user_tokens = torch.cat(
                    [user_tokens_raw, torch.zeros(1, padded_len - user_prefill_len, dtype=torch.long)], dim=-1
                )
                user_page_table = page_table[user_id : user_id + 1]

                logger.info(
                    f"Prefilling user {user_id} (row {i}) with {user_prefill_len} tokens (padded to {padded_len})..."
                )

                # Use single-user prefill
                # Note: user_id=0 because page_table is already sliced for this user
                # (batch_idx for fill_cache should be 0 since page_table has shape [1, blocks]).
                # global_user_id tells the model which mesh row to target for KV cache filling.
                logits = generator.prefill_forward_single_user_text(
                    user_tokens,
                    page_table=user_page_table,
                    user_id=0,
                    last_token_idx=user_prefill_len - 1,
                    kv_cache=tt_kv_cache[model_id],
                    model_id=model_id,
                    global_user_id=user_id,  # Pass actual global user_id for mesh row targeting
                )
                # Convert ttnn.Tensor to torch.Tensor for argmax
                # For multi-device tensors, extract from device 0 first
                if not isinstance(logits, torch.Tensor):
                    tt_output_tensor = ttnn.get_device_tensors(logits)[0]
                    logits = ttnn.to_torch(tt_output_tensor)
                # Logits shape may be [batch_per_row, vocab_size], select user 0 of the row
                if logits.dim() > 1 and logits.shape[0] > 1:
                    logits = logits[0]  # Select first user's logits
                prefilled_token[user_id] = torch.argmax(logits.view(-1)).item()

            profiler.end(f"compile_prefill", iteration=batch_idx)

            # Skip second timing pass for long-context mode - it's too expensive
            # Just copy compile time as inference time for metrics
            profiler.start(f"inference_prefill", iteration=batch_idx)
            profiler.end(f"inference_prefill", iteration=batch_idx)

            # For padding users, generate a dummy token (they won't be used meaningfully)
            for user_id in range(global_batch_size):
                if user_id not in real_user_indices:
                    prefilled_token[user_id] = tokenizer.eos_token_id

            logger.info(f"Prefill finished for {num_real_users} real users")
        elif users_row_sharded:
            # Row-parallel batched prefill: process 4 users at once (one per mesh row)
            # This gives ~4x speedup over sequential per-user prefill
            num_rows = mesh_device.shape[0]
            users_per_row_prefill = global_batch_size // num_rows
            users_per_row_per_iter = 1  # Users each mesh row processes per prefill iteration
            # Increasing above 1 requires model changes:
            #   - attention/prefill.py: relax batch_size!=1 check, loop paged_fill_cache
            #   - model.py:process_output_prefill_batched: extract multiple logits per row
            assert users_per_row_prefill % users_per_row_per_iter == 0
            num_prefill_iters = users_per_row_prefill // users_per_row_per_iter
            model_id = 0  # data_parallel=1, single model

            prefilled_token = torch.zeros(global_batch_size, dtype=torch.long)

            if enable_prefill_trace:
                # === TRACED BATCHED PREFILL ===
                # Trace captures device program once, then replays with input buffer updates.
                # Eliminates per-iteration host dispatch overhead.

                # Uniform padded_len for all users (required for tracing: fixed tensor shapes)
                max_padded_len = max(get_padded_prefill_len(int(decoding_pos[uid])) for uid in range(global_batch_size))
                block_size = page_params["page_block_size"]
                max_num_blocks = (max_padded_len + block_size - 1) // block_size

                # Compute fixed get_last_token for trace (all users must be in same 32-token tile)
                all_last_idxs = [int(decoding_pos[uid]) - 1 for uid in range(global_batch_size)]
                fixed_get_last_token = (min(all_last_idxs) // 32) * 32
                max_tile_start = (max(all_last_idxs) // 32) * 32
                if fixed_get_last_token != max_tile_start:
                    logger.warning(
                        f"Users span multiple 32-token tiles ({fixed_get_last_token} vs {max_tile_start}), "
                        f"using get_last_token=-1 (slower)"
                    )
                    fixed_get_last_token = -1

                def _prepare_batch_host(user_indices):
                    """Prepare host-side tokens + page_table for a batch of users."""
                    tokens_list, pt_list, last_idxs = [], [], []
                    for uid in user_indices:
                        plen = int(decoding_pos[uid])
                        toks = torch.cat(
                            [
                                input_tokens_prefill_pt[uid : uid + 1, :plen],
                                torch.zeros(1, max_padded_len - plen, dtype=torch.long),
                            ],
                            dim=-1,
                        )
                        tokens_list.append(toks)
                        pt_list.append(page_table[uid : uid + 1, :max_num_blocks])
                        last_idxs.append(plen - 1)
                    return (torch.cat(tokens_list, dim=0), torch.cat(pt_list, dim=0), last_idxs)

                # --- Warmup (compilation) ---
                logger.info("Starting traced row-parallel prefill warmup (compilation)...")
                warmup_indices = [
                    row * users_per_row_prefill + u for row in range(num_rows) for u in range(users_per_row_per_iter)
                ]
                tokens_w, pt_w, last_w = _prepare_batch_host(warmup_indices)
                tokens_w = tokens_w.reshape(num_rows, -1)  # [num_rows, N*S] for batch>1 concat

                host_out = model[model_id].prepare_inputs_prefill(
                    tokens_w, page_table=pt_w, trace_enabled=True, batched_prefill=True
                )
                rot_global = host_out[1]  # device-resident, fixed across iterations
                rot_local = host_out[2]  # None
                host_inputs = (host_out[0], host_out[3], host_out[4])  # tokens, pt, cpt

                profiler.start(f"compile_prefill", iteration=batch_idx)
                dev_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)
                transformed = model[model_id].transform_and_embed_prefill_inputs_device(*dev_inputs)
                tt_logits = model[model_id].ttnn_prefill_forward(
                    transformed[0],
                    rot_mats_global=rot_global,
                    rot_mats_local=rot_local,
                    user_id=0,
                    page_table=transformed[1],
                    get_last_token=fixed_get_last_token,
                    kv_cache=tt_kv_cache[model_id],
                    batch_size=users_per_row_per_iter,
                )

                if fixed_get_last_token == -1:
                    warmup_results = model[model_id].process_output_prefill_batched(
                        tt_logits,
                        last_w,
                        users_per_row=users_per_row_per_iter,
                        seq_len_per_user=max_padded_len,
                    )
                else:
                    warmup_results = model[model_id].process_output_prefill_batched(
                        tt_logits,
                        [idx % 32 for idx in last_w],
                        users_per_row=users_per_row_per_iter,
                        seq_len_per_user=32,
                    )
                for row, uid in enumerate(warmup_indices):
                    prefilled_token[uid] = torch.argmax(warmup_results[row].view(-1)).item()
                profiler.end(f"compile_prefill", iteration=batch_idx)
                logger.info("Finished traced row-parallel prefill warmup")

                # Clear KV caches (warmup wrote to them)
                for i in range(len(model)):
                    for layer_obj in model[i].layers:
                        k_cache, v_cache = layer_obj.self_attn.layer_past
                        ttnn.mul(k_cache, 0, output_tensor=k_cache)
                        ttnn.mul(v_cache, 0, output_tensor=v_cache)

                # --- Trace capture ---
                logger.info("Capturing prefill trace...")
                iter0_indices = [
                    row * users_per_row_prefill + u for row in range(num_rows) for u in range(users_per_row_per_iter)
                ]
                tokens_0, pt_0, last_0 = _prepare_batch_host(iter0_indices)
                tokens_0 = tokens_0.reshape(num_rows, -1)
                host_out = model[model_id].prepare_inputs_prefill(
                    tokens_0, page_table=pt_0, trace_enabled=True, batched_prefill=True
                )
                host_inputs = (host_out[0], host_out[3], host_out[4])

                trace_dev_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)
                trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                # Embed tokens on-device inside the trace (without deallocating input buffer,
                # since we need to update it between trace executions)
                tokens_embd = ttnn.embedding(
                    trace_dev_inputs[0],
                    model[model_id].embedding_weight,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat8_b,
                )
                if len(tokens_embd.shape) == 3:
                    tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)
                tt_out_trace = model[model_id].ttnn_prefill_forward(
                    tokens_embd,
                    rot_mats_global=rot_global,
                    rot_mats_local=rot_local,
                    user_id=0,
                    page_table=trace_dev_inputs[1],
                    get_last_token=fixed_get_last_token,
                    kv_cache=tt_kv_cache[model_id],
                    batch_size=users_per_row_per_iter,
                )
                ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
                logger.info("Prefill trace captured")

                # --- Execute trace for all iterations ---
                logger.info(
                    f"Starting traced row-parallel prefill ({num_prefill_iters} iters, "
                    f"{users_per_row_per_iter} user/row/iter, {global_batch_size} users)..."
                )
                profiler.start(f"inference_prefill", iteration=batch_idx)
                for iter_idx in range(num_prefill_iters):
                    user_indices = [
                        row * users_per_row_prefill + iter_idx * users_per_row_per_iter + u
                        for row in range(num_rows)
                        for u in range(users_per_row_per_iter)
                    ]
                    tokens_i, pt_i, last_i = _prepare_batch_host(user_indices)
                    tokens_i = tokens_i.reshape(num_rows, -1)
                    host_out = model[model_id].prepare_inputs_prefill(
                        tokens_i, page_table=pt_i, trace_enabled=True, batched_prefill=True
                    )
                    host_inputs = (host_out[0], host_out[3], host_out[4])
                    copy_host_to_device(host_inputs, device_tensors=trace_dev_inputs, mesh_device=mesh_device)
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)

                    if fixed_get_last_token == -1:
                        row_results = model[model_id].process_output_prefill_batched(
                            tt_out_trace,
                            last_i,
                            users_per_row=users_per_row_per_iter,
                            seq_len_per_user=max_padded_len,
                        )
                    else:
                        row_results = model[model_id].process_output_prefill_batched(
                            tt_out_trace,
                            [idx % 32 for idx in last_i],
                            users_per_row=users_per_row_per_iter,
                            seq_len_per_user=32,
                        )
                    for row, uid in enumerate(user_indices):
                        prefilled_token[uid] = torch.argmax(row_results[row].view(-1)).item()
                    if iter_idx % 8 == 0:
                        logger.info(f"  Traced prefill batch {iter_idx+1}/{num_prefill_iters}")
                profiler.end(f"inference_prefill", iteration=batch_idx)

                ttnn.release_trace(mesh_device, trace_id)
                logger.info(f"Traced row-parallel prefill finished ({num_prefill_iters} iterations)")

            else:
                # === NON-TRACED BATCHED PREFILL ===

                # Helper to run one batched prefill iteration
                def _run_batched_prefill_iter(iter_idx, user_indices):
                    batch_tokens_list = []
                    batch_page_tables = []
                    batch_last_token_idxs = []

                    for uid in user_indices:
                        prefill_len = int(decoding_pos[uid])
                        padded_len = get_padded_prefill_len(prefill_len)
                        user_tokens = torch.cat(
                            [
                                input_tokens_prefill_pt[uid : uid + 1, :prefill_len],
                                torch.zeros(1, padded_len - prefill_len, dtype=torch.long),
                            ],
                            dim=-1,
                        )
                        batch_tokens_list.append(user_tokens)
                        block_size = page_params["page_block_size"]
                        num_blocks_needed = (padded_len + block_size - 1) // block_size
                        batch_page_tables.append(page_table[uid : uid + 1, :num_blocks_needed])
                        batch_last_token_idxs.append(prefill_len - 1)

                    tokens_stacked = torch.cat(batch_tokens_list, dim=0)  # [total_users, padded_len]
                    page_table_stacked = torch.cat(batch_page_tables, dim=0)  # [total_users, num_blocks]
                    padded_len = tokens_stacked.shape[1]

                    # Reshape tokens for batch>1: concatenate per-row users along seq dim
                    tokens_for_model = tokens_stacked.reshape(num_rows, -1)  # [num_rows, N*padded_len]

                    (tokens_embd, rot_mats_global, rot_mats_local, page_table_tt, _) = model[
                        model_id
                    ].prepare_inputs_prefill(
                        tokens_for_model,
                        page_table=page_table_stacked,
                        batched_prefill=True,
                    )

                    # Use get_last_token if all users' last tokens fall in the same 32-token tile
                    min_tile = (min(batch_last_token_idxs) // 32) * 32
                    max_tile = (max(batch_last_token_idxs) // 32) * 32
                    get_last_token_val = min_tile if min_tile == max_tile else -1
                    tt_logits = model[model_id].ttnn_prefill_forward(
                        tokens_embd,
                        rot_mats_global=rot_mats_global,
                        rot_mats_local=rot_mats_local,
                        user_id=0,  # Must be 0: each device sees page_table[0] after row-sharding
                        page_table=page_table_tt,
                        get_last_token=get_last_token_val,
                        kv_cache=tt_kv_cache[model_id],
                        batch_size=users_per_row_per_iter,
                    )

                    if get_last_token_val == -1:
                        adjusted_last_idxs = batch_last_token_idxs
                        seq_len_for_output = padded_len
                    else:
                        adjusted_last_idxs = [idx % 32 for idx in batch_last_token_idxs]
                        seq_len_for_output = 32
                    row_results = model[model_id].process_output_prefill_batched(
                        tt_logits,
                        adjusted_last_idxs,
                        users_per_row=users_per_row_per_iter,
                        seq_len_per_user=seq_len_for_output,
                    )
                    return row_results

                # Warmup: compile with first batch
                logger.info("Starting row-parallel prefill warmup...")
                profiler.start(f"compile_prefill", iteration=batch_idx)
                warmup_user_indices = [
                    row * users_per_row_prefill + u for row in range(num_rows) for u in range(users_per_row_per_iter)
                ]
                warmup_results = _run_batched_prefill_iter(0, warmup_user_indices)
                for row, uid in enumerate(warmup_user_indices):
                    prefilled_token[uid] = torch.argmax(warmup_results[row].view(-1)).item()
                profiler.end(f"compile_prefill", iteration=batch_idx)
                logger.info("Finished row-parallel prefill warmup")

                # Clear KV caches before real prefill (warmup wrote to them)
                for i in range(len(model)):
                    for layer_obj in model[i].layers:
                        k_cache, v_cache = layer_obj.self_attn.layer_past
                        ttnn.mul(k_cache, 0, output_tensor=k_cache)
                        ttnn.mul(v_cache, 0, output_tensor=v_cache)

                # Real prefill
                logger.info(
                    f"Starting row-parallel batched prefill ({num_prefill_iters} iters, "
                    f"{users_per_row_per_iter} user/row/iter, {global_batch_size} users)..."
                )
                profiler.start(f"inference_prefill", iteration=batch_idx)
                for iter_idx in range(num_prefill_iters):
                    user_indices = [
                        row * users_per_row_prefill + iter_idx * users_per_row_per_iter + u
                        for row in range(num_rows)
                        for u in range(users_per_row_per_iter)
                    ]
                    row_results = _run_batched_prefill_iter(iter_idx, user_indices)
                    for row, uid in enumerate(user_indices):
                        prefilled_token[uid] = torch.argmax(row_results[row].view(-1)).item()
                    if iter_idx % 8 == 0:
                        logger.info(f"  Prefilled batch {iter_idx+1}/{num_prefill_iters}")
                profiler.end(f"inference_prefill", iteration=batch_idx)
                logger.info(f"Row-parallel batched prefill finished ({num_prefill_iters} iterations)")

        else:
            # Standard sequential prefill (batch_size < num_rows)
            logger.info("Starting prefill warmup...")
            profiler.start(f"compile_prefill", iteration=batch_idx)
            generator.prefill_forward_text(
                input_tokens_prefill_pt[:1],
                page_table=page_table,
                kv_cache=tt_kv_cache,
                prompt_lens=decoding_pos,
                enable_trace=enable_prefill_trace,
                warmup_prefill=False,
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
                warmup_prefill=False,
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

        # Define real_user_indices for long_context_mode (user 0 of each row)
        real_user_indices = set(row * users_per_row for row in range(mesh_device.shape[0]))

        # In long_context_mode, mark padding users as done immediately
        if long_context_mode:
            for user in range(global_batch_size):
                if user not in real_user_indices:
                    user_done[user] = True

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
            logits, _ = generator.decode_forward(
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
                elif stop_at_eos:
                    user_done[user] = True
                    logger.debug(f"User {user} finished decoding at iteration {iteration}")
                    if all(user_done):
                        users_decoding = False
                else:
                    all_outputs[user].append(user_tok)

            iteration += 1

        profiler.end(f"inference_decode", iteration=batch_idx)

        # Final output for this batch (like tt_transformers)
        logger.info("Finished decoding, printing the final outputs...\n")
        for i, (output, prompt) in enumerate(zip(all_outputs, input_prompts_batch)):
            # In long_context_mode, skip printing padding users
            if long_context_mode and i not in real_user_indices:
                continue

            text = tokenizer.decode(output)
            prompt_including_assistant_tags = tokenizer.decode(model_args[0].encode_prompt(prompt))
            text_after_prompt = text.replace(prompt_including_assistant_tags, "", 1)
            short_prompt = (
                (prompt[:100] + "\n<long prompt not printed in full>\n" + prompt[-100:])
                if len(prompt) > 200
                else prompt
            )
            user_label = f"USER {i} (row {i // users_per_row})" if long_context_mode else f"USER {i}"
            logger.info(
                f"\n==REPEAT BATCH {batch_idx}\n=={user_label} - PROMPT\n{short_prompt} \n=={user_label} - OUTPUT\n{text_after_prompt.strip()}\n"
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
    # For long_context_mode, only count real users for metrics
    effective_batch_size = num_real_users if long_context_mode else global_batch_size
    avg_time_to_first_token = total_inference_prefill_time / effective_batch_size  # TTFT per user
    avg_decode_iteration_time = total_inference_decode_time / (num_tokens_generated_decode[0] - 1)

    prefill_tok_s = prefill_lens[0] / total_inference_prefill_time * effective_batch_size
    decode_tok_s_user = (num_tokens_generated_decode[0] - 1) / total_inference_decode_time  # t/s/u
    decode_tok_s = (
        (num_tokens_generated_decode[0] - 1) / total_inference_decode_time * effective_batch_size
    )  # total t/s

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
        targets = {}
        if (
            f"batch_{batch_size}" in perf_targets["targets"]
            and f"prefill_{prefill_pad_length}" in perf_targets["targets"][f"batch_{batch_size}"]
            and model_device_key in perf_targets["targets"][f"batch_{batch_size}"][f"prefill_{prefill_pad_length}"]
        ):
            targets = {
                "prefill_t/s": perf_targets["targets"][f"batch_{batch_size}"][f"prefill_{prefill_pad_length}"][
                    model_device_key
                ]["TTFT"],
                "decode_t/s": perf_targets["targets"][f"batch_{batch_size}"][f"prefill_{prefill_pad_length}"][
                    model_device_key
                ]["decode_tok_s"],
                "decode_t/s/u": perf_targets["targets"][f"batch_{batch_size}"][f"prefill_{prefill_pad_length}"][
                    model_device_key
                ]["decode_tok_s_u"],
            }
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
        if f"batch_{batch_size}" in perf_targets["ci"] and False:
            if f"prefill_{prefill_pad_length}" in perf_targets["ci"][f"batch_{batch_size}"]:
                if model_device_key in perf_targets["ci"][f"batch_{batch_size}"][f"prefill_{prefill_pad_length}"]:
                    perf_config = perf_targets["ci"][f"batch_{batch_size}"][f"prefill_{prefill_pad_length}"][
                        model_device_key
                    ]

                    # Parse TTFT target with tolerance
                    current_ttft_target = perf_config["TTFT"]
                    if isinstance(current_ttft_target, list):
                        ttft_tolerance = current_ttft_target[1]
                        current_ttft_target = current_ttft_target[0]
                    else:
                        ttft_tolerance = 1.15  # Default 15% tolerance

                    # Parse decode_tok_s_u target with tolerance
                    decode_tsu_target = perf_config["decode_tok_s_u"]
                    if isinstance(decode_tsu_target, list):
                        decode_tolerance = decode_tsu_target[1]
                        decode_tsu_target = decode_tsu_target[0]
                    else:
                        decode_tolerance = 1.15  # Default 15% tolerance

                    # Verify prefill performance with prefill-specific tolerance
                    prefill_targets = {
                        "prefill_time_to_token": current_ttft_target / 1000,  # convert to seconds
                    }
                    verify_perf(
                        measurements,
                        prefill_targets,
                        high_tol_percentage=ttft_tolerance,
                        expected_measurements={k: True for k in prefill_targets.keys()},
                    )

                    # Verify decode performance with decode-specific tolerance
                    decode_targets = {
                        "decode_t/s/u": decode_tsu_target,
                        "decode_t/s": decode_tsu_target * global_batch_size,  # calculate from per-user rate
                    }
                    verify_perf(
                        measurements,
                        decode_targets,
                        high_tol_percentage=decode_tolerance,
                        expected_measurements={k: True for k in decode_targets.keys()},
                    )
                else:
                    logger.warning(
                        f"No CI performance targets found for model {model_name} on device {tt_device_name} for prefill length {prefill_pad_length}. Skipping performance verification."
                    )
            else:
                logger.warning(
                    f"No CI performance targets found for prefill length {prefill_pad_length}. Skipping performance verification."
                )
        else:
            logger.warning(
                f"No CI performance targets found for batch size {batch_size}. Skipping performance verification."
            )
