# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import PagedAttentionConfig
from models.tt_transformers.tt.embedding_model import EmbeddingTransformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs, determine_device_name


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
    all_prompts = []
    for i in range(len(user_input)):
        prompt = user_input[i]["prompt"]
        all_prompts.append(prompt)  # return all the prompts taken from the input file to be used when repeat_batch > 1
        if i in range(batch):
            in_prompt.append(prompt)
    return in_prompt, all_prompts


def create_tt_embedding_model(
    mesh_device,
    instruct,
    max_batch_size,
    optimizations,
    max_seq_len,
    paged_attention_config: PagedAttentionConfig = None,
    dtype=ttnn.bfloat8_b,
    state_dict=None,
    num_layers=None,
):
    """Create TT embedding model using the EmbeddingTransformer"""
    from models.tt_transformers.tt.model_config import ModelArgs

    tt_model_args = ModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
    )
    if num_layers is not None:
        tt_model_args.n_layers = num_layers

    # Avoid loading state_dict for every DP model
    if not state_dict:
        state_dict = tt_model_args.load_state_dict()

    # Create embedding model with limited layers for memory efficiency
    model = EmbeddingTransformer(
        args=tt_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )

    tt_kv_cache = [l.attention.layer_past for l in model.layers] if paged_attention_config else None

    return tt_model_args, model, tt_kv_cache, state_dict


@pytest.mark.parametrize(
    "input_prompts, max_seq_len, batch_size, max_embeddings, paged_attention, page_params",
    [
        (  # Batch-1 embedding generation - single user, small prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            128,  # max_seq_len (small for testing, divisible by 128)
            1,  # batch_size
            1,  # max_embeddings (always 1 for embeddings)
            False,  # paged_attention (disabled for embeddings)
            None,  # page_params
        ),
        (  # Batch-8 embedding generation - 8 users, small prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            512,  # max_seq_len
            8,  # batch_size
            1,  # max_embeddings (always 1 for embeddings)
            True,  # paged_attention
            {"page_block_size": 128, "page_max_num_blocks_per_dp": 1024},  # page_params
        ),
        (  # Batch-32 embedding generation - 32 users, small prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            512,  # max_seq_len
            32,  # batch_size
            1,  # max_embeddings (always 1 for embeddings)
            True,  # paged_attention
            {"page_block_size": 128, "page_max_num_blocks_per_dp": 1024},  # page_params
        ),
        (  # Long-context-1k embedding generation - single user, long prompt
            "models/tt_transformers/demo/sample_prompts/input_data_long_1k.json",  # input_prompts
            1024,  # max_seq_len
            1,  # batch_size
            1,  # max_embeddings (always 1 for embeddings)
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 2048},  # page_params
        ),
        (  # Long-context-4k embedding generation - single user, long prompt
            "models/tt_transformers/demo/sample_prompts/input_data_long_4k.json",  # input_prompts
            4096,  # max_seq_len
            1,  # batch_size
            1,  # max_embeddings (always 1 for embeddings)
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks_per_dp": 2048},  # page_params
        ),
    ],
    ids=[
        "batch-1",
        "batch-8",
        "batch-32",
        "long-context-1k",
        "long-context-4k",
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
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
            "BHGLX": (8, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_embedding_demo(
    input_prompts,
    max_seq_len,
    batch_size,
    max_embeddings,
    paged_attention,
    page_params,
    optimizations,
    mesh_device,
    is_ci_env,
    reset_seeds,
    request,
):
    """
    Embedding generation demo with limited dependence on reference code.
    """
    test_id = request.node.callspec.id

    # Skip if this is a CI environment and not a CI-only test
    if is_ci_env and "ci-" not in test_id:
        pytest.skip("CI only runs the CI-only tests")

    # TODO: Remove this once all batch sizes are supported on TG
    if os.environ.get("MESH_DEVICE") == "TG" and batch_size not in [1, 32]:
        pytest.skip("TG only supports batch 1 and 32")

    # Start profiler
    logger.info("Start profiler")
    profiler = BenchmarkProfiler()
    profiler.start("run")

    logger.info("Reading inputs...")
    profiler.start("loading_inputs")
    if len(input_prompts) == 1:  # Manual input
        input_prompts = input_prompts * batch_size
        all_prompts = input_prompts
    else:  # Inputs from file
        input_prompts, all_prompts = load_inputs(input_prompts, batch_size, False)  # instruct=False for embeddings
    profiler.end("loading_inputs")

    # Prepare model
    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1

    paged_attention_config = (
        PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks_per_dp"],
        )
        if paged_attention
        else None
    )

    # First try to create basic model args to get configuration
    try:
        basic_model_args = ModelArgs(
            mesh_device,
            instruct=False,
            max_batch_size=batch_size,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
        )
    except Exception as e:
        logger.error(f"Failed to create basic model args: {e}")
        raise e

    try:
        model_args, model, tt_kv_cache, state_dict = create_tt_embedding_model(
            mesh_device=mesh_device,
            instruct=False,  # Embeddings don't use instruction tuning
            max_batch_size=batch_size,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            num_layers=None,  # Use all layers for production
        )
    except RuntimeError as e:
        if "Out of Memory" in str(e):
            logger.warning(
                f"Full {basic_model_args.n_layers}-layer model too large for device. Reducing to fewer layers..."
            )
            # Try with half the layers
            reduced_layers = max(1, basic_model_args.n_layers // 2)
            logger.info(f"Retrying with {reduced_layers} layers...")
            model_args, model, tt_kv_cache, state_dict = create_tt_embedding_model(
                mesh_device=mesh_device,
                instruct=False,
                max_batch_size=batch_size,
                optimizations=optimizations,
                max_seq_len=max_seq_len,
                paged_attention_config=paged_attention_config,
                dtype=ttnn.bfloat8_b,
                num_layers=reduced_layers,
            )
            logger.info(f"Successfully loaded model with {reduced_layers} layers")
        else:
            raise e

    # Create tokenizer
    tokenizer = model_args.create_tokenizer()

    # Validate sequence length compatibility
    original_max_seq_len = max_seq_len
    if max_seq_len > model_args.max_seq_len:
        logger.warning(
            f"Requested sequence length {max_seq_len} exceeds model max {model_args.max_seq_len}. Using model max."
        )
        max_seq_len = model_args.max_seq_len

    logger.info(f"Using max_seq_len: {max_seq_len}, model max_seq_len: {model_args.max_seq_len}")

    # Prepare inputs (moved after model creation to use validated max_seq_len)
    logger.info("Preparing inputs...")
    profiler.start("preprocess_inputs")

    # For embeddings, we process all inputs at once
    tokens_list = []
    attention_masks = []

    for text in input_prompts:
        # Tokenize text
        tokens = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
        tokens = tokens.squeeze(0)  # Remove batch dimension

        # Pad or truncate to max_seq_len
        if len(tokens) < max_seq_len:
            padding = torch.full((max_seq_len - len(tokens),), tokenizer.pad_token_id)
            tokens = torch.cat([tokens, padding])
            attention_mask = torch.cat([torch.ones(len(tokens) - len(padding)), torch.zeros(len(padding))])
        else:
            tokens = tokens[:max_seq_len]
            attention_mask = torch.ones(len(tokens))

        tokens_list.append(tokens)
        attention_masks.append(attention_mask)

    # Stack into batch
    tokens_batch = torch.stack(tokens_list)  # [batch, seq_len]
    attention_mask_batch = torch.stack(attention_masks).unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq_len, 1]

    logger.info(f"Input batch shape: {tokens_batch.shape}, attention_mask shape: {attention_mask_batch.shape}")
    logger.info(f"First sequence length: {tokens_batch[0].shape[0] if len(tokens_batch) > 0 else 'N/A'}")

    profiler.end("preprocess_inputs")

    # Prepare inputs for TT model
    logger.info("Preparing TT inputs...")
    profiler.start("prepare_tt_inputs")

    (
        tt_tokens,
        tt_rot_mats_global,
        tt_rot_mats_local,
        tt_page_table,
        tt_chunk_page_table,
    ) = model.prepare_inputs_prefill(tokens_batch)

    # Convert attention mask to TT tensor
    tt_attention_mask = ttnn.from_torch(
        attention_mask_batch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    profiler.end("prepare_tt_inputs")

    # Warm-up phase
    logger.info("Running warm-up...")
    profiler.start("warmup")
    for warmup_iter in range(3):  # Run 3 warm-up iterations
        _ = model.forward(
            x=tt_tokens,
            current_pos=None,
            rot_mats_global=tt_rot_mats_global,
            rot_mats_local=tt_rot_mats_local,
            mode="prefill",
            attention_mask=tt_attention_mask,
        )
        logger.debug(f"Warm-up iteration {warmup_iter + 1}/3 completed")
    profiler.end("warmup")

    # Main embedding generation
    logger.info("Generating embeddings...")
    profiler.start("embedding_generation")

    tt_embeddings = model.forward(
        x=tt_tokens,
        current_pos=None,
        rot_mats_global=tt_rot_mats_global,
        rot_mats_local=tt_rot_mats_local,
        mode="prefill",
        attention_mask=tt_attention_mask,
    )

    # Convert to torch
    embeddings = ttnn.to_torch(tt_embeddings)

    # Extract embeddings (remove extra dimensions)
    embeddings = embeddings.squeeze(1).squeeze(1)  # [batch, hidden_dim]

    profiler.end("embedding_generation")

    # Validate outputs
    logger.info(f"Embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (
        batch_size,
        model_args.dim,
    ), f"Expected shape ({batch_size}, {model_args.dim}), got {embeddings.shape}"

    # Check that embeddings are not all zeros
    assert not torch.allclose(embeddings, torch.zeros_like(embeddings)), "Embeddings should not be all zeros"

    # Check for NaN or Inf values
    assert not torch.isnan(embeddings).any(), "Embeddings contain NaN values"
    assert not torch.isinf(embeddings).any(), "Embeddings contain Inf values"

    # Finish profiling
    profiler.end("run")

    # Calculate performance metrics
    embedding_generation_time = profiler.get_duration("embedding_generation")
    warmup_time = profiler.get_duration("warmup")
    total_time = profiler.get_duration("run")

    embeddings_per_second = batch_size / embedding_generation_time
    # Throughput excluding warm-up time
    embeddings_per_second_throughput = batch_size / (total_time - warmup_time)

    logger.info("")
    logger.info("=== Performance metrics ===")
    logger.info(".2f")
    logger.info(".2f")
    logger.info(".3f")
    logger.info(".2f")
    logger.info(f"Model: {model_args.model_name} ({model_args.n_layers} layers)")
    logger.info(f"Embedding dimension: {model_args.dim}")
    logger.info(f"Sequence length: {max_seq_len}")
    logger.info(f"Batch size: {batch_size}")

    # Print sample embeddings (first user)
    logger.info("")
    logger.info("=== Sample Embeddings ===")
    logger.info(f"First 10 values of first embedding: {embeddings[0, :10].tolist()}")

    # Save benchmark data for CI dashboard
    if is_ci_env:
        from models.demos.utils.llm_demo_utils import create_benchmark_data

        measurements = {
            "embedding_generation_time": embedding_generation_time,
            "total_time": total_time,
            "embeddings_per_second": embeddings_per_second,
            "embeddings_per_second_throughput": embeddings_per_second_throughput,
        }

        benchmark_data = create_benchmark_data(
            profiler,
            measurements,
            {"embedding_generation": 0},  # No warmup iterations for embeddings
            {},  # No performance targets yet
        )

        benchmark_data.save_partial_run_json(
            profiler,
            run_type=f"{determine_device_name(mesh_device)}-embedding-demo",
            ml_model_name=model_args.model_name,
            ml_model_type="embedding",
            num_layers=model_args.n_layers,
            batch_size=batch_size,
            config_params={"data_parallel": 1, "tensor_parallel": num_devices},
            input_sequence_length=max_seq_len,
            output_sequence_length=1,  # Embeddings are single vectors
        )

    logger.info("Embedding demo completed successfully!")
