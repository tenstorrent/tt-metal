# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import gather_cos_sin, precompute_freqs, rope_scaling_model_factory
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format
from models.tt_transformers.tt.rope import RotarySetup

from ...tt.layer import DecoderLayer
from ..test_factory import TestFactory, compare_tensors, parametrize_batch_seq, parametrize_mesh_with_fabric


# Helper Functions for Common Test Patterns
def run_component_comparison(tt_output, reference_output, mesh_device, pcc_threshold=0.99):
    """Standard component output comparison"""
    tt_output_tensors = ttnn.get_device_tensors(tt_output)

    passing_final = True
    for i in range(len(tt_output_tensors)):
        tt_output_torch = ttnn.to_torch(tt_output_tensors[i])
        passing, output = compare_tensors(tt_output_torch, reference_output, mesh_device, pcc_threshold=pcc_threshold)
        passing_final = passing_final and passing
    if passing_final:
        return True, output
    else:
        return False, output


def run_attention_component(
    mesh_device,
    hidden_shape,
    mask,
    position_embeddings,
    rope_mats,
    tt_position_idx,
    reference_layer,
    decoder_layer,
    is_decode,
    is_row_sharded,
    pcc_threshold,
):
    """Test attention component - extracted from decoder layer"""

    # Create input
    batch_size, seq_len, hidden_size = hidden_shape
    hidden_states = torch.randn(hidden_shape)

    # Convert to TTNN tensors
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(-2, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )

    tt_hidden_states = ttnn.from_torch(
        hidden_states.reshape(1, 1, -1, hidden_states.shape[-1]),
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
    )

    reference_attention = reference_layer.self_attn

    # Reference attention forward
    with torch.no_grad():
        reference_out, _ = reference_attention(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=mask,
            use_cache=True,
        )

    # TTNN attention forward (no mask needed, causal masking handled internally)
    attention_module = decoder_layer.self_attn
    tt_out = attention_module(
        tt_hidden_states,
        rope_mats=rope_mats,
        position_idx=tt_position_idx,
        page_table=None,
        kv_cache=None,
        is_decode=is_decode,
    )

    # Compare outputs
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
    tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=mesh_composer)[..., : batch_size * seq_len, :hidden_size]

    # Compare outputs
    passing, output = compare_tensors(tt_output_torch, reference_out, mesh_device, pcc_threshold=pcc_threshold)
    if passing:
        logger.info(f"Attention test passed. Output: {output}")
    else:
        assert passing, f"Attention test failed. Output: {output}"


def run_rms_norm_component(
    mesh_device, hidden_shape, reference_layer, decoder_layer, is_decode, is_row_sharded, pcc_threshold
):
    """Test RMSNorm component - extracted from decoder layer"""

    # Create input
    batch_size, seq_len, hidden_size = hidden_shape
    hidden_states = torch.randn(hidden_shape)

    # Extract reference RMSNorm from reference layer
    reference_rms_norm = reference_layer.input_layernorm

    # Reference RMSNorm forward
    with torch.no_grad():
        ref_output = reference_rms_norm(hidden_states)

    # Convert to TTNN tensors
    # tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )
    tt_hidden_states = ttnn.from_torch(
        hidden_states,
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
    )

    # TTNN RMSNorm forward
    rms_norm_module = decoder_layer.input_layernorm
    tt_output = rms_norm_module(tt_hidden_states)

    # Compare outputs
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape))
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)[:batch_size, :seq_len, :]
    # Compare outputs
    passing, output = compare_tensors(tt_output_torch, ref_output, mesh_device, pcc_threshold=pcc_threshold)
    if passing:
        logger.info(f"RMS Norm test passed. Output: {output}")
    else:
        assert passing, f"RMS Norm test failed. Output: {output}"


def run_topk_router_component(
    mesh_device, hidden_shape, reference_layer, decoder_layer, is_decode, is_row_sharded, pcc_threshold
):
    """Test TopK router component - extracted from decoder layer"""

    # Create input
    batch, seq_len, hidden_size = hidden_shape
    hidden_states = torch.randn(hidden_shape)

    # Extract reference TopK router from reference layer
    reference_router = reference_layer.mlp.router
    router_scores, router_indices = reference_router(hidden_states)
    if decoder_layer.mlp.use_throughput_experts:
        # When using throughput experts, we return a dense tensor of router_scores. Convert sparse reference router_scores to dense router_weights (note: this requires reorder the weights to match the order of the indices)
        dense_router_scores = torch.concat(
            [
                torch.tensor(
                    [router_scores[user, router_indices[user, i]] for i in range(router_indices.shape[1])]
                ).reshape(1, -1)
                for user in range(router_scores.shape[0])
            ],
            dim=0,
        )
        router_scores = dense_router_scores

    # Convert to TTNN tensors
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(-2, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )

    tt_hidden_states = ttnn.from_torch(
        hidden_states.reshape(1, 1, -1, 2880),
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    # Extract TT TopK router from decoder layer
    tt_router = decoder_layer.mlp.router
    tt_router_indices, tt_router_weights = tt_router(tt_hidden_states, decoder_layer.mlp.use_throughput_experts)
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape))
    tt_router_indices_torch = ttnn.to_torch(tt_router_indices, mesh_composer=mesh_composer)[:batch, :4]
    tt_router_weights_torch = ttnn.to_torch(tt_router_weights, mesh_composer=mesh_composer)[:batch, :4]

    # Compare outputs
    # We will sort the indices here as the order of the indices is not guaranteed to be the same in the reference and TT implementation.
    sorted_tt_indices, sorted_tt_indices_order = torch.sort(tt_router_indices_torch, dim=-1)
    sorted_ref_indices, sorted_ref_indices_order = torch.sort(router_indices, dim=-1)
    indices_passing, indices_output = compare_tensors(
        sorted_tt_indices, sorted_ref_indices, mesh_device, pcc_threshold=pcc_threshold
    )
    weights_passing, weights_output = compare_tensors(
        tt_router_weights_torch.squeeze()[
            sorted_tt_indices_order
        ],  # we have to squeeze here because it breaks the indexing otherwise
        router_scores.squeeze()[sorted_ref_indices_order],
        mesh_device,
        pcc_threshold=pcc_threshold,
    )
    if not (indices_passing and weights_passing):
        assert (
            False
        ), f"\nTopK Router test (indices) {indices_passing}. Output: {indices_output}\nTopK Router test (weights) {weights_passing}. Output: {weights_output}"
    else:
        logger.info(f"TopK Router indices test passed. Output: {indices_output}")
        logger.info(f"TopK Router weights test passed. Output: {weights_output}")


def run_throughput_experts_component(
    mesh_device, hidden_shape, config, reference_layer, decoder_layer, is_decode, is_row_sharded, pcc_threshold
):
    """Test experts component - extracted from decoder layer"""

    # Create input
    _, _, num_tokens, hidden_size = hidden_shape
    hidden_states = torch.randn(hidden_shape)

    router_indices = torch.zeros(num_tokens, config.num_experts_per_tok, dtype=torch.long)
    routing_weights = torch.zeros(num_tokens, config.num_local_experts)

    for t in range(num_tokens):
        active_experts = torch.randperm(config.num_local_experts)[: config.num_experts_per_tok]
        router_indices[..., t, :] = active_experts
        weights = torch.rand(config.num_experts_per_tok)
        weights = weights / weights.sum()  # Normalize
        routing_weights[..., t, active_experts] = weights
    topk_weights_dense = torch.tensor(
        [[routing_weights[..., i, j].item() for j in b] for i, b in enumerate(router_indices.squeeze())]
    )
    # Extract reference experts from reference layer
    reference_experts = reference_layer.mlp.experts.eval()  # Set to eval mode for inference
    reference_output = reference_experts(
        hidden_states, router_indices=router_indices.squeeze(), routing_weights=routing_weights.squeeze()
    )

    # Convert to TTNN tensors
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(-2, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )
    tt_hidden_states = ttnn.from_torch(
        hidden_states,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )
    tt_routing_weights = ttnn.from_torch(
        topk_weights_dense,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )
    tt_router_indices = ttnn.from_torch(
        router_indices,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint16,
        mesh_mapper=mesh_mapper,
    )

    # Extract TT experts from decoder layer
    tt_experts = decoder_layer.mlp.experts
    tt_output = tt_experts(
        hidden_states=tt_hidden_states,
        topk_expert_indices=tt_router_indices,
        topk_expert_weights=tt_routing_weights,
        is_decode=is_decode,
    )

    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
    tt_output = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)[..., :num_tokens, :hidden_size]
    # Compare outputs
    passing, output = compare_tensors(tt_output, reference_output, mesh_device, pcc_threshold=pcc_threshold)
    if passing:
        logger.info(f"High Throughput Experts test passed. Output: {output}")
    else:
        assert passing, f"High Throughput Experts test failed. Output: {output}"


def run_experts_component(mesh_device, hidden_shape, config, reference_layer, decoder_layer, is_decode, pcc_threshold):
    """Test experts component - extracted from decoder layer"""

    # Create input
    batch_size, seq_len, hidden_size = hidden_shape
    hidden_states = torch.randn(hidden_shape)
    import itertools

    router_indices = torch.zeros(batch_size * seq_len, config.num_experts_per_tok, dtype=torch.long)
    routing_weights = torch.zeros(batch_size * seq_len, config.num_local_experts)

    for b, s in itertools.product(range(batch_size), range(seq_len)):
        active_experts = torch.randperm(config.num_local_experts)[: config.num_experts_per_tok]
        router_indices[b * seq_len + s, :] = active_experts
        weights = torch.rand(config.num_experts_per_tok)
        weights = weights / weights.sum()  # Normalize
        routing_weights[b * seq_len + s, active_experts] = weights

    # Extract reference experts from reference layer
    reference_experts = reference_layer.mlp.experts.eval()  # Set to eval mode for inference
    reference_output = reference_experts(hidden_states, router_indices=router_indices, routing_weights=routing_weights)

    # Convert to TTNN tensors
    tt_hidden_states = ttnn.from_torch(
        hidden_states.unsqueeze(0),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(dims=(None, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device),
    )
    tt_routing_weights = ttnn.from_torch(
        routing_weights,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(dims=(None, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device),
    )

    # Extract TT experts from decoder layer
    tt_experts = decoder_layer.mlp.experts
    tt_output = tt_experts(
        hidden_states=tt_hidden_states,
        topk_expert_weights=tt_routing_weights,
        is_decode=is_decode,
    )
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
    tt_output = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)[..., : batch_size * seq_len, :hidden_size]
    # Compare outputs
    passing, output = compare_tensors(tt_output, reference_output, mesh_device, pcc_threshold=pcc_threshold)
    if passing:
        logger.info(f"Low Latency Experts test passed. Output: {output}")
    else:
        assert passing, f"Low Latency Experts test failed. Output: {output}"


def run_full_mlp_pipeline(
    mesh_device, hidden_shape, reference_layer, decoder_layer, is_decode, is_row_sharded, pcc_threshold
):
    """Test complete MLP (router + experts) - essential MoE functionality"""

    # Create input
    batch, seq_len, hidden_size = hidden_shape
    hidden_states = torch.randn(hidden_shape)

    reference_model = reference_layer.mlp
    reference_output, routing_scores = reference_model(hidden_states)

    # Convert to TTNN tensors
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )
    tt_hidden_states = ttnn.from_torch(
        # hidden_states.reshape(-1, 1, 2880),
        hidden_states.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
    )

    # Create TT MLP using TestFactory setup
    tt_mlp = decoder_layer.mlp
    tt_output = tt_mlp(tt_hidden_states, is_decode=is_decode)
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)[..., : batch * seq_len, :hidden_size]

    # Compare outputs
    passing, output = compare_tensors(tt_output_torch, reference_output, mesh_device, pcc_threshold=pcc_threshold)
    if passing:
        logger.info(f"MLP Pipeline test passed. Output: {output}")
    else:
        assert passing, f"MLP Pipeline test failed. Output: {output}"


def setup_reference_layer(setup, layer_idx=0):
    logger.info("Setting up reference layer...")
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer

    reference_layer = GptOssDecoderLayer(setup["config"], layer_idx=layer_idx)
    with torch.no_grad():
        for name, param in reference_layer.named_parameters():
            if any(proj in name for proj in ["router", "experts", "sinks"]):
                param.data.normal_(0, 1)
    return reference_layer


def setup_decoder_layer(setup, reference_layer, local_batch_size, seq_len, layer_idx=0):
    logger.info("Setting up TT decoder layer...")
    reference_state = reference_layer.state_dict()
    config = setup["config"]
    # Convert HF QKV weights to Meta format for RoPE compatibility
    reference_state_swizzled = convert_hf_qkv_to_meta_format(reference_state, config.head_dim)
    max_seq_len = getattr(config, "max_position_embeddings", 131072)
    rope_scaling = rope_scaling_model_factory(config.rope_scaling)
    rope_setup = RotarySetup(
        device=setup["mesh_device"],
        batch_size=1,
        head_dim=config.head_dim,
        max_seq_len=max_seq_len,
        rope_theta=getattr(config, "rope_theta", 10000.0),
        rope_scaling=rope_scaling,
        datatype=ttnn.bfloat16,
    )
    transformation_mats = rope_setup.get_both_trans_mats()
    decoder_layer = DecoderLayer(
        setup["mesh_device"],
        config,
        reference_state_swizzled,
        layer_idx=layer_idx,
        ccl_manager=setup["ccl_manager"],
        dtype=setup["dtype"],
        mesh_config=setup["mesh_config"],
        transformation_mats=transformation_mats,
        max_seq_len=max(seq_len, 128),
        max_local_batch_size=local_batch_size,
        use_throughput_experts=setup["mesh_device"].shape[0] > 1
        and local_batch_size * seq_len > 1,  # high throughput experts don't support single user decode currently
    )
    return decoder_layer


@parametrize_mesh_with_fabric()
@parametrize_batch_seq(
    [
        (1, 1),  # decode
        (32, 1),  # decode
        (128, 1),  # decode
        (1, 128),  # prefill
        (1, 4096),  # prefill 4k
    ],
    ids=[
        "decode_1",
        "decode_32",
        "decode_128",
        "prefill_128",
        "prefill_4096",
    ],
)
@pytest.mark.parametrize(
    "mesh_shape",
    [
        (1, 8),
        (4, 8),
    ],
    ids=[
        "mesh_1x8",
        "mesh_4x8",
    ],
)
@pytest.mark.parametrize(
    # We want to test the first two layers so we capture both sliding and global attention layers
    "layer_idx",
    [0, 1],
    ids=[
        "layer_0",
        "layer_1",
    ],
)
def test_decoder(
    mesh_device, device_params, batch_size, seq_len, mesh_shape, layer_idx, test_modules, test_thresholds, reset_seeds
):
    """
    Test decoder layer components.

    Args:
        test_modules: Which modules to test (from --test-modules flag). Options:
            - "all": Test all components (default)
            - "attention": Test attention only
            - "rms_norm": Test RMS normalization only
            - "router": Test TopK router only (decode mode only)
            - "mlp": Test full MLP pipeline (router + experts)
            - "decoder": Test full decoder layer only
            - Comma-separated: "attention,mlp" or "router,experts" etc.

    Usage:
        pytest test_modules.py  # runs all tests
        pytest test_modules.py --test-modules=attention
        pytest test_modules.py --test-modules=attention,mlp
    """
    if mesh_shape[0] == 1 and batch_size > 1:
        pytest.skip(
            f"Skipping batch size {batch_size} for mesh shape {mesh_shape}. Only batch size 1 is supported for mesh shape (1, 8)."
        )

    assert batch_size == 1 or seq_len == 1, "Only single user prefill or single token decode is supported"
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(mesh_shape))
    is_decode = seq_len == 1
    mode = "decode" if is_decode else "prefill"

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    pcc_thresholds = test_thresholds[setup["model_args"].model_name][mode]
    # Set attention implementation for transformers compatibility
    config = setup["config"]
    config._attn_implementation = "eager"

    if batch_size > 32:
        if mesh_shape[0] == 1:
            pytest.skip(f"Batch size > 32 is not supported for mesh shape {mesh_shape}")
        is_row_sharded = True
        assert (
            batch_size % setup["mesh_device"].shape[0] == 0
        ), "Batch size must be evenly divisible by mesh device shape"
        local_batch_size = batch_size // setup["mesh_device"].shape[0]
    else:
        is_row_sharded = False
        local_batch_size = batch_size

    # Create reference model
    reference_layer = setup_reference_layer(setup, layer_idx=layer_idx)
    decoder_layer = setup_decoder_layer(setup, reference_layer, local_batch_size, seq_len, layer_idx=layer_idx)

    # Create input
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Create position IDs first
    position_ids = torch.concat([torch.arange(seq_len, dtype=torch.long) for _ in range(batch_size)])

    # Create attention mask like the working attention test
    mask = torch.triu(torch.full((1, 1, seq_len, seq_len), -float("inf")), diagonal=1)
    if reference_layer.attention_type == "sliding_attention":
        mask += torch.tril(torch.full((1, 1, seq_len, seq_len), -float("inf")), diagonal=-config.sliding_window)

    # Handle decode mode for TT model like original
    if seq_len == 1:  # decode
        mask = None

    # Create RoPE embeddings using Meta format (matching tt-transformers test)
    max_seq_len = seq_len

    # For TTNN: use precompute_freqs and gather_cos_sin to get cos/sin tensors
    # TODO: To test longer sequences we will need to change this so we can apply rope scaling using Yarn implementation
    cos_full, sin_full = precompute_freqs(
        dim=config.head_dim,
        end=max_seq_len * 2,
        theta=config.rope_theta,
        scale_factor=None,
        orig_context_len=131072,
    )

    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

    rope_embeddings_ref = GptOssRotaryEmbedding(config)
    cos_hf_ref, sin_hf_ref = rope_embeddings_ref(hidden_states, position_ids.unsqueeze(1 if is_decode else 0))
    position_embeddings_ref = (cos_hf_ref, sin_hf_ref)

    # Create TTNN RoPE embeddings in Meta format using gather_cos_sin
    cos_meta, sin_meta = gather_cos_sin(position_ids, cos_full, sin_full)
    cos_meta = cos_meta.reshape(1, batch_size, seq_len, config.head_dim)
    sin_meta = sin_meta.reshape(1, batch_size, seq_len, config.head_dim)

    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(-3, None), mesh_shape=setup["mesh_device"].shape, mesh_device=setup["mesh_device"])
        if is_row_sharded
        else None
    )
    tt_cos = ttnn.from_torch(
        cos_meta, device=setup["mesh_device"], mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    tt_sin = ttnn.from_torch(
        sin_meta, device=setup["mesh_device"], mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    # For decode mode, convert cos/sin to HEIGHT_SHARDED to match Q/K/V from nlp_create_qkv_heads_decode
    if mode == "decode":
        grid_size = ttnn.CoreCoord(8, 8)  # Safe limit: max 8 per dimension to avoid Galaxy hangs
        batch_grid = ttnn.num_cores_to_corerangeset(local_batch_size, grid_size, row_wise=True)
        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, config.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        tt_cos = ttnn.interleaved_to_sharded(tt_cos, mem_config)
        tt_sin = ttnn.interleaved_to_sharded(tt_sin, mem_config)

    # rope_mats is now [cos, sin] in Meta format
    rope_mats = [tt_cos, tt_sin]

    # Create position index for TTNN
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=setup["mesh_device"].shape, mesh_device=setup["mesh_device"])
        if is_row_sharded
        else None
    )
    tt_position_idx = ttnn.from_torch(
        position_ids.squeeze(),
        device=setup["mesh_device"],
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )

    # Parse test_modules (supports comma-separated values)
    modules_to_test = set(test_modules.split(","))
    run_all = "all" in modules_to_test

    logger.info(f"Running tests: {test_modules}")

    # Helper to check if a module should be tested
    def should_test(module_name):
        return run_all or module_name in modules_to_test

    if should_test("router"):
        # Only run router test in decode mode
        if seq_len == 1:
            logger.info("Testing TopK Router...")
            run_topk_router_component(
                setup["mesh_device"],
                hidden_states.shape,
                reference_layer,
                decoder_layer,
                is_decode=is_decode,
                is_row_sharded=is_row_sharded,
                pcc_threshold=pcc_thresholds["router"],
            )
        else:
            logger.info("Router test only runs in decode mode (seq_len=1). Skipping...")

    if should_test("experts"):
        if decoder_layer.mlp.use_throughput_experts:
            logger.info(f"Testing High Throughput Experts (EP=32) for mesh shape {mesh_shape}...")
            hidden_states_throughput_experts = hidden_states.reshape(1, 1, batch_size * seq_len, -1)
            run_throughput_experts_component(
                setup["mesh_device"],
                hidden_states_throughput_experts.shape,
                config,
                reference_layer,
                decoder_layer,
                is_decode=is_decode,
                is_row_sharded=is_row_sharded,
                pcc_threshold=pcc_thresholds["experts"],
            )
        else:
            logger.info(f"Testing Low Throughput Experts (EP=4) for mesh shape {mesh_shape}...")
            run_experts_component(
                setup["mesh_device"],
                hidden_states.shape,
                config,
                reference_layer,
                decoder_layer,
                is_decode=is_decode,
                pcc_threshold=pcc_thresholds["experts"],
            )

    if should_test("attention"):
        logger.info("Testing Attention...")
        run_attention_component(
            setup["mesh_device"],
            hidden_states.shape,
            mask,
            position_embeddings_ref,
            rope_mats,
            tt_position_idx,
            reference_layer,
            decoder_layer,
            is_decode=is_decode,
            is_row_sharded=is_row_sharded,
            pcc_threshold=pcc_thresholds["attention"],
        )

    if should_test("rms_norm"):
        logger.info("Testing RMS Norm...")
        run_rms_norm_component(
            setup["mesh_device"],
            hidden_states.shape,
            reference_layer,
            decoder_layer,
            is_decode=is_decode,
            is_row_sharded=is_row_sharded,
            pcc_threshold=pcc_thresholds["rms_norm"],
        )

    if should_test("mlp"):
        logger.info("Testing Full MLP Pipeline...")
        run_full_mlp_pipeline(
            setup["mesh_device"],
            hidden_states.shape,
            reference_layer,
            decoder_layer,
            is_decode=is_decode,
            is_row_sharded=is_row_sharded,
            pcc_threshold=pcc_thresholds["mlp"],
        )

    if should_test("decoder"):
        logger.info("Testing Full Decoder Layer...")
        # Create TTNN tensors for decoder layer test
        if is_row_sharded:
            mesh_mapper = ttnn.ShardTensor2dMesh(
                dims=(-2, None), mesh_shape=setup["mesh_device"].shape, mesh_device=setup["mesh_device"]
            )
        else:
            mesh_mapper = None
        tt_hidden_states = ttnn.from_torch(
            hidden_states.reshape(1, 1, batch_size * seq_len, -1),
            device=setup["mesh_device"],
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
        )

        # Test full decoder layer integration
        with torch.no_grad():
            reference_output = reference_layer(hidden_states, position_embeddings=position_embeddings_ref)

        tt_output = decoder_layer(
            tt_hidden_states, position_embeddings=rope_mats, position_idx=tt_position_idx, is_decode=is_decode
        )

        # Compare outputs
        pcc_threshold = (pcc_thresholds["decoder"],)
        mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
        tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)[
            ..., : batch_size * seq_len, : config.hidden_size
        ]
        passing, output = compare_tensors(
            tt_output_torch.squeeze(), reference_output.squeeze(), mesh_device, pcc_threshold=pcc_threshold
        )
        if passing:
            logger.info(f"Decoder Layer test passed. Output: {output}")
        else:
            assert passing, f"Decoder Layer test failed. Output: {output}"

    tested_modules = [m for m in modules_to_test if m != "router" or seq_len == 1]
    logger.info(f"✓ Tests completed successfully: {', '.join(tested_modules)}")


def run_model_forward_test(
    mesh_device,
    config,
    state_dict_meta,
    state_dict_hf,
    mesh_config,
    batch_size,
    seq_len,
    is_decode,
    pcc_threshold=0.88,
):
    """
    Run a single forward pass test comparing TT model to reference model.

    Args:
        mesh_device: TTNN mesh device
        config: HuggingFace config (with num_hidden_layers already modified)
        state_dict_meta: Model weights in meta format for TT model
        state_dict_hf: Model weights in HF format for reference model
        mesh_config: Mesh configuration
        batch_size: Batch size
        seq_len: Sequence length
        is_decode: True for decode mode (seq_len=1), False for prefill mode
        pcc_threshold: PCC threshold for comparison
    """
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    from models.demos.gpt_oss.tt.ccl import CCLManager
    from models.demos.gpt_oss.tt.model import Model

    # Determine local batch size for row sharding
    if batch_size > 32:
        is_row_sharded = True
        assert batch_size % mesh_device.shape[0] == 0, "Batch size must be divisible by mesh rows"
        local_batch_size = batch_size // mesh_device.shape[0]
    else:
        is_row_sharded = False
        local_batch_size = batch_size

    # Create CCL manager
    ccl_manager = CCLManager(mesh_device)

    # Create TT model with meta format weights
    # Use throughput experts for row-sharded batches (batch > 32 on multi-row mesh)
    use_throughput_experts = is_row_sharded and mesh_device.shape[0] > 1
    tt_model = Model(
        mesh_device=mesh_device,
        hf_config=config,
        state_dict=state_dict_meta,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        paged_attention_config=None,
        mesh_config=mesh_config,
        create_kv_cache=True,
        max_local_batch_size=local_batch_size,
        users_row_sharded=is_row_sharded,
        use_throughput_experts=use_throughput_experts,
    )

    # Create reference model with HF format weights
    reference_model = GptOssForCausalLM(config)
    reference_model.load_state_dict(state_dict_hf, strict=False)
    reference_model.eval()

    # Create random input tokens
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Create position IDs
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    # Run reference forward pass
    with torch.no_grad():
        reference_output = reference_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,  # Let the model handle masking
            use_cache=False,
        )
        reference_logits = reference_output.logits  # [batch_size, seq_len, vocab_size]

    # Prepare inputs for TT model
    if is_decode:
        # Decode mode: use ttnn_decode_forward
        # Flatten tokens for decode
        tokens_flat = input_ids.reshape(-1)  # [batch_size]
        current_pos = torch.zeros(batch_size, dtype=torch.long)

        # Prepare inputs using model's method
        tt_tokens, tt_current_pos, tt_rope_idxs, _ = tt_model.prepare_inputs_decode(
            tokens_flat, current_pos, page_table=None
        )

        # Run TT decode forward
        tt_logits, _ = tt_model.ttnn_decode_forward(
            tokens=tt_tokens,
            current_pos=tt_current_pos,
            rot_mat_idxs=tt_rope_idxs,
            page_table=None,
            kv_cache=None,
        )
    else:
        # Prefill mode: use ttnn_prefill_forward
        # Embed tokens first
        tt_tokens = ttnn.from_torch(
            input_ids.unsqueeze(0).unsqueeze(0),  # [1, 1, batch, seq]
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        tt_embeds = ttnn.embedding(tt_tokens, tt_model.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        if len(tt_embeds.shape) == 3:
            tt_embeds = ttnn.unsqueeze_to_4D(tt_embeds)

        # Run TT prefill forward
        tt_logits = tt_model.ttnn_prefill_forward(
            x=tt_embeds,
            user_id=0,
            rot_mats_global=None,  # Let model compute RoPE
            page_table=None,
            kv_cache=None,
            get_last_token=-1,  # Get all tokens
        )

    # Convert TT output to torch
    mesh_composer_dims = (-2, 1) if is_row_sharded else (0, 1)
    tt_logits_torch = ttnn.to_torch(
        tt_logits,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=mesh_composer_dims, mesh_shape=tuple(mesh_device.shape)
        ),
    )

    # Slice to match reference shape
    tt_logits_torch = tt_logits_torch[0, 0]

    # Reshape to match reference
    tt_logits_torch = tt_logits_torch.reshape(batch_size, seq_len, -1)

    # Compare outputs
    passing, output = compare_tensors(tt_logits_torch, reference_logits, mesh_device, pcc_threshold=pcc_threshold)
    return passing, output


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize(
    "batch_size, seq_len, mode",
    [
        (1, 128, "prefill"),  # Prefill test: batch=1, seq=128
        (1, 1, "decode"),  # Decode test: batch=128, seq=1 - disabled due to breakpoint in model.py
        (128, 1, "decode"),  # Decode test: batch=128, seq=1 - disabled due to breakpoint in model.py
    ],
    ids=[
        "prefill_b1_s128",
        "decode_b1_s1",
        "decode_b128_s1",
    ],
)
@pytest.mark.parametrize(
    "mesh_shape",
    [
        (4, 8),
    ],
)
@pytest.mark.parametrize(
    "num_layers",
    [1, 5],
    ids=[
        "1_layer",
        "5_layers",
    ],
)
def test_model(mesh_device, device_params, batch_size, seq_len, mode, mesh_shape, num_layers, reset_seeds):
    """
    Test full model forward pass comparing TT implementation to HuggingFace reference.

    This test:
    1. Loads model config and overrides num_hidden_layers
    2. Creates both TT model and reference model with real weights
    3. Runs prefill (batch=1, seq=128) or decode (batch=128, seq=1) forward pass
    4. Compares outputs using PCC

    Args:
        mesh_device: TTNN mesh device fixture
        device_params: Device parameters fixture
        batch_size: Batch size for the test
        seq_len: Sequence length for the test
        mode: "prefill" or "decode"
        mesh_shape: Mesh shape tuple
        num_layers: Number of layers to use (overrides config.num_hidden_layers)
        reset_seeds: Fixture to reset random seeds
    """
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.model_config import ModelArgs

    is_decode = mode == "decode"

    # Create submesh with specified shape
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(mesh_shape))

    # Setup test using TestFactory
    setup = TestFactory.setup_test(mesh_device, use_real_weights=True)
    config = setup["config"]

    # Override number of layers
    original_num_layers = config.num_hidden_layers
    config.num_hidden_layers = num_layers
    logger.info(f"Overriding num_hidden_layers from {original_num_layers} to {num_layers}")

    # Set attention implementation
    config._attn_implementation = "eager"

    # Create mesh config
    mesh_config = MeshConfig(mesh_shape, decode=ModeConfig(tp=mesh_shape[1], ep=mesh_shape[0]))

    # Load state dict in HF format for reference model
    model_args = ModelArgs(mesh_device=mesh_device, dummy_weights=False)
    state_dict_hf = model_args.load_state_dict(
        weights_path=model_args.model_path,
        dummy_weights=False,
        convert_to_meta_format=False,  # HF format for reference
    )

    # Convert to meta format for TT model
    state_dict_meta = convert_hf_qkv_to_meta_format(state_dict_hf, config.head_dim)

    logger.info(f"Running {mode} test with batch_size={batch_size}, seq_len={seq_len}, num_layers={num_layers}")

    # Run the forward test
    passing, output = run_model_forward_test(
        mesh_device=mesh_device,
        config=config,
        state_dict_meta=state_dict_meta,
        state_dict_hf=state_dict_hf,
        mesh_config=mesh_config,
        batch_size=batch_size,
        seq_len=seq_len,
        is_decode=is_decode,
        pcc_threshold=0.95 if num_layers == 1 else 0.85,  # Use slightly lower threshold for full model
    )

    if passing:
        logger.info(f"✓ Model {mode} test passed. PCC: {output}")
    else:
        assert passing, f"Model {mode} test failed. PCC: {output}"
