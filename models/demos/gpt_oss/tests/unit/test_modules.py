# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import precompute_freqs_cis
from models.tt_transformers.tt.common import gather_cos_sin, precompute_freqs
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
):
    """Test attention component - extracted from decoder layer"""

    # Create input
    hidden_states = torch.randn(hidden_shape)
    batch_size, seq_len, hidden_size = hidden_states.shape

    # Convert to TTNN tensors
    # tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
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
            # hidden_states=hidden_states.reshape(-1, 1, hidden_states.shape[-1]),
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
    # passing, output = run_component_comparison(tt_out, reference_out, mesh_device, pcc_threshold=0.96)
    # assert passing, f"Attention test failed. Output: {output}"
    tt_output_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )
    if is_row_sharded:
        # we sharded the input so we want to keep all rows but slice the replicated columns
        tt_output_torch = tt_output_torch[0]
    else:
        # inputs were replicated along the rows so we want to slice the replicated columns to get single output
        tt_output_torch = tt_output_torch[0, : seq_len * batch_size, :]

    # Compare outputs
    passing, output = compare_tensors(tt_output_torch, reference_out, mesh_device, pcc_threshold=0.96)
    if passing:
        logger.info(f"Attention test passed. Output: {output}")
    else:
        assert passing, f"Attention test failed. Output: {output}"


def run_rms_norm_component(mesh_device, hidden_shape, reference_layer, decoder_layer):
    """Test RMSNorm component - extracted from decoder layer"""

    # Create input
    hidden_states = torch.randn(hidden_shape)

    # Extract reference RMSNorm from reference layer
    reference_rms_norm = reference_layer.input_layernorm

    # Reference RMSNorm forward
    with torch.no_grad():
        ref_output = reference_rms_norm(hidden_states)

    # Convert to TTNN tensors
    # tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    tt_hidden_states = ttnn.from_torch(
        hidden_states.reshape(-1, 1, 2880),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
    )

    # TTNN RMSNorm forward
    rms_norm_module = decoder_layer.input_layernorm
    tt_output = rms_norm_module(tt_hidden_states)

    # Compare outputs
    # passing, output = run_component_comparison(tt_output, ref_output, mesh_device, pcc_threshold=0.99)
    # assert passing, f"RMS norm test failed. Output: {output}"
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )[:, :1, :]

    # Compare outputs
    passing, output = compare_tensors(tt_output_torch, ref_output, mesh_device, pcc_threshold=0.99)
    if passing:
        logger.info(f"Experts test passed. Output: {output}")
    else:
        assert passing, f"Experts test failed. Output: {output}"


def run_topk_router_component(mesh_device, hidden_shape, reference_layer, decoder_layer, is_decode, is_row_sharded):
    """Test TopK router component - extracted from decoder layer"""

    # Create input
    hidden_states = torch.randn(hidden_shape)

    # Extract reference TopK router from reference layer
    reference_router = reference_layer.mlp.router
    router_scores, router_indices = reference_router(hidden_states)
    # We return a dense tensor of router_scores. Convert sparse router_scores to dense router_weights (note: this requires reorder the weights to match the order of the indices)
    router_weights = torch.concat(
        [
            torch.tensor(
                [router_scores[user, router_indices[user, i]] for i in range(router_indices.shape[1])]
            ).reshape(1, -1)
            for user in range(router_scores.shape[0])
        ],
        dim=0,
    )

    # Convert to TTNN tensors
    # tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )

    tt_hidden_states = ttnn.from_torch(
        hidden_states.reshape(-1, 1, 2880),
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
    )

    # Extract TT TopK router from decoder layer
    tt_router = decoder_layer.mlp.router
    tt_router_indices, tt_router_weights = tt_router(tt_hidden_states)
    tt_router_indices_torch = ttnn.to_torch(
        tt_router_indices,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )[:, :4]
    tt_router_weights_torch = ttnn.to_torch(
        tt_router_weights,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )[:, :4]

    # Compare outputs
    # We will sort the indices here as the order of the indices is not guaranteed to be the same in the reference and TT implementation.
    sorted_tt_indices, sorted_tt_indices_order = torch.sort(tt_router_indices_torch, dim=-1)
    sorted_ref_indices, sorted_ref_indices_order = torch.sort(router_indices, dim=-1)
    indices_passing, indices_output = compare_tensors(
        sorted_tt_indices, sorted_ref_indices, mesh_device, pcc_threshold=0.98
    )
    weights_passing, weights_output = compare_tensors(
        tt_router_weights_torch[sorted_tt_indices_order],
        router_weights[sorted_ref_indices_order],
        mesh_device,
        pcc_threshold=0.98,
    )
    if not (indices_passing and weights_passing):
        assert (
            False
        ), f"\nTopK Router test failed (indices). Output: {indices_output}\nTopK Router test failed (weights). Output: {weights_output}"
    else:
        logger.info(f"TopK Router indices test passed. Output: {indices_output}")
        logger.info(f"TopK Router weights test passed. Output: {weights_output}")
    # for tt_output, reference_output in zip(tt_router_scores, router_scores):
    #     passing, output = run_component_comparison(tt_output, reference_output, mesh_device, pcc_threshold=0.945)
    #     assert passing, f"TopK router test failed. Output: {output}"


def run_experts_component(mesh_device, hidden_shape, config, reference_layer, decoder_layer, is_decode, is_row_sharded):
    """Test experts component - extracted from decoder layer"""

    # Create input
    # hidden_shape = torch.Size([128, 1, 2880])
    hidden_states = torch.randn(hidden_shape)
    seq_len = hidden_shape[1]
    batch_size = hidden_shape[0]
    # Choose routing based on seq_len (sparse for seq_len=1, dense for seq_len>1)
    # if seq_len == 1:
    # Sparse routing
    import itertools

    router_indices = torch.zeros(batch_size * seq_len, config.num_experts_per_tok, dtype=torch.long)
    routing_weights = torch.zeros(batch_size * seq_len, config.num_local_experts)

    for b, s in itertools.product(range(batch_size), range(seq_len)):
        active_experts = torch.randperm(config.num_local_experts)[: config.num_experts_per_tok]
        router_indices[b * seq_len + s, :] = active_experts
        weights = torch.rand(config.num_experts_per_tok)
        weights = weights / weights.sum()  # Normalize
        routing_weights[b * seq_len + s, active_experts] = weights
    # else:
    #     # Dense routing
    #     routing_weights = torch.ones(hidden_states.shape[-2], config.num_local_experts) / config.num_local_experts
    topk_weights_dense = torch.tensor([[routing_weights[i, j].item() for j in b] for i, b in enumerate(router_indices)])
    # Extract reference experts from reference layer
    reference_experts = reference_layer.mlp.experts.eval()  # Set to eval mode for inference
    reference_output = reference_experts(hidden_states, router_indices=router_indices, routing_weights=routing_weights)

    # Convert to TTNN tensors
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )
    tt_hidden_states = ttnn.from_torch(
        hidden_states.unsqueeze(1),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )
    tt_routing_weights = ttnn.from_torch(
        topk_weights_dense.unsqueeze(1).unsqueeze(1),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )
    tt_router_indices = ttnn.from_torch(
        router_indices.unsqueeze(1).unsqueeze(1),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint16,
        mesh_mapper=mesh_mapper,
    )

    # Extract TT experts from decoder layer
    # tt_experts = decoder_layer.mlp.experts
    tt_experts = decoder_layer.mlp.throughput_experts
    tt_output = tt_experts(tt_hidden_states, tt_router_indices, tt_routing_weights, is_decode)

    tt_output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )
    if is_row_sharded:
        # we sharded the input so we want to keep all rows but slice the replicated columns
        tt_output = tt_output[0]
    else:
        # inputs were replicated along the rows so we want to slice the replicated columns to get single output
        tt_output = tt_output[0, ..., : seq_len * batch_size, :]
    # Compare outputs
    # passing, output = run_component_comparison(tt_output, reference_output, mesh_device, pcc_threshold=0.93)
    passing, output = compare_tensors(tt_output, reference_output, mesh_device, pcc_threshold=0.93)
    if passing:
        logger.info(f"Experts test passed. Output: {output}")
    else:
        assert passing, f"Experts test failed. Output: {output}"


def run_full_mlp_pipeline(mesh_device, hidden_shape, reference_layer, decoder_layer, is_decode, is_row_sharded):
    """Test complete MLP (router + experts) - essential MoE functionality"""

    # Create input
    hidden_states = torch.randn(hidden_shape)
    batch_size, seq_len, hidden_size = hidden_states.shape

    reference_model = reference_layer.mlp
    reference_output, routing_scores = reference_model(hidden_states)

    # Convert to TTNN tensors
    # tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )
    tt_hidden_states = ttnn.from_torch(
        hidden_states.reshape(-1, 1, 2880),
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
    )

    # Create TT MLP using TestFactory setup
    tt_mlp = decoder_layer.mlp
    tt_output = tt_mlp(tt_hidden_states, is_decode=is_decode)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )
    if is_row_sharded:
        # we sharded the input so we want to keep all rows but slice the replicated columns
        tt_output_torch = tt_output_torch[0]
    else:
        # inputs were replicated along the rows so we want to slice the replicated columns to get single output
        tt_output_torch = tt_output_torch[0, ..., : seq_len * batch_size, :]

    # Compare outputs
    passing, output = compare_tensors(tt_output_torch, reference_output, mesh_device, pcc_threshold=0.88)
    if passing:
        logger.info(f"MLP Pipeline test passed. Output: {output}")
    else:
        assert passing, f"MLP Pipeline test failed. Output: {output}"
    # passing, output = run_component_comparison(tt_output, reference_output, mesh_device, pcc_threshold=0.88)
    # assert passing, f"MLP test failed. Output: {output}"


@parametrize_mesh_with_fabric()
@parametrize_batch_seq(
    [
        # (1, 1),  # decode
        # (32, 1),  # decode
        (128, 1),  # decode
        # (1, 128),  # prefill
        # (1, 4096),  # prefill 4k
    ],
)
@pytest.mark.parametrize(
    "mesh_shape",
    [
        (4, 8),
    ],
)
def test_decoder(mesh_device, device_params, batch_size, seq_len, mesh_shape, test_modules, reset_seeds):
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
    assert batch_size == 1 or seq_len == 1, "Only single user prefill or single token decode is supported"
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(mesh_shape))
    is_decode = seq_len == 1

    setup = TestFactory.setup_test(mesh_device, use_real_weights=True)
    config = setup["config"]

    # Set attention implementation for transformers compatibility
    config._attn_implementation = "eager"

    if batch_size > 32:
        is_row_sharded = True
        assert (
            batch_size % setup["mesh_device"].shape[0] == 0
        ), "Batch size must be evenly divisible by mesh device shape"
        local_batch_size = batch_size // setup["mesh_device"].shape[0]
    else:
        is_row_sharded = False
        local_batch_size = batch_size

    # Create reference model
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer

    reference_layer = GptOssDecoderLayer(config, layer_idx=0)

    with torch.no_grad():
        for name, param in reference_layer.named_parameters():
            if any(proj in name for proj in ["router", "experts", "sinks"]):
                param.data.normal_(0, 1)

    reference_state = reference_layer.state_dict()
    # Convert HF QKV weights to Meta format for RoPE compatibility
    reference_state_swizzled = convert_hf_qkv_to_meta_format(reference_state, config.head_dim)

    # Setup RoPE using tt-transformers RotarySetup (handles cos/sin and transformation matrices)
    max_seq_len = getattr(config, "max_position_embeddings", 131072)
    rope_setup = RotarySetup(
        device=setup["mesh_device"],
        batch_size=1,
        head_dim=config.head_dim,
        max_seq_len=max_seq_len,
        rope_theta=getattr(config, "rope_theta", 10000.0),
        rope_scaling=None,
        datatype=ttnn.bfloat16,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    decoder_layer = DecoderLayer(
        setup["mesh_device"],
        config,
        reference_state_swizzled,
        layer_idx=0,
        ccl_manager=setup["ccl_manager"],
        dtype=setup["dtype"],
        mesh_config=setup["mesh_config"],
        transformation_mats=transformation_mats,
        max_local_batch_size=local_batch_size,
    )

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

    # For reference: use precompute_freqs_cis and index by positions (like tt-transformers)
    # position_ids_1d = position_ids.squeeze()
    freqs_cis_full = precompute_freqs_cis(
        dim=config.head_dim,
        end=max_seq_len * 2,
        theta=config.rope_theta,
    )

    # For TTNN: use precompute_freqs and gather_cos_sin to get cos/sin tensors
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
    with torch.no_grad():
        reference_output = reference_layer(hidden_states, position_embeddings=position_embeddings_ref)

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
    if seq_len == 1:
        grid_size = setup["mesh_device"].compute_with_storage_grid_size()
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
    # tt_position_idx = ttnn.from_torch(
    #     position_ids, device=setup["mesh_device"], layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
    # )
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
        if seq_len == 1:
            logger.info("Testing TopK Router...")
            run_topk_router_component(
                setup["mesh_device"],
                hidden_states.shape,
                reference_layer,
                decoder_layer,
                is_decode=is_decode,
                is_row_sharded=is_row_sharded,
            )
        elif "router" in modules_to_test:
            pytest.skip("Router test only runs in decode mode (seq_len=1)")

    if should_test("experts"):
        logger.info("Testing Experts...")
        run_experts_component(
            setup["mesh_device"],
            hidden_states.shape,
            config,
            reference_layer,
            decoder_layer,
            is_decode=is_decode,
            is_row_sharded=is_row_sharded,
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
        )

    if should_test("decoder"):
        logger.info("Testing Full Decoder Layer...")
        # Create TTNN tensors for decoder layer test
        # tt_hidden_states = ttnn.from_torch(
        #     hidden_states, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b
        # )
        # hidden_states = torch.reshape(hidden_states, (1, 1, batch_size*seq_len, -1))
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
        for _ in range(100):
            # Test full decoder layer integration
            tt_output = decoder_layer(
                tt_hidden_states, position_embeddings=rope_mats, position_idx=tt_position_idx, is_decode=is_decode
            )

            # Compare outputs
            pcc_threshold = 0.916 if seq_len == 1 else 0.88
            tt_output_torch = ttnn.to_torch(
                tt_output,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
            )
            if is_row_sharded:
                tt_output_torch = tt_output_torch[0]
            else:
                tt_output_torch = tt_output_torch[0, ..., : seq_len * batch_size, :]
            passing, output = compare_tensors(
                tt_output_torch.squeeze(), reference_output.squeeze(), mesh_device, pcc_threshold=pcc_threshold
            )
            if passing:
                logger.info(f"Decoder Layer test passed. Output: {output}")
            else:
                assert passing, f"Decoder Layer test failed. Output: {output}"
        # passing, output = run_component_comparison(
        #     tt_output, reference_output, setup["mesh_device"], pcc_threshold=pcc_threshold
        # )
        # logger.info(f"Decoder layer test: {passing} with output: {output}")
        # assert passing, f"Decoder layer test failed. Output: {output}"

    tested_modules = [m for m in modules_to_test if m != "router" or seq_len == 1]
    logger.info(f"✓ Tests completed successfully: {', '.join(tested_modules)}")
