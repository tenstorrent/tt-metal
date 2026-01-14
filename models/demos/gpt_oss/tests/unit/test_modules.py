# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os

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
):
    """Test attention component - extracted from decoder layer"""

    # Create input
    hidden_states = torch.randn(hidden_shape)

    # Convert to TTNN tensors
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    reference_attention = reference_layer.self_attn

    # Reference attention forward
    reference_out, _ = reference_attention(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=mask,
        use_cache=True,
    )

    # TTNN attention forward (no mask needed, causal masking handled internally)
    attention_module = decoder_layer.self_attn
    tt_out = attention_module(tt_hidden_states, rope_mats, tt_position_idx)

    # Compare outputs
    passing, output = run_component_comparison(tt_out, reference_out, mesh_device, pcc_threshold=0.95)
    logger.info(f"Attention test: {'passed' if passing else 'failed'} with output: {output}")
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
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    # TTNN RMSNorm forward
    rms_norm_module = decoder_layer.input_layernorm
    tt_output = rms_norm_module(tt_hidden_states)

    # Compare outputs
    passing, output = run_component_comparison(tt_output, ref_output, mesh_device, pcc_threshold=0.99)
    logger.info(f"RMS norm test: {'passed' if passing else 'failed'} with output: {output}")
    assert passing, f"RMS norm test failed. Output: {output}"


def run_topk_router_component(mesh_device, hidden_shape, reference_layer, decoder_layer):
    """Test TopK router component - extracted from decoder layer"""

    # Create input
    hidden_states = torch.randn(hidden_shape)

    # Extract reference TopK router from reference layer
    reference_router = reference_layer.mlp.router
    router_scores, router_indices = reference_router(hidden_states)

    # Convert to TTNN tensors
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    # Extract TT TopK router from decoder layer
    tt_router = decoder_layer.mlp.router
    tt_router_scores, tt_router_indices, tt_router_logits = tt_router(tt_hidden_states)

    # Compare outputs
    for tt_output, reference_output in zip(tt_router_scores, router_scores):
        passing, output = run_component_comparison(tt_output, reference_output, mesh_device, pcc_threshold=0.945)
        logger.info(f"TopK router test: {'passed' if passing else 'failed'} with output: {output}")
        assert passing, f"TopK router test failed. Output: {output}"


def run_experts_component(mesh_device, hidden_shape, config, reference_layer, decoder_layer):
    """Test experts component - extracted from decoder layer"""

    # Create input
    hidden_states = torch.randn(hidden_shape)
    seq_len = hidden_shape[1]
    batch_size = hidden_shape[0]
    # Choose routing based on seq_len (sparse for seq_len=1, dense for seq_len>1)
    if seq_len == 1:
        # Sparse routing
        import itertools

        routing_weights = torch.zeros(batch_size * seq_len, config.num_local_experts)

        for b, s in itertools.product(range(batch_size), range(seq_len)):
            active_experts = torch.randperm(config.num_local_experts)[: config.num_experts_per_tok]
            weights = torch.rand(config.num_experts_per_tok)
            weights = weights / weights.sum()  # Normalize
            routing_weights[b * seq_len + s, active_experts] = weights
    else:
        # Dense routing
        routing_weights = torch.ones(hidden_states.shape[-2], config.num_local_experts) / config.num_local_experts
    # Extract reference experts from reference layer
    reference_experts = reference_layer.mlp.experts.eval()  # Set to eval mode for inference
    reference_output = reference_experts(hidden_states, routing_weights=routing_weights)

    # Convert to TTNN tensors
    tt_hidden_states = ttnn.from_torch(
        hidden_states,
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
    tt_output = tt_experts(tt_hidden_states, tt_routing_weights)
    # Compare outputs
    passing, output = run_component_comparison(tt_output, reference_output, mesh_device, pcc_threshold=0.93)
    logger.info(f"Experts test: {'passed' if passing else 'failed'} with output: {output}")
    assert passing, f"Experts test failed. Output: {output}"


def run_full_mlp_pipeline(mesh_device, hidden_shape, reference_layer, decoder_layer):
    """Test complete MLP (router + experts) - essential MoE functionality"""

    # Create input
    hidden_states = torch.randn(hidden_shape)

    reference_model = reference_layer.mlp
    reference_output, routing_scores = reference_model(hidden_states)

    # Convert to TTNN tensors
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    # Create TT MLP using TestFactory setup
    tt_mlp = decoder_layer.mlp
    tt_output, routing_scores = tt_mlp(tt_hidden_states)

    # Compare outputs
    passing, output = run_component_comparison(tt_output, reference_output, mesh_device, pcc_threshold=0.88)

    logger.info(f"MLP test: {'passed' if passing else 'failed'} with output: {output}")
    assert passing, f"MLP test failed. Output: {output}"


@parametrize_mesh_with_fabric()
@parametrize_batch_seq(
    [
        (1, 1),  # decode
        (1, 128),  # prefill
        # (1, 4096),  # prefill 4k TODO: Disabling until #35313 is resolved - instability observed
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
    if mesh_shape[0] == 1 and seq_len > 128 and os.environ.get("CI"):
        pytest.skip("Skip test for mesh_shape[0] == 1 and seq_len > 128 in CI due to known issue (see #35313).")

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(mesh_shape))

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    model_name = getattr(setup["model_args"], "model_name", None)

    config = setup["config"]

    # Set attention implementation for transformers compatibility
    config._attn_implementation = "eager"

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
        tensor_cache_path=setup["tensor_cache_path"] / "module_tests_",
    )

    # Create input
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Create position IDs first
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

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
    position_ids_1d = position_ids.squeeze(0)
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
    cos_hf_ref, sin_hf_ref = rope_embeddings_ref(hidden_states, position_ids)
    position_embeddings_ref = (cos_hf_ref, sin_hf_ref)
    with torch.no_grad():
        reference_output = reference_layer(hidden_states, position_embeddings=position_embeddings_ref)

    # Create TTNN RoPE embeddings in Meta format using gather_cos_sin
    cos_meta, sin_meta = gather_cos_sin(position_ids_1d, cos_full, sin_full)

    tt_cos = ttnn.from_torch(cos_meta, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_sin = ttnn.from_torch(sin_meta, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # For decode mode, convert cos/sin to HEIGHT_SHARDED to match Q/K/V from nlp_create_qkv_heads_decode
    if seq_len == 1:
        grid_size = setup["mesh_device"].compute_with_storage_grid_size()
        batch_grid = ttnn.num_cores_to_corerangeset(1, grid_size, row_wise=True)

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
    tt_position_idx = ttnn.from_torch(
        position_ids, device=setup["mesh_device"], layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
    )

    # Create TTNN tensors for component tests
    tt_hidden_states = ttnn.from_torch(
        hidden_states, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b
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
            run_topk_router_component(setup["mesh_device"], hidden_states.shape, reference_layer, decoder_layer)
        elif "router" in modules_to_test:
            pytest.skip("Router test only runs in decode mode (seq_len=1)")

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
        )

    if should_test("rms_norm"):
        logger.info("Testing RMS Norm...")
        run_rms_norm_component(setup["mesh_device"], hidden_states.shape, reference_layer, decoder_layer)

    if should_test("mlp"):
        logger.info("Testing Full MLP Pipeline...")
        run_full_mlp_pipeline(setup["mesh_device"], hidden_states.shape, reference_layer, decoder_layer)

    if should_test("decoder"):
        logger.info("Testing Full Decoder Layer...")
        # Test full decoder layer integration
        tt_output = decoder_layer(tt_hidden_states, position_embeddings=rope_mats, position_idx=tt_position_idx)

        # Compare outputs
        pcc_threshold = 0.924 if seq_len == 1 else 0.88
        passing, output = run_component_comparison(
            tt_output, reference_output, setup["mesh_device"], pcc_threshold=pcc_threshold
        )
        logger.info(f"Decoder layer test: {'passed' if passing else 'failed'} with output: {output}")
        assert passing, f"Decoder layer test failed. Output: {output}"

    tested_modules = [m for m in modules_to_test if m != "router" or seq_len == 1]
    logger.info(f"✓ Tests completed successfully: {', '.join(tested_modules)}")
