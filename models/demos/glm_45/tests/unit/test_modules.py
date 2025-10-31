# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

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
    tt_mask,
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
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Extract reference attention from reference layer
    reference_attention = reference_layer.self_attn

    # Reference attention forward
    reference_out, _ = reference_attention(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=mask,
        use_cache=True,
    )

    # TTNN attention forward
    attention_module = decoder_layer.self_attn
    tt_out = attention_module(tt_hidden_states, tt_mask, rope_mats, tt_position_idx)

    # Compare outputs
    passing, output = run_component_comparison(tt_out, reference_out, mesh_device, pcc_threshold=0.99)
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
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # TTNN RMSNorm forward
    rms_norm_module = decoder_layer.input_layernorm
    tt_output = rms_norm_module(tt_hidden_states)

    # Compare outputs
    passing, output = run_component_comparison(tt_output, ref_output, mesh_device, pcc_threshold=0.99)
    assert passing, f"RMS norm test failed. Output: {output}"


def run_topk_router_component(mesh_device, hidden_shape, reference_layer, decoder_layer):
    """Test TopK router component - extracted from decoder layer"""

    # Create input
    hidden_states = torch.randn(hidden_shape)

    # Extract reference TopK router from reference layer
    reference_router = reference_layer.mlp.router
    router_scores, router_indices = reference_router(hidden_states)

    # Convert to TTNN tensors
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Extract TT TopK router from decoder layer
    tt_router = decoder_layer.mlp.router
    tt_router_scores, tt_router_indices, tt_router_logits = tt_router(tt_hidden_states)

    # Compare outputs
    for tt_output, reference_output in zip(tt_router_scores, router_scores):
        passing, output = run_component_comparison(tt_output, reference_output, mesh_device, pcc_threshold=0.945)
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
    assert passing, f"Experts test failed. Output: {output}"


def run_full_mlp_pipeline(mesh_device, hidden_shape, reference_layer, decoder_layer):
    """Test complete MLP (router + experts) - essential MoE functionality"""

    # Create input
    hidden_states = torch.randn(hidden_shape)

    reference_model = reference_layer.mlp
    reference_output, routing_scores = reference_model(hidden_states)

    # Convert to TTNN tensors
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Create TT MLP using TestFactory setup
    tt_mlp = decoder_layer.mlp
    tt_output, routing_scores = tt_mlp(tt_hidden_states)

    # Compare outputs
    passing, output = run_component_comparison(tt_output, reference_output, mesh_device, pcc_threshold=0.88)
    assert passing, f"MLP test failed. Output: {output}"


@parametrize_mesh_with_fabric()
@parametrize_batch_seq(
    [
        (1, 1),
        (1, 128),
    ]
)
@pytest.mark.parametrize("mesh_shape", [(1, 4)])
def test_decoder(mesh_device, device_params, batch_size, seq_len, mesh_shape, reset_seeds):
    """Test complete decoder layer - combines attention + MLP + norms"""
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(mesh_shape))

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]

    # Set attention implementation for transformers compatibility
    config._attn_implementation = "eager"

    # Create reference model
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeDecoderLayer

    reference_layer = Glm4MoeDecoderLayer(config, layer_idx=0)

    with torch.no_grad():
        for name, param in reference_layer.named_parameters():
            if any(proj in name for proj in ["router", "experts", "sinks"]):
                param.data.normal_(0, 1)

    reference_state = reference_layer.state_dict()

    decoder_layer = DecoderLayer(
        setup["mesh_device"],
        config,
        reference_state,
        layer_idx=0,
        ccl_manager=setup["ccl_manager"],
        dtype=setup["dtype"],
        tensor_cache_path=setup["tensor_cache_path"] + "module_tests",
        mesh_config=setup["mesh_config"],
    )

    # Create input
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Create position IDs first
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    # Create attention mask like the working attention test
    mask = torch.triu(torch.full((1, 1, seq_len, seq_len), -float("inf")), diagonal=1)

    # Handle decode mode for TT model like original
    if seq_len == 1:  # decode
        from models.demos.glm_45.utils.general_utils import get_decode_mask

        sliding_window = 0  # No sliding window for this test
        mask = get_decode_mask(position_ids[0].item(), sliding_window)

    # Create position embeddings for reference model
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeDecoderLayer

    rope_embeddings = Glm4MoeRotaryEmbedding(config)
    cos, sin = rope_embeddings(hidden_states, position_ids)
    position_embeddings = (cos, sin)

    # Reference forward pass
    with torch.no_grad():
        reference_output = reference_layer(hidden_states, position_embeddings=position_embeddings)

    # Create TTNN RoPE embeddings for decoder layer
    tt_cos = ttnn.from_torch(
        cos.unsqueeze(-2), device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    tt_sin = ttnn.from_torch(
        sin.unsqueeze(-2), device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    from models.demos.glm_45.tt.rope import ApplyRotaryPosEmb

    apply_rope = ApplyRotaryPosEmb(config)
    rope_mats = (apply_rope, tt_cos, tt_sin)

    # Create position index for TTNN
    tt_position_idx = ttnn.from_torch(
        position_ids, device=setup["mesh_device"], layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
    )

    # Create TTNN mask
    tt_mask = mask.repeat(1, config.num_attention_heads // setup["mesh_device"].shape[1], 1, 1).transpose(1, 2)
    tt_mask = ttnn.from_torch(
        tt_mask.transpose(1, 2) if seq_len > 1 else tt_mask,
        device=setup["mesh_device"],
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    # Create TTNN tensors for component tests
    tt_hidden_states = ttnn.from_torch(
        hidden_states, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    # Test individual components

    if seq_len == 1:
        run_topk_router_component(setup["mesh_device"], hidden_states.shape, reference_layer, decoder_layer)

    # run_experts_component(setup["mesh_device"], hidden_states.shape, config, reference_layer, decoder_layer)
    run_attention_component(
        setup["mesh_device"],
        hidden_states.shape,
        mask,
        tt_mask,
        position_embeddings,
        rope_mats,
        tt_position_idx,
        reference_layer,
        decoder_layer,
    )

    run_rms_norm_component(setup["mesh_device"], hidden_states.shape, reference_layer, decoder_layer)

    run_full_mlp_pipeline(setup["mesh_device"], hidden_states.shape, reference_layer, decoder_layer)

    # Test full decoder layer integration
    tt_output = decoder_layer(
        tt_hidden_states, attention_mask=tt_mask, position_embeddings=rope_mats, position_idx=tt_position_idx
    )

    # Compare outputs
    pcc_threshold = 0.93 if seq_len == 1 else 0.88
    passing, output = run_component_comparison(
        tt_output, reference_output, setup["mesh_device"], pcc_threshold=pcc_threshold
    )
    assert passing, f"Decoder layer test failed. Output: {output}"
