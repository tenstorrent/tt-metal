"""
Minimal utility component tests - RoPE, SDPA, etc.
"""

import math

import pytest
import torch

import ttnn

from ...reference.modeling_gpt_oss import GptOssRotaryEmbedding
from ...tt.rope import ApplyRotaryPosEmb
from ...tt.sdpa import sdpa as tt_sdpa
from ..test_factory import TestFactory, parametrize_batch_seq, parametrize_mesh


@parametrize_mesh(["1x8"])
@parametrize_batch_seq([(1, 1), (1, 32)])
def test_rope_embeddings(mesh_device, batch_size, seq_len, reset_seeds):
    """Test rotary position embeddings - essential for attention"""

    setup = TestFactory.setup_test(mesh_device)
    config = setup["config"]

    # RoPE setup
    rope_embeddings = GptOssRotaryEmbedding(config)
    tt_rope = ApplyRotaryPosEmb(config)

    # Test dimensions
    head_dim = config.head_dim
    num_heads = setup["mesh_config"].shard_size(config.num_attention_heads)

    # Generate position embeddings
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rope_embeddings(torch.zeros(batch_size, seq_len, head_dim), position_ids)

    # Test queries and keys
    queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Convert to TT tensors
    tt_queries = ttnn.from_torch(queries, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"])
    tt_keys = ttnn.from_torch(keys, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"])
    tt_cos = ttnn.from_torch(cos, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"])
    tt_sin = ttnn.from_torch(sin, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"])

    # Apply RoPE
    q_rot, k_rot = tt_rope(tt_queries, tt_keys, tt_cos, tt_sin)

    # Verify shapes preserved
    assert q_rot.shape == tt_queries.shape
    assert k_rot.shape == tt_keys.shape

    # Verify outputs are reasonable
    q_torch = ttnn.to_torch(q_rot, mesh_composer=ttnn.ConcatMesh2dToTensor(setup["mesh_device"]))
    k_torch = ttnn.to_torch(k_rot, mesh_composer=ttnn.ConcatMesh2dToTensor(setup["mesh_device"]))

    assert torch.isfinite(q_torch).all()
    assert torch.isfinite(k_torch).all()


@parametrize_mesh(["1x8"])
def test_scaled_dot_product_attention(mesh_device, reset_seeds):
    """Test SDPA - core attention computation"""

    setup = TestFactory.setup_test(mesh_device)
    config = setup["config"]

    batch_size, seq_len = 1, 32
    num_heads = setup["mesh_config"].shard_size(config.num_attention_heads)
    head_dim = config.head_dim

    # Generate Q, K, V
    queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Attention mask
    attention_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    # Convert to TT
    tt_q = ttnn.from_torch(queries, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"])
    tt_k = ttnn.from_torch(keys, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"])
    tt_v = ttnn.from_torch(values, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"])
    tt_mask = ttnn.from_torch(
        attention_mask, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"]
    )

    # Test SDPA
    output = tt_sdpa(tt_q, tt_k, tt_v, attention_mask=tt_mask, is_causal=True, scale=1.0 / math.sqrt(head_dim))

    # Verify shape
    assert output.shape == tt_q.shape

    # Verify output is reasonable
    output_torch = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMesh2dToTensor(setup["mesh_device"]))
    assert torch.isfinite(output_torch).all()


# Test model configuration flexibility - no hardcoded values
@pytest.mark.parametrize(
    "head_config",
    [
        {"num_attention_heads": 32, "num_key_value_heads": 32, "head_dim": 64},  # Standard
        {"num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 64},  # GQA
        {"num_attention_heads": 40, "num_key_value_heads": 40, "head_dim": 128},  # Different dims
    ],
)
def test_attention_head_flexibility(mesh_device, head_config, reset_seeds):
    """Test components work with different attention head configurations"""

    setup = TestFactory.setup_test(mesh_device)
    config = setup["config"]

    # Override head configuration
    for key, value in head_config.items():
        setattr(config, key, value)

    # Test RoPE with flexible heads
    rope_embeddings = GptOssRotaryEmbedding(config)
    tt_rope = ApplyRotaryPosEmb(config)

    # Generate appropriate sized tensors
    batch_size, seq_len = 1, 16
    num_heads_per_device = setup["mesh_config"].shard_size(config.num_attention_heads)

    queries = torch.randn(batch_size, num_heads_per_device, seq_len, config.head_dim)
    keys = torch.randn(batch_size, num_heads_per_device, seq_len, config.head_dim)

    # Generate RoPE embeddings
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rope_embeddings(torch.zeros(batch_size, seq_len, config.head_dim), position_ids)

    # Convert to TT
    tt_q = ttnn.from_torch(queries, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"])
    tt_k = ttnn.from_torch(keys, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"])
    tt_cos = ttnn.from_torch(cos, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"])
    tt_sin = ttnn.from_torch(sin, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"])

    # Apply RoPE - should work with any head configuration
    q_rot, k_rot = tt_rope(tt_q, tt_k, tt_cos, tt_sin)

    assert q_rot.shape == tt_q.shape
    assert k_rot.shape == tt_k.shape


# Minimal demo/inference test
@parametrize_mesh(["1x8"])
def test_minimal_inference_pipeline(mesh_device, reset_seeds):
    """Test minimal inference pipeline - essential for deployment"""

    setup = TestFactory.setup_test(mesh_device)
    config = setup["config"]

    # Test token processing pipeline
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size

    # Input tokens (batch_size=1 for inference)
    input_tokens = torch.randint(0, vocab_size, (1, 8))

    # Embedding lookup (simplified)
    embedding_weight = torch.randn(vocab_size, hidden_size)
    embedded = torch.nn.functional.embedding(input_tokens, embedding_weight)

    # Convert to TT
    tt_embedded = ttnn.from_torch(embedded, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"])

    # Test that we can process through core components
    from ...tt.experts import Experts
    from ...tt.rms_norm import RMSNorm

    # RMS norm
    norm_state = {"weight": torch.ones(hidden_size)}
    norm = RMSNorm(setup["mesh_device"], config, norm_state, mesh_config=setup["mesh_config"])

    # Experts
    experts = Experts(
        setup["mesh_device"], config, setup["state_dict"], setup["ccl_manager"], mesh_config=setup["mesh_config"]
    )

    # Process through pipeline
    normalized = norm(tt_embedded)

    # Generate dummy routing weights for experts
    routing_weights = torch.softmax(torch.randn(1, 8, config.num_local_experts), dim=-1)
    tt_routing = ttnn.from_torch(
        routing_weights, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"]
    )

    expert_output = experts(normalized, tt_routing)

    # Verify pipeline works end-to-end
    assert expert_output.shape == tt_embedded.shape

    # Convert back to tokens (simplified logits)
    output_torch = ttnn.to_torch(expert_output, mesh_composer=ttnn.ConcatMesh2dToTensor(setup["mesh_device"]))
    logits = torch.matmul(output_torch, embedding_weight.T)  # Simplified lm_head

    # Verify we get reasonable logits
    assert logits.shape == (1, 8, vocab_size)
    assert torch.isfinite(logits).all()


# Memory and performance smoke test
def test_memory_efficiency(mesh_device, reset_seeds):
    """Smoke test for memory efficiency - no OOM with reasonable sizes"""

    setup = TestFactory.setup_test(mesh_device)
    config = setup["config"]

    # Test with larger sequence length to stress memory
    batch_size, seq_len = 1, 128

    # Create larger hidden states
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    tt_hidden = ttnn.from_torch(
        hidden_states, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"]
    )

    # Test that we can process without OOM
    from ...tt.experts import Experts

    experts = Experts(
        setup["mesh_device"], config, setup["state_dict"], setup["ccl_manager"], mesh_config=setup["mesh_config"]
    )

    routing_weights = torch.softmax(torch.randn(batch_size, seq_len, config.num_local_experts), dim=-1)
    tt_routing = ttnn.from_torch(
        routing_weights, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"]
    )

    # Should complete without memory issues
    output = experts(tt_hidden, tt_routing)
    assert output.shape == tt_hidden.shape

    # Clean up explicitly
    tt_hidden.deallocate(force=True)
    tt_routing.deallocate(force=True)
    output.deallocate(force=True)
