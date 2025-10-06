"""
Minimal utility component tests - RoPE and SDPA only
"""

import math

import torch

import ttnn

from ...reference.modeling_gpt_oss import GptOssRotaryEmbedding
from ...tt.rope import ApplyRotaryPosEmb
from ...tt.sdpa import sdpa as tt_sdpa
from ..test_factory import TestFactory, parametrize_batch_seq, parametrize_mesh_with_fabric


@parametrize_mesh_with_fabric()
@parametrize_batch_seq([(1, 1), (1, 32)])
def test_rope_embeddings(mesh_device, batch_size, seq_len, reset_seeds):
    """Test rotary position embeddings - essential for attention"""

    setup = TestFactory.setup_test(mesh_device)
    config = setup["config"]

    # Create position IDs and RoPE embeddings like original
    position_ids = torch.arange(seq_len).unsqueeze(0)
    rope_embeddings = GptOssRotaryEmbedding(config)
    torch_inputs = torch.randn(batch_size, seq_len, config.hidden_size)
    cos, sin = rope_embeddings(torch_inputs, position_ids)

    # Create Q/K tensors with proper shapes like original
    q_torch = torch.randn(batch_size, config.num_attention_heads, seq_len, config.head_dim)
    k_torch = torch.randn(batch_size, config.num_key_value_heads, seq_len, config.head_dim)

    # Get reference outputs using original apply_rotary_pos_emb
    from models.demos.gpt_oss.reference.modeling_gpt_oss import apply_rotary_pos_emb

    q_rope_torch, k_rope_torch = apply_rotary_pos_emb(q_torch, k_torch, cos, sin)

    # Convert to TTNN tensors with proper sharding like original
    q_tt = ttnn.from_torch(
        q_torch.permute(0, 2, 1, 3),  # [batch, seq_len, num_heads, head_dim]
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(None, -2)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    k_tt = ttnn.from_torch(
        k_torch.permute(0, 2, 1, 3),  # [batch, seq_len, num_heads, head_dim]
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(None, -2)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Convert cos/sin with proper sharding like original
    cos_tt = ttnn.from_torch(
        cos.unsqueeze(-2),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(None, None)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    sin_tt = ttnn.from_torch(
        sin.unsqueeze(-2),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(None, None)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Apply RoPE like original
    apply_rope = ApplyRotaryPosEmb(config)
    q_tt_rotated = apply_rope(q_tt, cos_tt, sin_tt)
    k_tt_rotated = apply_rope(k_tt, cos_tt, sin_tt)

    # Convert back to torch and compare like original
    q_tt_rotated_torch = ttnn.to_torch(
        q_tt_rotated, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_device.shape, dims=(0, -2))
    ).permute(0, 2, 1, 3)[
        :1
    ]  # Back to [batch, num_heads, seq_len, head_dim]

    k_tt_rotated_torch = ttnn.to_torch(
        k_tt_rotated, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_device.shape, dims=(0, -2))
    ).permute(0, 2, 1, 3)[
        :1
    ]  # Back to [batch, num_heads, seq_len, head_dim]

    # Debug: Print shapes to understand the mismatch
    print(f"q_tt_rotated_torch shape: {q_tt_rotated_torch.shape}")
    print(f"q_rope_torch shape: {q_rope_torch.shape}")
    print(f"k_tt_rotated_torch shape: {k_tt_rotated_torch.shape}")
    print(f"k_rope_torch shape: {k_rope_torch.shape}")

    # Compare with reference using PCC like original
    from models.utility_functions import comp_pcc

    passing, pcc_message = comp_pcc(q_tt_rotated_torch, q_rope_torch)
    mse = torch.nn.functional.mse_loss(q_tt_rotated_torch, q_rope_torch)
    print(f"Q: {pcc_message}, mse: {mse}")
    assert passing, f"q_tt_rotated_torch: {pcc_message}"

    passing, pcc_message = comp_pcc(k_tt_rotated_torch, k_rope_torch)
    mse = torch.nn.functional.mse_loss(k_tt_rotated_torch, k_rope_torch)
    print(f"K: {pcc_message}, mse: {mse}")
    assert passing, f"k_tt_rotated_torch: {pcc_message}"


@parametrize_mesh_with_fabric()
def test_scaled_dot_product_attention(mesh_device, device_params, reset_seeds):
    """Test SDPA - core attention computation"""

    setup = TestFactory.setup_test(mesh_device)
    config = setup["config"]

    batch_size, seq_len = 1, 32
    num_heads = setup["mesh_config"].shard_size(config.num_attention_heads)
    num_kv_heads = setup["mesh_config"].shard_size(config.num_key_value_heads)
    head_dim = config.head_dim

    # Generate Q, K, V with correct shapes for SDPA function
    queries = torch.randn(seq_len, batch_size, num_heads, head_dim)
    keys = torch.randn(seq_len, num_kv_heads, head_dim)
    values = torch.randn(seq_len, num_kv_heads, head_dim)
    sinks = torch.randn(num_heads)

    # Convert to TTNN tensors
    tt_q = ttnn.from_torch(queries, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_k = ttnn.from_torch(keys, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_v = ttnn.from_torch(values, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_sink = ttnn.from_torch(sinks, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Test SDPA
    output, cache = tt_sdpa(tt_q, tt_k, tt_v, tt_sink, sm_scale=1.0 / math.sqrt(head_dim))

    # Verify shape and output
    expected_shape = (1, 1, seq_len, head_dim * num_heads)
    assert output.shape == expected_shape

    mesh_shape = mesh_device.shape
    output_torch = ttnn.to_torch(
        output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(-2, -1))
    )
    assert torch.isfinite(output_torch).all()
