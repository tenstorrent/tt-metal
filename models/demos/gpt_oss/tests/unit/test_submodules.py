# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal utility component tests - RoPE and SDPA only
"""


import torch
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding, apply_rotary_pos_emb

import ttnn
from models.common.utility_functions import comp_pcc

from ...tt.rope import ApplyRotaryPosEmb
from ...tt.sdpa import sdpa as tt_sdpa
from ..test_factory import TestFactory, parametrize_batch_seq, parametrize_mesh_with_fabric


def reference_sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window)
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


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
    # # Use HuggingFace model for RoPE reference
    # from transformers import AutoModelForCausalLM

    # # Simple RoPE implementation for testing
    # def apply_rotary_pos_emb(q, k, cos, sin):
    #     q_embed = (q * cos) + (rotate_half(q) * sin)
    #     k_embed = (k * cos) + (rotate_half(k) * sin)
    #     return q_embed, k_embed

    # def rotate_half(x):
    #     x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    #     return torch.cat((-x2, x1), dim=-1)

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
    passing, pcc_message = comp_pcc(q_tt_rotated_torch, q_rope_torch)
    mse = torch.nn.functional.mse_loss(q_tt_rotated_torch, q_rope_torch)
    assert passing, f"q_tt_rotated_torch: {pcc_message}"

    passing, pcc_message = comp_pcc(k_tt_rotated_torch, k_rope_torch)
    mse = torch.nn.functional.mse_loss(k_tt_rotated_torch, k_rope_torch)
    assert passing, f"k_tt_rotated_torch: {pcc_message}"


@parametrize_mesh_with_fabric()
def test_scaled_dot_product_attention(mesh_device, device_params, reset_seeds):
    """Test SDPA - core attention computation"""

    setup = TestFactory.setup_test(mesh_device)
    config = setup["config"]
    nh = setup["mesh_config"].shard_size(config.num_attention_heads)
    nkv = setup["mesh_config"].shard_size(config.num_key_value_heads)
    dim = config.head_dim
    sliding_window = 128
    dtype = ttnn.bfloat16
    device = mesh_device

    q, k, v = None, None, None
    tt_cache = None
    all_passing = True
    num_iters = 5
    num_tokens = 128
    for n in range(num_iters):
        cur_seq_len = num_tokens + n

        # Torch input
        q = torch.randn(num_tokens, nkv, nh // nkv, dim) if n == 0 else q
        k = torch.randn(num_tokens, nkv, dim) if k is None else k
        v = torch.randn(num_tokens, nkv, dim) if v is None else v
        s = torch.randn(1, nh, 1, 1)
        sm_scale = 1.0 / (dim**0.5)

        if n > 0:
            q = torch.cat([q, torch.randn(1, nkv, nh // nkv, dim)], dim=0)
            k = torch.cat([k, torch.randn(1, nkv, dim)], dim=0)
            v = torch.cat([v, torch.randn(1, nkv, dim)], dim=0)

        mask = torch.triu(torch.full((1, 1, cur_seq_len, cur_seq_len), -float("inf")), diagonal=1)
        if sliding_window > 0:
            mask += torch.tril(torch.full((1, 1, cur_seq_len, cur_seq_len), -float("inf")), diagonal=-sliding_window)

        if n > 0:
            mask = mask[:, :, -1:, :]

        # Torch output
        reference_out = reference_sdpa(q, k, v, s, sm_scale, sliding_window)

        # TT input
        if n == 0:
            q_in = q.view(num_tokens, 1, nh, dim)
            k_in = k
            v_in = v
        else:
            q_in = q[-1:, ...].view(1, 1, nh, dim)
            k_in = k[-1:, ...]
            v_in = v[-1:, ...]

        tt_q = ttnn.from_torch(
            q_in, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_k = ttnn.from_torch(
            k_in, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_v = ttnn.from_torch(
            v_in, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_sink = ttnn.from_torch(
            s, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_mask = ttnn.from_torch(
            mask, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # TT output
        tt_out, tt_cache = tt_sdpa(tt_q, tt_k, tt_v, tt_sink, sm_scale=sm_scale, tt_mask=tt_mask, tt_cache=tt_cache)
        tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])

        # Only compare the last token
        if n > 0:
            reference_out = reference_out[-1:, ...]

        # Compare outputs
        pcc = 0.99
        passed, pcc_message = comp_pcc(reference_out, tt_out_torch, pcc)
        if not passed:
            print(f"Iteration {n} | Test passed: {passed}, PCC: {pcc_message}")
        all_passing = all_passing and passed

    assert all_passing, "Test failed: Outputs do not match"
