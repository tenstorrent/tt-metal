# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


def test_sdpa_and_joint_golden_contracts_host_only():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 6)
    repeated_k = k.repeat_interleave(2, dim=1)
    repeated_v = v.repeat_interleave(2, dim=1)

    golden = ttnn.get_golden_function(ttnn.transformer.scaled_dot_product_attention)
    actual = golden(q, k, v, is_causal=True)
    expected = torch.nn.functional.scaled_dot_product_attention(q, repeated_k, repeated_v, is_causal=True)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

    joint_q = torch.randn(1, 4, 2, 8)
    joint_k = torch.randn(1, 2, 2, 8)
    joint_v = torch.randn(1, 2, 2, 6)
    joint_golden = ttnn.get_golden_function(ttnn.transformer.joint_scaled_dot_product_attention)
    actual_main, actual_joint = joint_golden(
        q,
        k,
        v,
        joint_q,
        joint_k,
        joint_v,
        joint_strategy="rear",
    )
    combined_q = torch.cat((q, joint_q), dim=-2)
    combined_k = torch.cat((k, joint_k), dim=-2).repeat_interleave(2, dim=1)
    combined_v = torch.cat((v, joint_v), dim=-2).repeat_interleave(2, dim=1)
    expected = torch.nn.functional.scaled_dot_product_attention(combined_q, combined_k, combined_v)
    assert torch.allclose(actual_main, expected[..., : q.shape[-2], :], atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_joint, expected[..., q.shape[-2] :, :], atol=1e-6, rtol=1e-6)


def test_decode_paged_and_chunked_sdpa_golden_contracts_host_only():
    torch.manual_seed(1)
    q_decode = torch.randn(1, 1, 2, 4)
    k = torch.randn(1, 1, 4, 4)
    v = torch.randn(1, 1, 4, 3)

    decode = ttnn.get_golden_function(ttnn.transformer.scaled_dot_product_attention_decode)
    actual = decode(q_decode, k, v, cur_pos=[2])
    expected = torch.nn.functional.scaled_dot_product_attention(
        q_decode[:, 0].unsqueeze(-2),
        k[..., :3, :],
        v[..., :3, :],
    )[0, :, 0]
    assert torch.allclose(actual[0, 0], expected, atol=1e-6, rtol=1e-6)

    decode_mask = torch.zeros(1, 1, q_decode.shape[2], k.shape[-2])
    decode_mask[..., 1] = float("-inf")
    actual = decode(q_decode, k, v, is_causal=False, attn_mask=decode_mask)
    expected_noncausal = torch.nn.functional.scaled_dot_product_attention(
        q_decode[:, 0].unsqueeze(-2),
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
        attn_mask=decode_mask.transpose(1, 2),
    )
    assert torch.allclose(actual[0], expected_noncausal[:, :, 0], atol=1e-6, rtol=1e-6)

    page_table = torch.tensor([[1, 0]], dtype=torch.int32)
    paged_k = torch.stack((k[0, :, 2:4], k[0, :, 0:2]))
    paged_v = torch.stack((v[0, :, 2:4], v[0, :, 0:2]))
    paged_decode = ttnn.get_golden_function(ttnn.transformer.paged_scaled_dot_product_attention_decode)
    actual = paged_decode(q_decode, paged_k, paged_v, page_table, cur_pos_tensor=torch.tensor([2]))
    assert torch.allclose(actual[0, 0], expected, atol=1e-6, rtol=1e-6)

    q_geometry = torch.randn(1, 1, 1, 32)
    allocated_k = torch.randn(1, 1, 32, 64)
    allocated_v = torch.randn(1, 1, 32, 64)
    geometry = ttnn.PagedCacheGeometryOverride(block_size=64, num_kv_heads=1)

    def reinterpret_geometry(cache):
        cache = cache.reshape(1, 1, 1, 32, 2, 32).permute(0, 1, 2, 4, 3, 5).contiguous()
        cache = cache.reshape(1, 1, 2, 1, 32, 32).permute(0, 1, 2, 4, 3, 5).contiguous()
        return cache.reshape(1, 1, 64, 32)

    actual = paged_decode(
        q_geometry,
        allocated_k,
        allocated_v,
        torch.zeros(1, 1, dtype=torch.int32),
        cur_pos_tensor=torch.tensor([63]),
        paged_cache_geometry=geometry,
    )
    expected_geometry = torch.nn.functional.scaled_dot_product_attention(
        q_geometry[:, 0].unsqueeze(-2),
        reinterpret_geometry(allocated_k),
        reinterpret_geometry(allocated_v),
    )
    assert torch.allclose(actual[0, 0], expected_geometry[0, :, 0], atol=1e-6, rtol=1e-6)

    full_q = torch.randn(1, 2, 4, 4)
    chunked = ttnn.get_golden_function(ttnn.transformer.chunked_scaled_dot_product_attention)
    actual = chunked(full_q[..., 2:4, :], paged_k, paged_v, page_table, 2)
    expected = torch.nn.functional.scaled_dot_product_attention(
        full_q,
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
        is_causal=True,
    )[..., 2:4, :]
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_sparse_sdpa_goldens_host_only(expect_error):
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    kv = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]]])
    indices = torch.tensor([[[[0, 2], [1, 2]]]], dtype=torch.uint32)
    sparse = ttnn.get_golden_function(ttnn.transformer.sparse_sdpa)
    actual = sparse(q, kv, indices, 1, kv_format=ttnn.transformer.SparseKVFormat.BF16, scale=1.0)

    expected = torch.empty_like(actual)
    for row, selected in enumerate(([0, 2], [1, 2])):
        keys = kv[0, 0, selected]
        probabilities = torch.softmax(q[0, 0, row] @ keys.T, dim=-1)
        expected[0, 0, row] = probabilities @ keys[:, :1]
    assert torch.allclose(actual, expected)

    msa = ttnn.get_golden_function(ttnn.transformer.sparse_sdpa_msa)
    msa_indices = torch.tensor([[[[0], [1]]]], dtype=torch.uint32)
    actual = msa(q, kv, kv[..., :1], msa_indices, block_size=2, scale=1.0)
    expected = torch.empty_like(actual)
    for row, selected in enumerate(([0, 1], [2, 3])):
        keys = kv[0, 0, selected]
        probabilities = torch.softmax(q[0, 0, row] @ keys.T, dim=-1)
        expected[0, 0, row] = probabilities @ keys[:, :1]
    assert torch.allclose(actual, expected)

    with expect_error(ValueError, "block-cyclic"):
        sparse(
            q,
            kv,
            indices,
            1,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            block_cyclic_sp_axis=0,
            block_cyclic_chunk_local=2,
        )
