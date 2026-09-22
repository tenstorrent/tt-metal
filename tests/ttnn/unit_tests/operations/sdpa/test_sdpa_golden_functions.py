# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


SDPA_GOLDEN_OPERATIONS = (
    ttnn.transformer.chunked_flash_mla_prefill,
    ttnn.transformer.chunked_scaled_dot_product_attention,
    ttnn.transformer.exp_ring_joint_scaled_dot_product_attention,
    ttnn.transformer.flash_mla_prefill,
    ttnn.transformer.flash_multi_latent_attention_decode,
    ttnn.transformer.joint_scaled_dot_product_attention,
    ttnn.transformer.paged_flash_multi_latent_attention_decode,
    ttnn.transformer.paged_scaled_dot_product_attention_decode,
    ttnn.transformer.ring_distributed_scaled_dot_product_attention,
    ttnn.transformer.ring_joint_scaled_dot_product_attention,
    ttnn.transformer.ring_mla,
    ttnn.transformer.scaled_dot_product_attention,
    ttnn.transformer.scaled_dot_product_attention_decode,
    ttnn.transformer.sparse_sdpa,
    ttnn.transformer.sparse_sdpa_msa,
)


def test_sdpa_family_has_registered_golden_functions():
    for operation in SDPA_GOLDEN_OPERATIONS:
        assert callable(ttnn.get_golden_function(operation))


def test_scaled_dot_product_attention_golden_matches_torch_gqa_causal():
    torch.manual_seed(0)
    query = torch.randn(1, 4, 5, 8)
    key = torch.randn(1, 2, 5, 8)
    value = torch.randn(1, 2, 5, 6)
    key_expanded = key.repeat_interleave(2, dim=1)
    value_expanded = value.repeat_interleave(2, dim=1)

    expected = torch.nn.functional.scaled_dot_product_attention(query, key_expanded, value_expanded, is_causal=True)
    golden = ttnn.get_golden_function(ttnn.transformer.scaled_dot_product_attention)
    actual = golden(query, key, value, is_causal=True)

    torch.testing.assert_close(actual, expected)


def test_sdpa_decode_golden_supports_mixed_query_and_kv_dtypes():
    torch.manual_seed(6)
    query = torch.randn(1, 1, 2, 4, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 3, 4, dtype=torch.float32)
    value = torch.randn(1, 1, 3, 4, dtype=torch.float32)

    golden = ttnn.get_golden_function(ttnn.transformer.scaled_dot_product_attention_decode)
    actual = golden(query, key, value, cur_pos=[2])

    expected = torch.nn.functional.scaled_dot_product_attention(
        query.permute(1, 2, 0, 3),
        key.to(query.dtype).repeat_interleave(2, dim=1),
        value.to(query.dtype).repeat_interleave(2, dim=1),
    ).permute(2, 0, 1, 3)
    torch.testing.assert_close(actual, expected)


def test_paged_decode_golden_reconstructs_logical_cache():
    torch.manual_seed(1)
    batch, num_query_heads, num_kv_heads, head_dim = 2, 4, 2, 8
    query = torch.randn(1, batch, num_query_heads, head_dim)
    logical_key = torch.randn(batch, num_kv_heads, 4, head_dim)
    logical_value = torch.randn(batch, num_kv_heads, 4, head_dim)
    page_table = torch.tensor([[1], [0]], dtype=torch.int64)

    # P == B and block_size > pages_per_user used to trigger the ambiguous
    # "already contiguous" shape heuristic.
    block_size = 4
    key_pages = torch.empty(batch, num_kv_heads, block_size, head_dim)
    value_pages = torch.empty_like(key_pages)
    for user in range(batch):
        for logical_page, physical_page in enumerate(page_table[user]):
            start = logical_page * block_size
            key_pages[physical_page] = logical_key[user, :, start : start + block_size]
            value_pages[physical_page] = logical_value[user, :, start : start + block_size]

    positions = torch.tensor([2, 3])
    golden = ttnn.get_golden_function(ttnn.transformer.paged_scaled_dot_product_attention_decode)
    actual = golden(
        query,
        key_pages,
        value_pages,
        page_table,
        cur_pos_tensor=positions,
        is_causal=True,
    )

    expected_rows = []
    query_by_batch = query.permute(1, 2, 0, 3)
    for user, position in enumerate(positions.tolist()):
        expected_rows.append(
            torch.nn.functional.scaled_dot_product_attention(
                query_by_batch[user : user + 1],
                logical_key[user : user + 1, :, : position + 1].repeat_interleave(2, dim=1),
                logical_value[user : user + 1, :, : position + 1].repeat_interleave(2, dim=1),
            )
        )
    expected = torch.cat(expected_rows).permute(2, 0, 1, 3)
    torch.testing.assert_close(actual, expected)


def test_paged_decode_golden_applies_geometry_override():
    torch.manual_seed(3)
    query = torch.randn(1, 1, 4, 2)
    page_table = torch.tensor([[1, 0]], dtype=torch.int64)
    logical_key_pages = torch.randn(2, 2, 2, 2)
    logical_value_pages = torch.randn(2, 2, 2, 2)
    physical_key_pages = logical_key_pages.reshape(2, 1, 2, 4)
    physical_value_pages = logical_value_pages.reshape(2, 1, 2, 4)
    geometry = ttnn.PagedCacheGeometryOverride(block_size=2, num_kv_heads=2)

    golden = ttnn.get_golden_function(ttnn.transformer.paged_scaled_dot_product_attention_decode)
    actual = golden(
        query,
        physical_key_pages,
        physical_value_pages,
        page_table,
        cur_pos_tensor=torch.tensor([3]),
        paged_cache_geometry=geometry,
    )

    key = logical_key_pages[page_table[0]].permute(1, 0, 2, 3).reshape(1, 2, 4, 2)
    value = logical_value_pages[page_table[0]].permute(1, 0, 2, 3).reshape(1, 2, 4, 2)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.permute(1, 2, 0, 3),
        key.repeat_interleave(2, dim=1),
        value.repeat_interleave(2, dim=1),
    ).permute(2, 0, 1, 3)
    torch.testing.assert_close(actual, expected)


def test_paged_decode_golden_gathers_wrapped_sliding_window_in_logical_order():
    query = torch.tensor([[[[1.0]]]])
    page_table = torch.tensor([[0, 1]], dtype=torch.int64)
    # Absolute positions 4 and 5 have overwritten circular slots 0 and 1.
    key_pages = torch.tensor([[[[4.0], [5.0]]], [[[2.0], [3.0]]]])
    value_pages = torch.tensor([[[[40.0], [50.0]]], [[[20.0], [30.0]]]])

    golden = ttnn.get_golden_function(ttnn.transformer.paged_scaled_dot_product_attention_decode)
    actual = golden(
        query,
        key_pages,
        value_pages,
        page_table,
        cur_pos_tensor=torch.tensor([5]),
        sliding_window_size=2,
        cache_position_modulo=4,
        scale=1.0,
    )

    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        torch.tensor([[[[4.0], [5.0]]]]),
        torch.tensor([[[[40.0], [50.0]]]]),
        scale=1.0,
    )
    torch.testing.assert_close(actual, expected)


def test_sparse_sdpa_golden_honors_masked_index_and_attention_sink():
    query = torch.tensor([[[[1.0, 0.0]]]])
    kv = torch.tensor([[[[1.0, 2.0], [0.0, 4.0], [-1.0, 8.0]]]])
    indices = torch.tensor([[[[0, 2, 0xFFFFFFFF]]]], dtype=torch.int64)
    sink = torch.tensor([[[[0.25]]]])

    golden = ttnn.get_golden_function(ttnn.transformer.sparse_sdpa)
    actual = golden(
        query,
        kv,
        indices,
        1,
        kv_format=None,
        scale=1.0,
        attention_sink=sink,
    )

    logits = torch.tensor([1.0, -1.0, 0.25])
    weights = torch.softmax(logits, dim=0)[:2]
    expected = (weights * torch.tensor([1.0, -1.0])).sum().reshape(1, 1, 1, 1)
    torch.testing.assert_close(actual, expected)


def test_ring_distributed_golden_returns_rank_query_chunks():
    query = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8, 1)
    key = torch.ones_like(query)
    value = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8, 1)
    full = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
    ring = ttnn.get_golden_function(ttnn.transformer.ring_distributed_scaled_dot_product_attention)

    rank_one = ring(query, key, value, 2, 1)
    expected = torch.cat([full[..., 2:4, :], full[..., 4:6, :]], dim=-2)
    torch.testing.assert_close(rank_one, expected)


def test_joint_sdpa_golden_splits_combined_output():
    torch.manual_seed(2)
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)
    joint_query = torch.randn(1, 2, 2, 4)
    joint_key = torch.randn(1, 2, 2, 4)
    joint_value = torch.randn(1, 2, 2, 4)
    golden = ttnn.get_golden_function(ttnn.transformer.joint_scaled_dot_product_attention)

    output, joint_output = golden(
        query,
        key,
        value,
        joint_query,
        joint_key,
        joint_value,
        joint_strategy="rear",
        program_config=None,
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        torch.cat([query, joint_query], dim=-2),
        torch.cat([key, joint_key], dim=-2),
        torch.cat([value, joint_value], dim=-2),
    )
    torch.testing.assert_close(output, expected[..., :3, :])
    torch.testing.assert_close(joint_output, expected[..., 3:, :])


def test_ring_joint_golden_uses_the_same_sink_logits_for_output_and_lse():
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [1.0, -1.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, -1.0]]]])
    value = torch.tensor([[[[2.0, 1.0], [4.0, 3.0]], [[1.0, 5.0], [2.0, 6.0]]]])
    sink = torch.tensor([[[[0.25]], [[-0.5]]]])
    scale = 0.75

    golden = ttnn.get_golden_function(ttnn.transformer.ring_joint_scaled_dot_product_attention)
    output, joint_output, lse = golden(
        query,
        key,
        value,
        logical_n=2,
        is_causal=True,
        attention_sink=sink,
        scale=scale,
    )

    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    causal_mask = torch.tril(torch.ones(2, 2, dtype=torch.bool)).reshape(1, 1, 2, 2)
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    sink_logits = sink.expand(1, 2, 2, 1) * scale
    logits = torch.cat([scores, sink_logits], dim=-1)
    expected_probabilities = torch.softmax(logits, dim=-1)[..., :2]
    expected_output = torch.matmul(expected_probabilities, value)
    expected_lse = torch.logsumexp(logits, dim=-1, keepdim=True)

    torch.testing.assert_close(output, expected_output)
    assert joint_output.shape[-2] == 0
    torch.testing.assert_close(lse, expected_lse)


def test_ring_joint_golden_resolves_indexed_layer_metadata():
    query = torch.tensor([[[[1.0, 0.0]]]])
    key = torch.randn(4, 1, 2, 2)
    value = torch.randn(4, 1, 2, 3)
    slot_id = torch.tensor([1])
    actual_kv_length = torch.tensor([0])

    golden = ttnn.get_golden_function(ttnn.transformer.ring_joint_scaled_dot_product_attention)
    output, _, _ = golden(
        query,
        key,
        value,
        logical_n=2,
        slot_id=slot_id,
        kv_actual_isl_tensor=actual_kv_length,
        kv_cache_num_layers=2,
        kv_cache_layer_idx=0,
    )

    expected = torch.nn.functional.scaled_dot_product_attention(query, key[2:3], value[2:3])
    torch.testing.assert_close(output, expected)


def test_ring_joint_golden_uses_actual_kv_length_as_causal_query_offset():
    torch.manual_seed(4)
    query = torch.randn(1, 1, 2, 3)
    key = torch.randn(1, 1, 4, 3)
    value = torch.randn(1, 1, 4, 2)

    golden = ttnn.get_golden_function(ttnn.transformer.ring_joint_scaled_dot_product_attention)
    output, _, _ = golden(
        query,
        key,
        value,
        logical_n=4,
        kv_actual_isl=2,
        is_causal=True,
    )

    query_positions = torch.arange(2, 4).unsqueeze(1)
    key_positions = torch.arange(4).unsqueeze(0)
    causal_mask = (key_positions <= query_positions).reshape(1, 1, 2, 4)
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=causal_mask)
    torch.testing.assert_close(output, expected)


def test_ring_joint_golden_cross_attention_uses_only_logical_keys():
    torch.manual_seed(5)
    query = torch.randn(1, 2, 2, 4)
    key = torch.randn(1, 1, 5, 4)
    value = torch.randn(1, 1, 5, 3)

    golden = ttnn.get_golden_function(ttnn.transformer.ring_joint_scaled_dot_product_attention)
    output, _, _ = golden(
        query,
        key,
        value,
        logical_n=3,
        is_causal=False,
        is_cross=True,
    )

    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key[..., :3, :].repeat_interleave(2, dim=1),
        value[..., :3, :].repeat_interleave(2, dim=1),
    )
    torch.testing.assert_close(output, expected)
