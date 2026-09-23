# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from ttnn.operations.transformer_golden import sparse_mla


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


@pytest.mark.parametrize(
    "query_dtype, kv_dtype",
    [(torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)],
    ids=["bf16-q-fp8-kv", "fp8-q-bf16-kv"],
)
def test_sparse_sdpa_golden_supports_mixed_query_and_kv_dtypes(query_dtype, kv_dtype):
    torch.manual_seed(15)
    # FP8 operands reach the golden as FP8-quantized float32 tensors.
    query = torch.randn(1, 2, 3, 8).to(torch.float8_e4m3fn).float().to(query_dtype)
    kv = torch.randn(1, 1, 5, 8).to(torch.float8_e4m3fn).float().to(kv_dtype)
    indices = torch.tensor([[[[0, 2, 0xFFFFFFFF], [1, 3, 4], [4, 0xFFFFFFFF, 0xFFFFFFFF]]]], dtype=torch.int64)

    golden = ttnn.get_golden_function(ttnn.transformer.sparse_sdpa)
    actual = golden(query, kv, indices, 4, kv_format=None, scale=0.5)

    expected = sparse_mla(query.float(), kv[0, 0].float(), indices, 0.5, 4).to(query_dtype)
    assert actual.dtype == query_dtype
    torch.testing.assert_close(actual, expected)


def _pack_scaled_fp8_kv(latent, scales, rope):
    raw = torch.cat((latent.view(torch.uint8), scales.view(torch.uint8), rope.view(torch.uint8)), dim=-1)
    return raw.view(torch.float8_e4m3fn)


def test_sparse_sdpa_golden_decodes_packed_scaled_fp8_kv():
    torch.manual_seed(16)
    latent_dim, rope_dim, tokens = 512, 64, 3
    latent = torch.randn(1, 1, tokens, latent_dim).to(torch.float8_e4m3fn)
    scales = torch.tensor([0.25, 0.5, 1.0, 2.0]).reshape(1, 1, 1, 4).expand(1, 1, tokens, 4).contiguous()
    rope = torch.randn(1, 1, tokens, rope_dim).to(torch.bfloat16)
    packed = _pack_scaled_fp8_kv(latent, scales, rope)
    assert packed.shape[-1] == 656

    logical = torch.cat(
        ((latent.float() * scales.repeat_interleave(128, dim=-1)).to(torch.bfloat16), rope),
        dim=-1,
    )
    query = torch.randn(1, 2, 2, latent_dim + rope_dim).to(torch.bfloat16)
    indices = torch.tensor([[[[0, 2], [1, 0xFFFFFFFF]]]], dtype=torch.int64)

    golden = ttnn.get_golden_function(ttnn.transformer.sparse_sdpa)
    actual = golden(query, packed, indices, latent_dim, kv_format=ttnn.transformer.SparseKVFormat.SCALED_FP8)
    expected = golden(query, logical, indices, latent_dim, kv_format=ttnn.transformer.SparseKVFormat.BF16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sparse_sdpa_golden_rejects_value_converted_scaled_fp8_kv(expect_error):
    query = torch.randn(1, 1, 1, 576)
    packed = torch.zeros(1, 1, 2, 656)
    indices = torch.tensor([[[[0, 1]]]], dtype=torch.int64)

    golden = ttnn.get_golden_function(ttnn.transformer.sparse_sdpa)
    with expect_error(ValueError, "packed bytes"):
        golden(query, packed, indices, 512, kv_format=ttnn.transformer.SparseKVFormat.SCALED_FP8)


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


def test_sdpa_decode_golden_uses_padded_per_head_sink():
    torch.manual_seed(7)
    batch, heads, dim, seq = 2, 2, 4, 3
    query = torch.randn(1, batch, heads, dim)
    key = torch.randn(batch, 1, seq, dim)
    value = torch.randn(batch, 1, seq, dim)
    sink = torch.zeros(heads, 32)
    sink[:, 0] = torch.tensor([0.5, -1.25])

    golden = ttnn.get_golden_function(ttnn.transformer.scaled_dot_product_attention_decode)
    actual = golden(query, key, value, cur_pos=[seq - 1, seq - 1], is_causal=True, attention_sink=sink, scale=1.0)

    attended_query = query.permute(1, 2, 0, 3).float()
    attended_key = key.repeat_interleave(heads, dim=1).float()
    attended_value = value.repeat_interleave(heads, dim=1).float()
    scores = torch.matmul(attended_query, attended_key.transpose(-2, -1))
    sink_logits = sink[:, :1].reshape(1, heads, 1, 1).expand(batch, heads, 1, 1)
    probabilities = torch.softmax(torch.cat([scores, sink_logits], dim=-1), dim=-1)[..., :-1]
    expected = torch.matmul(probabilities, attended_value).permute(2, 0, 1, 3)
    torch.testing.assert_close(actual, expected)


def test_sdpa_decode_golden_converts_device_mask_layout():
    torch.manual_seed(8)
    batch, heads, dim, keys = 2, 4, 4, 5
    query = torch.randn(1, batch, heads, dim)
    key = torch.randn(batch, 1, keys, dim)
    value = torch.randn(batch, 1, keys, dim)
    logical_mask = torch.zeros(batch, heads, 1, keys)
    logical_mask[:, 0, :, 0] = -1.0e4
    logical_mask[:, 1, :, 1] = -1.0e4
    device_mask = logical_mask.transpose(1, 2).contiguous()

    golden = ttnn.get_golden_function(ttnn.transformer.scaled_dot_product_attention_decode)
    actual = golden(
        query,
        key,
        value,
        is_causal=False,
        attn_mask=device_mask,
        cur_pos=[keys - 1] * batch,
        scale=1.0,
    )

    expected = torch.nn.functional.scaled_dot_product_attention(
        query.permute(1, 2, 0, 3),
        key.repeat_interleave(heads, dim=1),
        value.repeat_interleave(heads, dim=1),
        attn_mask=logical_mask,
        scale=1.0,
    ).permute(2, 0, 1, 3)
    torch.testing.assert_close(actual, expected)


def test_sdpa_decode_golden_broadcasts_a_single_mask_batch():
    torch.manual_seed(13)
    batch, heads, dim, keys = 3, 2, 4, 4
    query = torch.randn(1, batch, heads, dim)
    key = torch.randn(batch, 1, keys, dim)
    value = torch.randn(batch, 1, keys, dim)
    logical_row = torch.zeros(1, heads, 1, keys)
    logical_row[:, 0, :, 2:] = -1.0e4
    device_mask = logical_row.transpose(1, 2).contiguous()

    golden = ttnn.get_golden_function(ttnn.transformer.scaled_dot_product_attention_decode)
    actual = golden(
        query,
        key,
        value,
        is_causal=False,
        attn_mask=device_mask,
        cur_pos=[keys - 1] * batch,
        scale=1.0,
    )

    expected = torch.nn.functional.scaled_dot_product_attention(
        query.permute(1, 2, 0, 3),
        key.repeat_interleave(heads, dim=1),
        value.repeat_interleave(heads, dim=1),
        attn_mask=logical_row.expand(batch, -1, -1, -1),
        scale=1.0,
    ).permute(2, 0, 1, 3)
    torch.testing.assert_close(actual, expected)


def test_sdpa_decode_golden_keeps_query_batch_with_shared_cache():
    torch.manual_seed(14)
    batch, heads, dim, keys = 2, 32, 64, 64
    query = torch.randn(1, batch, heads, dim)
    key = torch.randn(1, 1, keys, dim)
    value = torch.randn(1, 1, keys, dim)
    positions = [63, 31]

    golden = ttnn.get_golden_function(ttnn.transformer.scaled_dot_product_attention_decode)
    actual = golden(query, key, value, cur_pos=positions, share_cache=True)

    query_by_batch = query.permute(1, 2, 0, 3)
    expected_rows = [
        torch.nn.functional.scaled_dot_product_attention(
            query_by_batch[user : user + 1],
            key[..., : position + 1, :].repeat_interleave(heads, dim=1),
            value[..., : position + 1, :].repeat_interleave(heads, dim=1),
        )
        for user, position in enumerate(positions)
    ]
    expected = torch.cat(expected_rows).permute(2, 0, 1, 3)
    torch.testing.assert_close(actual, expected)


def test_paged_decode_golden_converts_circular_mask_layout():
    query = torch.tensor([[[[1.0], [2.0]]]])
    page_table = torch.tensor([[0, 1]], dtype=torch.int64)
    key_pages = torch.tensor([[[[4.0], [5.0]]], [[[2.0], [3.0]]]])
    value_pages = torch.tensor([[[[40.0], [50.0]]], [[[20.0], [30.0]]]])
    logical_mask = torch.zeros(1, 2, 1, 6)
    logical_mask[:, 0, :, 4] = -1.0e4
    logical_mask[:, 1, :, 5] = -1.0e4
    device_mask = logical_mask.transpose(1, 2).contiguous()

    golden = ttnn.get_golden_function(ttnn.transformer.paged_scaled_dot_product_attention_decode)
    actual = golden(
        query,
        key_pages,
        value_pages,
        page_table,
        cur_pos_tensor=torch.tensor([5]),
        sliding_window_size=2,
        cache_position_modulo=4,
        attn_mask=device_mask,
        scale=1.0,
    )

    gathered_key = torch.tensor([[[[4.0], [5.0]]]])
    gathered_value = torch.tensor([[[[40.0], [50.0]]]])
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.permute(1, 2, 0, 3),
        gathered_key.repeat_interleave(2, dim=1),
        gathered_value.repeat_interleave(2, dim=1),
        attn_mask=logical_mask[..., 4:6],
        scale=1.0,
    ).permute(2, 0, 1, 3)
    torch.testing.assert_close(actual, expected)


def test_ring_mla_golden_selects_cache_slot_when_batch_sizes_match():
    torch.manual_seed(3)
    query = torch.randn(4, 2, 3, 6)
    kv = torch.randn(4, 1, 3, 6)

    golden = ttnn.get_golden_function(ttnn.transformer.ring_mla)
    output, _ = golden(query, kv, head_dim_v=3, logical_n=3, kv_cache_batch_idx=2)

    key = kv[2:3, :, :3, :]
    value = key[..., :3]
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key.repeat_interleave(2, dim=1),
        value.repeat_interleave(2, dim=1),
        is_causal=True,
    )
    torch.testing.assert_close(output, expected)


def test_ring_mla_golden_uses_runtime_slot_and_logical_prefix():
    torch.manual_seed(9)
    query = torch.randn(1, 2, 2, 6)
    kv = torch.randn(4, 1, 8, 6)

    golden = ttnn.get_golden_function(ttnn.transformer.ring_mla)
    output, _ = golden(
        query,
        kv,
        head_dim_v=3,
        logical_n=8,
        slot_id=torch.tensor([1]),
        kv_actual_isl_tensor=torch.tensor([2]),
        kv_cache_num_layers=2,
        kv_cache_layer_idx=0,
    )

    key = kv[2:3, :, :4, :]
    value = key[..., :3]
    query_positions = torch.arange(2, 4).unsqueeze(1)
    key_positions = torch.arange(4).unsqueeze(0)
    causal_mask = (key_positions <= query_positions).reshape(1, 1, 2, 4)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key.repeat_interleave(2, dim=1),
        value.repeat_interleave(2, dim=1),
        attn_mask=causal_mask,
    )
    torch.testing.assert_close(output, expected)


def test_flash_mla_prefill_golden_casts_kv_to_query_dtype():
    torch.manual_seed(11)
    query = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 4, 8, dtype=torch.float32)

    golden = ttnn.get_golden_function(ttnn.transformer.flash_mla_prefill)
    actual = golden(query, key, head_dim_v=4, is_causal=True)

    key_bf16 = key.to(query.dtype).repeat_interleave(2, dim=1)
    value_bf16 = key[..., :4].to(query.dtype).repeat_interleave(2, dim=1)
    expected = torch.nn.functional.scaled_dot_product_attention(query, key_bf16, value_bf16, is_causal=True)
    torch.testing.assert_close(actual, expected)


def test_sparse_sdpa_msa_golden_treats_unsigned_masked_index_as_sentinel():
    torch.manual_seed(12)
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 1, 4, 4)
    value = torch.randn(1, 1, 4, 3)
    signed = torch.tensor([[[[0, -1], [1, -1], [0, 1]]]], dtype=torch.int64)
    unsigned = signed.clone()
    unsigned[unsigned < 0] = 0xFFFFFFFF

    golden = ttnn.get_golden_function(ttnn.transformer.sparse_sdpa_msa)
    signed_output = golden(query, key, value, signed, block_size=2, scale=1.0)
    unsigned_output = golden(query, key, value, unsigned, block_size=2, scale=1.0)
    torch.testing.assert_close(unsigned_output, signed_output)


def test_ring_ops_register_persistent_buffers_as_inplace_outputs():
    exp_ring = ttnn.transformer.exp_ring_joint_scaled_dot_product_attention
    ring_joint = ttnn.transformer.ring_joint_scaled_dot_product_attention
    ring_mla = ttnn.transformer.ring_mla

    assert "persistent_output_buffer_k" in exp_ring.output_tensor_kwarg_names
    assert "persistent_output_buffer_v" in exp_ring.output_tensor_kwarg_names
    assert "persistent_output_buffer_k" in ring_joint.output_tensor_kwarg_names
    assert "persistent_output_buffer_v" in ring_joint.output_tensor_kwarg_names
    assert "persistent_output_buffer_joint_k" in ring_joint.output_tensor_kwarg_names
    assert "persistent_output_buffer_joint_v" in ring_joint.output_tensor_kwarg_names
    assert "persistent_output_buffer_kv" in ring_mla.output_tensor_kwarg_names
