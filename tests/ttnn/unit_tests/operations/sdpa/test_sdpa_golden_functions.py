# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from ttnn.operations.transformer import _preprocess_sparse_sdpa_golden_inputs
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


@pytest.mark.parametrize("use_keyword_kv", [False, True], ids=["positional-kv", "keyword-kv"])
def test_sparse_sdpa_wrapper_pipeline_preserves_packed_fp8_for_global_golden(use_keyword_kv, monkeypatch):
    # Exercise the wrapper's local-to-global preprocessing path so packed mixed-format KV rows remain raw bytes
    # even though ordinary global FP8 inputs are value-converted.
    class TransportTensor:
        def __init__(self, value, dtype):
            self.value = value
            self.dtype = dtype
            self.tensor_id = ttnn._ttnn.fetch_and_increment_tensor_id()

        @property
        def shape(self):
            return self.value.shape

        def device(self):
            return None

        def tensor_topology(self):
            raise RuntimeError("Transport stand-in has no mesh topology")

    torch.manual_seed(19)
    latent_dim, rope_dim, tokens = 512, 64, 3
    latent = torch.randn(1, 1, tokens, latent_dim).to(torch.float8_e4m3fn)
    scales = torch.ones(1, 1, tokens, latent_dim // 128)
    rope = torch.randn(1, 1, tokens, rope_dim).to(torch.bfloat16)
    packed = _pack_scaled_fp8_kv(latent, scales, rope)
    query = torch.randn(1, 2, 1, latent_dim + rope_dim).to(torch.bfloat16)
    indices = torch.tensor([[[[0, 2]]]], dtype=torch.int64)
    query_transport = TransportTensor(query, ttnn.bfloat16)
    kv_transport = TransportTensor(packed, ttnn.DataType.FP8_E4M3)
    indices_transport = TransportTensor(indices, ttnn.uint32)

    monkeypatch.setattr(ttnn, "Tensor", TransportTensor)
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda _: [])
    monkeypatch.setattr(ttnn, "is_tensor_storage_on_device", lambda _: False)
    monkeypatch.setattr(ttnn, "to_torch", lambda tensor, **_: tensor.value)
    monkeypatch.setattr(
        ttnn,
        "to_dtype",
        lambda tensor, dtype: TransportTensor(tensor.value.float(), dtype),
    )

    common_kwargs = {"kv_format": ttnn.transformer.SparseKVFormat.SCALED_FP8}
    if use_keyword_kv:
        function_args = ()
        function_kwargs = {
            "q": query_transport,
            "kv": kv_transport,
            "indices": indices_transport,
            "v_dim": latent_dim,
            **common_kwargs,
        }
    else:
        function_args = (query_transport, kv_transport, indices_transport, latent_dim)
        function_kwargs = common_kwargs

    tensor_ids = [query_transport.tensor_id, kv_transport.tensor_id, indices_transport.tensor_id]
    try:
        local_inputs = _preprocess_sparse_sdpa_golden_inputs(function_args, function_kwargs)
        with ttnn.manage_config("report_name", "sparse_sdpa_fp8_global_golden"):
            global_inputs = ttnn.decorators.preprocess_global_golden_function_inputs(function_args, function_kwargs)
        global_inputs = ttnn.decorators._merge_local_golden_metadata_into_global_inputs(local_inputs, global_inputs)
        local_args, local_kwargs = local_inputs
        global_args, global_kwargs = global_inputs

        assert global_kwargs["_ttnn_sparse_sdpa_packed_kv"].dtype == torch.float8_e4m3fn
        value_converted_global_kv = global_kwargs["kv"] if use_keyword_kv else global_args[1]
        assert value_converted_global_kv.dtype == torch.float32

        golden = ttnn.get_golden_function(ttnn.transformer.sparse_sdpa)
        local_output = golden(*local_args, **local_kwargs)
        global_output = golden(*global_args, **global_kwargs)
        torch.testing.assert_close(global_output, local_output, rtol=0, atol=0)
    finally:
        for tensor_id in tensor_ids:
            ttnn.decorators.TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR.pop(tensor_id, None)


def test_sparse_sdpa_golden_rejects_value_converted_scaled_fp8_kv(expect_error):
    query = torch.randn(1, 1, 1, 576)
    packed = torch.zeros(1, 1, 2, 656)
    indices = torch.tensor([[[[0, 1]]]], dtype=torch.int64)

    golden = ttnn.get_golden_function(ttnn.transformer.sparse_sdpa)
    with expect_error(ValueError, "packed bytes"):
        golden(query, packed, indices, 512, kv_format=ttnn.transformer.SparseKVFormat.SCALED_FP8)


@pytest.mark.parametrize(
    "mesh_shape, chunk_local, cache_tp_sharded, stripes, stripe_chunk",
    [
        ((2, 1), 32, False, 2, 32),
        ((2, 2), 64, True, 4, 32),
    ],
    ids=["sp-only", "sp-tp-sharded"],
)
def test_sparse_sdpa_golden_remaps_natural_indices_for_block_cyclic_cache(
    mesh_shape, chunk_local, cache_tp_sharded, stripes, stripe_chunk
):
    # Encode token IDs in a physically block-cyclic cache to prove natural sparse indices are remapped before
    # gathering, including the finer SP×TP-striped layout.
    cache_length = 256
    natural_tokens = torch.arange(cache_length, dtype=torch.float32)
    natural_cache = torch.stack((natural_tokens, torch.zeros_like(natural_tokens)), dim=-1)
    natural_indices = torch.arange(cache_length, dtype=torch.int64)
    block_index = natural_indices // stripe_chunk
    slab = block_index // stripes
    stripe = block_index.remainder(stripes)
    shard_length = cache_length // stripes
    physical_indices = natural_indices + stripe * (shard_length - stripe_chunk) - slab * stripe_chunk * (stripes - 1)
    physical_cache = torch.empty_like(natural_cache)
    physical_cache[physical_indices] = natural_cache

    query = torch.zeros(1, 1, 1, 2)
    indices = torch.tensor([[[[32, 0xFFFFFFFF]]]], dtype=torch.int64)
    golden = ttnn.get_golden_function(ttnn.transformer.sparse_sdpa)
    actual = golden(
        query,
        physical_cache.reshape(1, 1, cache_length, 2),
        indices,
        1,
        kv_format=ttnn.transformer.SparseKVFormat.BF16,
        scale=1.0,
        block_cyclic_sp_axis=0,
        block_cyclic_chunk_local=chunk_local,
        block_cyclic_cache_tp_sharded=cache_tp_sharded,
        _ttnn_sparse_sdpa_mesh_shape=mesh_shape,
    )

    torch.testing.assert_close(actual, torch.tensor([[[[32.0]]]]))


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


@pytest.mark.parametrize(
    "operation",
    [
        ttnn.transformer.ring_joint_scaled_dot_product_attention,
        ttnn.transformer.exp_ring_joint_scaled_dot_product_attention,
    ],
    ids=["ring-joint", "experimental-ring-joint"],
)
def test_ring_joint_golden_uses_sink_logits_and_marks_stats_as_scratch(operation):
    # Keep validating sink-aware attention numerics while asserting that the non-semantic device stats output is
    # represented by a correctly shaped, explicitly skipped scratch tensor.
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [1.0, -1.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, -1.0]]]])
    value = torch.tensor([[[[2.0, 1.0], [4.0, 3.0]], [[1.0, 5.0], [2.0, 6.0]]]])
    sink = torch.tensor([[[[0.25]], [[-0.5]]]])
    scale = 0.75

    golden = ttnn.get_golden_function(operation)
    output, joint_output, stats = golden(
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

    torch.testing.assert_close(output, expected_output)
    assert joint_output.shape[-2] == 0
    assert stats.shape == (1, 2, 64, 1)
    assert stats._ttnn_comparison_config.method == "skip"
    assert stats._ttnn_comparison_config.scope == "all"


def test_ring_joint_stats_scratch_uses_padded_query_and_joint_lengths():
    # The device allocates running-max/running-sum scratch from tile-padded Q and joint lengths; comparison must
    # skip only that leaf while continuing to check both semantic outputs.
    torch.manual_seed(17)
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)
    joint_query = torch.randn(1, 2, 5, 4)
    joint_key = torch.randn(1, 2, 5, 4)
    joint_value = torch.randn(1, 2, 5, 4)

    golden = ttnn.get_golden_function(ttnn.transformer.ring_joint_scaled_dot_product_attention)
    output, joint_output, stats = golden(
        query,
        key,
        value,
        joint_query,
        joint_key,
        joint_value,
        logical_n=3,
        logical_l=5,
    )

    assert output.shape[-2] == 3
    assert joint_output.shape[-2] == 5
    assert stats.shape == (1, 2, 128, 1)
    golden_outputs = (output, joint_output, stats)
    runtime_outputs = (output.clone(), joint_output.clone(), torch.ones(1, 2, 1, 1))
    ttnn.decorators.set_tensor_id(ttnn.decorators.get_all_tensors(golden_outputs), force=True)
    ttnn.decorators.set_tensor_id(ttnn.decorators.get_all_tensors(runtime_outputs), force=True)
    records = ttnn.decorators.compare_tensors_using_pcc(
        "ttnn.transformer.ring_joint_scaled_dot_product_attention",
        golden_outputs,
        runtime_outputs,
        desired_pcc=0.99,
        level="locally",
        fail_on_bad_comparison=True,
    )
    assert len(records) == 2


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


@pytest.mark.parametrize("paged", [False, True], ids=["unpaged", "paged-circular"])
@pytest.mark.parametrize(
    "positions",
    [(1, 1), (1, -1), (-1, -1)],
    ids=["active", "mixed", "inactive"],
)
def test_mla_decode_golden_sizes_inactive_rows_from_value_width(paged, positions):
    # MLA uses wider Q/K than V, so skipped users must emit V-width zeros that concatenate with active rows for
    # both unpaged and paged-circular decode.
    torch.manual_seed(18)
    batch, heads, query_width, value_width, cache_length = 2, 2, 576, 512, 2
    query = torch.randn(1, batch, heads, query_width)
    key = torch.randn(batch, 1, cache_length, query_width)
    value = torch.randn(batch, 1, cache_length, value_width)
    positions_tensor = torch.tensor(positions)

    if paged:
        page_table = torch.tensor([[0], [1]], dtype=torch.int64)
        golden = ttnn.get_golden_function(ttnn.transformer.paged_flash_multi_latent_attention_decode)
        actual = golden(
            query,
            key,
            value,
            head_dim_v=value_width,
            page_table_tensor=page_table,
            cur_pos_tensor=positions_tensor,
            cache_position_modulo=cache_length,
            sliding_window_size=cache_length,
        )
    else:
        golden = ttnn.get_golden_function(ttnn.transformer.flash_multi_latent_attention_decode)
        actual = golden(
            query,
            key,
            value,
            head_dim_v=value_width,
            cur_pos_tensor=positions_tensor,
        )

    query_by_batch = query.permute(1, 2, 0, 3)
    expected_rows = []
    for user, position in enumerate(positions):
        if position < 0:
            expected_rows.append(query.new_zeros((1, heads, 1, value_width)))
        else:
            expected_rows.append(
                torch.nn.functional.scaled_dot_product_attention(
                    query_by_batch[user : user + 1],
                    key[user : user + 1, :, : position + 1].repeat_interleave(heads, dim=1),
                    value[user : user + 1, :, : position + 1].repeat_interleave(heads, dim=1),
                )
            )
    expected = torch.cat(expected_rows).permute(2, 0, 1, 3)

    assert actual.shape == (1, batch, heads, value_width)
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
    # Validate indexed cache-slot selection and ensure ring MLA exposes its second output as skipped device scratch
    # rather than a fabricated final LSE.
    torch.manual_seed(3)
    query = torch.randn(4, 2, 3, 6)
    kv = torch.randn(4, 1, 3, 6)

    golden = ttnn.get_golden_function(ttnn.transformer.ring_mla)
    output, stats = golden(query, kv, head_dim_v=3, logical_n=3, kv_cache_batch_idx=2)

    key = kv[2:3, :, :3, :]
    value = key[..., :3]
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key.repeat_interleave(2, dim=1),
        value.repeat_interleave(2, dim=1),
        is_causal=True,
    )
    torch.testing.assert_close(output, expected)
    assert stats.shape == (4, 2, 64, 1)
    assert stats._ttnn_comparison_config.method == "skip"


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
