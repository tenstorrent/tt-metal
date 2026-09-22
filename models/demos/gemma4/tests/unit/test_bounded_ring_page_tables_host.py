# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Check real Gemma page-table sizing with host cache-layout handles."""

import sys
from types import SimpleNamespace

import pytest
import torch

# Pytest discovers these imported fixtures by name.
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import adapter  # noqa: F401
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import expect_error  # noqa: F401
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import DeviceShape


def _cache(blocks=288, heads=2, block_size=64, head_dim=256):
    shape = DeviceShape((blocks, heads, block_size, head_dim))
    return SimpleNamespace(shape=shape, padded_shape=shape)


def _attention(modulo=2048, heads=16, head_dim=256, tp=8, replicated=False):
    return SimpleNamespace(
        config=SimpleNamespace(cache_position_modulo=modulo, num_key_value_heads=heads, head_dim=head_dim),
        weights=SimpleNamespace(kv_replicated=replicated),
        mesh_config=SimpleNamespace(tp=tp),
    )


@pytest.fixture
def ring_model(adapter, monkeypatch):
    model = adapter.Gemma4ForCausalLM.__new__(adapter.Gemma4ForCausalLM)
    model._bounded_sliding_kv_cache = True
    sliding, full = _attention(), _attention(modulo=None, heads=8, head_dim=512)
    model.model = [
        SimpleNamespace(
            layers=[SimpleNamespace(self_attn=sliding), SimpleNamespace(self_attn=full)],
            hf_config=SimpleNamespace(sliding_window=1024, layer_types=["sliding_attention", "full_attention"]),
        )
    ]
    model.model_args = [SimpleNamespace(max_batch_size=32, max_seq_len=4096)]
    model.kv_cache = [[[_cache(), _cache()], [_cache(heads=1, head_dim=512), _cache(heads=1, head_dim=512)]]]
    for layer, pair in zip(model.model[0].layers, model.kv_cache[0]):
        layer.self_attn.kv_cache = pair
    common = sys.modules["models.tt_transformers.tt.common"]
    monkeypatch.setattr(common, "num_blocks_in_seq", lambda length, block: (length + block - 1) // block, raising=False)
    monkeypatch.setattr(common, "get_block_size", lambda cache: cache[0][0].shape[2], raising=False)
    return model


@pytest.mark.parametrize("modulo,expected", [(1024, 16), (2048, 32), (4096, 64)])
def test_minimum_columns_follow_actual_ring_not_hf_window(ring_model, modulo, expected):
    ring_model.model[0].layers[0].self_attn.config.cache_position_modulo = modulo
    assert ring_model._text_config().sliding_window == 1024
    assert ring_model._bounded_sliding_min_page_table_cols(ring_model.kv_cache[0]) == expected


def test_hybrid_allocation_uses_effective_heads_and_block_size(ring_model):
    ring_model.kv_cache[0][0] = [_cache(heads=1, block_size=128), _cache(heads=1, block_size=128)]
    assert ring_model._bounded_sliding_min_page_table_cols(ring_model.kv_cache[0]) == 32


def test_replicated_kv_heads_use_the_actual_local_view(ring_model):
    attention = ring_model.model[0].layers[0].self_attn
    attention.weights.kv_replicated = True
    attention.config.num_key_value_heads = 4
    ring_model.kv_cache[0][0] = [_cache(heads=1), _cache(heads=1)]
    assert ring_model._bounded_sliding_min_page_table_cols(ring_model.kv_cache[0]) == 32


@pytest.mark.parametrize("disabled", ["unbounded", "no_cache", "no_modulo"])
def test_inactive_bounded_sizing_returns_none(ring_model, disabled):
    cache = ring_model.kv_cache[0]
    if disabled == "unbounded":
        ring_model._bounded_sliding_kv_cache = False
    elif disabled == "no_cache":
        cache = None
    else:
        ring_model.model[0].layers[0].self_attn.config.cache_position_modulo = None
    assert ring_model._bounded_sliding_min_page_table_cols(cache) is None


def test_invalid_ring_multiple_fails_before_device_work(ring_model, expect_error):
    ring_model.model[0].layers[0].self_attn.config.cache_position_modulo = 1030
    with expect_error(ValueError, match="positive multiple"):
        ring_model._bounded_sliding_min_page_table_cols(ring_model.kv_cache[0])


def test_single_user_prefill_keeps_complete_headroom_ring(ring_model):
    source = torch.arange(64, dtype=torch.int32).reshape(1, 64)
    actual = ring_model._get_prefill_user_page_table(source, ring_model.kv_cache[0], 128, prefill_seq_len=128)
    assert actual.shape == (1, 32)
    assert torch.equal(actual, source[:, :32])


def test_short_table_is_padded_to_ring_without_changing_existing_ids(ring_model):
    source = torch.tensor([[7, 8]], dtype=torch.int32)
    actual = ring_model._get_prefill_user_page_table(source, ring_model.kv_cache[0], 128, prefill_seq_len=128)
    assert actual.shape == (1, 32)
    assert actual[0, :2].tolist() == [7, 8]
    assert not actual[0, 2:].any()
    assert source.tolist() == [[7, 8]]


def test_batched_prefill_keeps_ring_width_and_original_slot_placement(ring_model):
    source = torch.arange(128, dtype=torch.int32).reshape(2, 64)
    actual = ring_model._get_prefill_user_page_table(
        source,
        ring_model.kv_cache[0],
        [128, 128],
        prefill_seq_len=128,
        use_batched_prefill=True,
        user_id=[1, 3],
        padded_batch_size=4,
    )
    assert actual.shape == (4, 32)
    assert torch.equal(actual[[1, 3]], source[:, :32])
    assert not actual[[0, 2]].any()


def test_unbounded_prefill_keeps_existing_short_prompt_width(ring_model):
    ring_model._bounded_sliding_kv_cache = False
    source = torch.arange(64, dtype=torch.int32).reshape(1, 64)
    actual = ring_model._get_prefill_user_page_table(source, ring_model.kv_cache[0], 128, prefill_seq_len=128)
    assert actual.shape == (1, 4)
    assert torch.equal(actual, source[:, :4])


def test_warmup_global_width_and_full_cache_layout_remain_unchanged(ring_model):
    full_cache = ring_model.kv_cache[0][1][0]
    before = tuple(full_cache.shape)
    warmup = ring_model._mock_tokens(1, 128, ring_model.kv_cache, 0)
    assert warmup["page_table"].shape == (1, 64)
    assert not warmup["page_table"].any()
    page_table = ring_model._get_prefill_user_page_table(
        warmup["page_table"], ring_model.kv_cache[0], 128, prefill_seq_len=128
    )
    assert page_table.shape == (1, 32)
    assert tuple(full_cache.shape) == before == (288, 1, 64, 512)


@pytest.mark.parametrize("ring,columns", [(1024, 16), (2048, 32)])
def test_serving_slot_stride_matches_the_actual_ring(ring_model, ring, columns):
    ring_model.model[0].layers[0].self_attn.config.cache_position_modulo = ring
    source = torch.tensor([[101, 102], [201, 202]], dtype=torch.int32)
    actual = ring_model._pad_sliding_page_tables_for_bounded([source, source], ring_model.kv_cache, authoritative=True)
    assert actual[0].tolist() == [list(range(columns)), list(range(columns, 2 * columns))]
    assert actual[1] is source
    assert ring_model._bounded_ring_slot_map == {101: 0, 201: 1}


def test_row_permutation_preserves_request_ring_blocks(ring_model):
    source = torch.tensor([[101, 102], [201, 202]], dtype=torch.int32)
    initial = ring_model._pad_sliding_page_tables_for_bounded([source, source], ring_model.kv_cache, authoritative=True)
    permuted = source[[1, 0]]
    actual = ring_model._pad_sliding_page_tables_for_bounded(
        [permuted, permuted], ring_model.kv_cache, authoritative=True
    )
    assert torch.equal(actual[0], initial[0][[1, 0]])
    assert ring_model._bounded_ring_slot_map == {101: 0, 201: 1}


def test_peer_prefill_keeps_existing_request_ring_ownership(ring_model):
    first = torch.tensor([[101, 102]], dtype=torch.int32)
    peer = torch.tensor([[201, 202]], dtype=torch.int32)
    original = ring_model._pad_sliding_page_tables_for_bounded([first, first], ring_model.kv_cache)
    joined = ring_model._pad_sliding_page_tables_for_bounded([peer, peer], ring_model.kv_cache)
    resumed = ring_model._pad_sliding_page_tables_for_bounded([first, first], ring_model.kv_cache)
    assert torch.equal(resumed[0], original[0])
    assert joined[0].tolist() == [list(range(32, 64))]


def test_null_rows_do_not_require_nominal_batch_capacity(ring_model):
    ring_model.kv_cache[0][0] = [_cache(blocks=64), _cache(blocks=64)]
    source = torch.zeros(32, 2, dtype=torch.int32)
    actual = ring_model._pad_sliding_page_tables_for_bounded([source, source], ring_model.kv_cache)
    assert actual[0].shape == (32, 32)
    assert not actual[0].any()
    assert ring_model._bounded_ring_slot_map == {}


def test_small_pool_accepts_fitting_live_rings_and_rejects_first_out_of_range_slot(ring_model, expect_error):
    ring_model.kv_cache[0][0] = [_cache(blocks=64), _cache(blocks=64)]
    source = torch.tensor([[101, 102], [201, 202]], dtype=torch.int32)
    actual = ring_model._pad_sliding_page_tables_for_bounded([source, source], ring_model.kv_cache)
    assert actual[0].max() == 63
    third = torch.tensor([[301, 302]], dtype=torch.int32)
    with expect_error(ValueError, match="slot 2 needs 96 blocks in layer 0.*64 blocks"):
        ring_model._pad_sliding_page_tables_for_bounded([third, third], ring_model.kv_cache)


def test_capacity_check_includes_value_cache(ring_model, expect_error):
    ring_model.kv_cache[0][0] = [_cache(blocks=64), _cache(blocks=32)]
    source = torch.tensor([[101, 102], [201, 202]], dtype=torch.int32)
    with expect_error(ValueError, match="slot 1 needs 64 blocks in layer 0.*32 blocks"):
        ring_model._pad_sliding_page_tables_for_bounded([source, source], ring_model.kv_cache)


def test_each_sliding_layer_uses_its_own_ring_and_cache_layout(ring_model):
    ring_model.model[0].hf_config.layer_types[1] = "sliding_attention"
    ring_model.model[0].layers[1].self_attn = _attention(modulo=4096)
    ring_model.kv_cache[0][1] = [_cache(block_size=128), _cache(block_size=128)]
    source = torch.tensor([[101, 102], [201, 202]], dtype=torch.int32)
    actual = ring_model._pad_sliding_page_tables_for_bounded([source, source], ring_model.kv_cache)
    assert actual[0].shape == actual[1].shape == (2, 32)
    assert torch.equal(actual[0], actual[1])
    ring_model.kv_cache[0][1] = [_cache(), _cache()]
    assert ring_model._bounded_sliding_min_page_table_cols(ring_model.kv_cache[0]) == 64
    actual = ring_model._pad_sliding_page_tables_for_bounded([source, source], ring_model.kv_cache)
    assert actual[0].shape == (2, 32)
    assert actual[1].shape == (2, 64)
    assert actual[1][1].tolist() == list(range(64, 128))


def test_serving_can_use_model_owned_cache_handles(ring_model):
    source = torch.tensor([[101, 102]], dtype=torch.int32)
    actual = ring_model._pad_sliding_page_tables_for_bounded([source, source], None)
    assert actual[0].tolist() == [list(range(32))]


def test_unbounded_serving_does_not_remap_or_create_request_slots(ring_model):
    ring_model._bounded_sliding_kv_cache = False
    source = torch.tensor([[101, 102]], dtype=torch.int32)
    tables = [source, source]
    assert ring_model._pad_sliding_page_tables_for_bounded(tables, ring_model.kv_cache) is tables
    assert not hasattr(ring_model, "_bounded_ring_slot_map")
