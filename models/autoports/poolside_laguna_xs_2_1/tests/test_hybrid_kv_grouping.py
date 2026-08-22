# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free tests for Laguna hybrid-KV group and tensor alias identities."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from models.autoports.poolside_laguna_xs_2_1.tt import generator_vllm as generator_vllm_module
from models.autoports.poolside_laguna_xs_2_1.tt.generator_vllm import LagunaForCausalLM
from models.autoports.poolside_laguna_xs_2_1.tt.kv_grouping import (
    LAGUNA_HYBRID_LAYER_KINDS,
    build_hybrid_kv_layout,
    build_laguna_hybrid_kv_layout,
    validate_per_layer_tensor_aliases,
)


def _laguna_kinds(num_layers=40):
    return ["full" if layer % 4 == 0 else "sliding" for layer in range(num_layers)]


def test_laguna_exact_four_group_layout_and_ten_tensor_aliases():
    layout = build_laguna_hybrid_kv_layout(_laguna_kinds())

    assert layout.groups == (
        tuple(range(0, 40, 4)),
        tuple(range(1, 40, 4)),
        tuple(range(2, 40, 4)),
        tuple(range(3, 40, 4)),
    )
    assert layout.num_groups == 4
    assert layout.num_tensors == 10
    assert layout.representative_layers == (0, 1, 2, 3)
    for layer, alias in enumerate(layout.aliases):
        assert alias.layer_index == layer
        assert alias.group_id == layer % 4
        assert alias.tensor_index == layer // 4


def test_laguna_production_layout_fails_closed_for_reduced_or_changed_stacks(expect_error):
    assert tuple(_laguna_kinds()) == LAGUNA_HYBRID_LAYER_KINDS
    with expect_error(ValueError, "exact 40-layer"):
        build_laguna_hybrid_kv_layout(_laguna_kinds(4))
    changed = _laguna_kinds()
    changed[17] = "full"
    with expect_error(ValueError, "exact 40-layer"):
        build_laguna_hybrid_kv_layout(changed)


def test_group_page_tables_expand_in_logical_layer_order():
    layout = build_hybrid_kv_layout(_laguna_kinds())
    group_tables = [object() for _ in range(4)]

    expanded = layout.expand_group_values(group_tables)

    assert len(expanded) == 40
    assert all(expanded[layer] is group_tables[layer % 4] for layer in range(40))


def test_plugin_tensor_alias_contract_collapses_40_layers_to_10_buffers():
    layout = build_hybrid_kv_layout(_laguna_kinds())
    dtype = object()
    shape = (2048, 2, 64, 128)
    specs = [(shape, dtype, layer // 4) for layer in range(40)]

    unique = validate_per_layer_tensor_aliases(specs, layout)

    assert len(unique) == 10
    assert all(descriptor == (shape, dtype) for descriptor in unique.values())


def test_tensor_alias_validation_fails_closed_on_plugin_drift(expect_error):
    layout = build_hybrid_kv_layout(_laguna_kinds())
    dtype = object()
    specs = [((2048, 2, 64, 128), dtype, layer // 4) for layer in range(40)]
    specs[17] = (specs[17][0], dtype, 9)
    with expect_error(ValueError, "plugin tensor_idx"):
        validate_per_layer_tensor_aliases(specs, layout)

    specs = [((2048, 2, 64, 128), dtype, layer // 4) for layer in range(40)]
    specs[4] = ((1024, 2, 64, 128), dtype, 1)
    with expect_error(ValueError, "inconsistent descriptors"):
        validate_per_layer_tensor_aliases(specs, layout)


def test_grouping_matches_vllm_close_count_padding_heuristic():
    kinds = ["a"] * 12 + ["b"] * 13
    layout = build_hybrid_kv_layout(kinds)

    assert layout.num_groups == 2
    assert tuple(len(group) for group in layout.groups) == (12, 13)
    assert layout.num_tensors == 13


def test_grouping_validates_inputs(expect_error):
    with expect_error(ValueError, "at least one"):
        build_hybrid_kv_layout([])
    with expect_error(ValueError, "non-empty"):
        build_hybrid_kv_layout(["full", ""])


def _hybrid_bridge():
    bridge = object.__new__(LagunaForCausalLM)
    bridge._HYBRID_KV_CACHE_GROUPS_ENABLED = True
    bridge._PREFIX_CACHE_ENABLED = False
    bridge._layer_kinds = _laguna_kinds()
    bridge._hybrid_layout_cache = None
    bridge._kv_dtype = object()
    bridge.mesh_device = object()
    bridge._decode = {"stale": object()}
    bridge._verify_dec = {"stale": object()}
    bridge._report_dram = lambda *_args, **_kwargs: None
    return bridge


def _allocate_hybrid_cache(monkeypatch, *, num_blocks=2):
    bridge = _hybrid_bridge()
    allocations = []

    def fake_from_torch(tensor, **_kwargs):
        allocation = SimpleNamespace(index=len(allocations), shape=tuple(tensor.shape))
        allocations.append(allocation)
        return allocation

    monkeypatch.setattr(generator_vllm_module.ttnn, "from_torch", fake_from_torch)
    monkeypatch.setattr(generator_vllm_module, "_replicate", lambda _mesh: None)
    dtype = object()
    specs = [((num_blocks, 1, 1, 1), dtype, layer // 4) for layer in range(40)]
    return bridge, bridge.allocate_kv_cache_per_layer(specs), allocations


def test_hybrid_allocator_creates_ten_physical_pairs_and_forty_exact_aliases(monkeypatch):
    bridge, cache, allocations = _allocate_hybrid_cache(monkeypatch)

    assert len(allocations) == 20
    assert {allocation.shape for allocation in allocations} == {(3, 1, 1, 1)}
    assert len(cache) == 40
    for layer, entry in enumerate(cache):
        assert entry["logical_layer_idx"] == layer
        assert entry["hybrid_group_id"] == layer % 4
        assert entry["hybrid_kind"] == _laguna_kinds()[layer]
        assert entry["tensor_idx"] == layer // 4
        assert entry["blocks_per_user"] == 2
        assert entry["scratch_block_idx"] == 2
        assert entry["k"] is cache[(layer // 4) * 4]["k"]
        assert entry["v"] is cache[(layer // 4) * 4]["v"]
    assert cache[0]["k"] is cache[3]["k"]
    assert cache[0]["k"] is not cache[4]["k"]
    assert bridge._kv_cache_is_hybrid(cache) is True
    assert bridge._decode == {}
    assert bridge._verify_dec == {}


def test_qualified_pool_keeps_2460_vllm_ids_and_adds_private_2461st_row(monkeypatch):
    _bridge, cache, allocations = _allocate_hybrid_cache(monkeypatch, num_blocks=2460)

    assert {allocation.shape for allocation in allocations} == {(2461, 1, 1, 1)}
    assert {entry["blocks_per_user"] for entry in cache} == {2460}
    assert {entry["scratch_block_idx"] for entry in cache} == {2460}


def test_hybrid_cache_validation_rejects_metadata_or_physical_alias_drift(monkeypatch, expect_error):
    bridge, cache, _ = _allocate_hybrid_cache(monkeypatch)
    cache[17]["hybrid_group_id"] = 0
    with expect_error(ValueError, "metadata drift"):
        bridge._kv_cache_is_hybrid(cache)

    bridge, cache, _ = _allocate_hybrid_cache(monkeypatch)
    cache[3]["k"] = object()
    with expect_error(ValueError, "not physically shared"):
        bridge._kv_cache_is_hybrid(cache)


def _group_tables(*, rows=1, width=3, delta_group=None):
    values = []
    for layer in range(40):
        value = layer % 4
        if delta_group is not None and layer % 4 == delta_group:
            value += 10
        values.append(torch.full((rows, width), value, dtype=torch.int32))
    return values


def test_prefill_grouped_page_tables_upload_four_buffers_and_expand_by_identity(monkeypatch):
    bridge = _hybrid_bridge()
    bridge._pf_pt_groups = {}
    copies = []
    next_buffer = iter(range(100, 104))
    bridge.gen = SimpleNamespace(
        _rep=lambda tensor, _dtype: SimpleNamespace(id=next(next_buffer), shape=tuple(tensor.shape)),
        _host=lambda tensor, _dtype: tensor.clone(),
    )
    monkeypatch.setattr(
        generator_vllm_module.ttnn,
        "copy_host_to_device_tensor",
        lambda host, device: copies.append((host.clone(), device)),
    )

    expanded = bridge._prefill_pt_grouped(_group_tables())

    assert len(bridge._pf_pt_groups) == 4
    assert len(copies) == 4
    assert len(expanded) == 40
    for layer in range(40):
        assert expanded[layer] is expanded[layer % 4]
    assert len({id(expanded[group]) for group in range(4)}) == 4


def test_grouped_page_tables_fail_closed_on_length_or_intragroup_drift(expect_error):
    bridge = _hybrid_bridge()
    with expect_error(ValueError, "39 entries"):
        bridge._validated_group_page_tables(_group_tables()[:-1], purpose="decode")

    tables = _group_tables()
    tables[5][0, 0] = 99
    with expect_error(ValueError, "disagree within hybrid group 1"):
        bridge._validated_group_page_tables(tables, purpose="decode")


def test_decode_grouped_page_tables_refresh_only_changed_groups(monkeypatch):
    bridge = _hybrid_bridge()
    bridge.gen = SimpleNamespace(
        _rep=lambda tensor, _dtype: SimpleNamespace(shape=tuple(tensor.shape)),
        counters={"page_table_refresh": 0},
    )
    bridge._page_table_to_device_host = lambda tensor: tensor.clone()
    copies = []
    monkeypatch.setattr(
        generator_vllm_module.ttnn,
        "copy_host_to_device_tensor",
        lambda host, device: copies.append((host.clone(), device)),
    )

    expanded, groups, reps = bridge._decode_pt_grouped_alloc(_group_tables(rows=2))
    state = {
        "pt_groups": groups,
        "pt_reps": reps,
        "last_pt_host_groups": {},
    }
    assert len(groups) == 4
    assert all(expanded[layer] is groups[layer % 4] for layer in range(40))

    bridge._decode_pt_grouped_refresh(state, _group_tables(rows=2))
    assert len(copies) == 4
    assert bridge.gen.counters["page_table_refresh"] == 4
    bridge._decode_pt_grouped_refresh(state, _group_tables(rows=2))
    assert len(copies) == 4

    bridge._decode_pt_grouped_refresh(state, _group_tables(rows=2, delta_group=2))
    assert len(copies) == 5
    assert bridge.gen.counters["page_table_refresh"] == 5


def test_hybrid_cache_and_per_layer_tables_must_travel_together(monkeypatch, expect_error):
    bridge, cache, _ = _allocate_hybrid_cache(monkeypatch)
    with expect_error(ValueError, "hybrid KV cache but uniform page tables"):
        bridge._validate_page_table_mode(cache, None, operation="decode")
    with expect_error(ValueError, "uniform KV cache but per-layer page tables"):
        bridge._validate_page_table_mode(
            [{"block_size": 64}],
            _group_tables(),
            operation="decode",
        )


def _vllm_config(layer_types=None, *, num_layers=40, sliding_window=512):
    raw_types = (
        ["full_attention" if kind == "full" else "sliding_attention" for kind in _laguna_kinds()]
        if layer_types is None
        else layer_types
    )
    hf_config = SimpleNamespace(
        num_hidden_layers=num_layers,
        layer_types=raw_types,
        sliding_window=sliding_window,
    )
    model_config = SimpleNamespace(
        hf_config=hf_config,
        dtype=torch.bfloat16,
        get_num_kv_heads=lambda _parallel_config: 8,
        get_head_size=lambda: 128,
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=64),
        parallel_config=SimpleNamespace(),
    )


def test_get_kv_cache_spec_is_opt_in_and_emits_exact_ten_full_thirty_sliding(monkeypatch):
    monkeypatch.setattr(LagunaForCausalLM, "_PREFIX_CACHE_ENABLED", False)
    monkeypatch.setattr(LagunaForCausalLM, "_HYBRID_KV_CACHE_GROUPS_ENABLED", False)
    assert LagunaForCausalLM.get_kv_cache_spec(_vllm_config()) is None

    monkeypatch.setattr(LagunaForCausalLM, "_HYBRID_KV_CACHE_GROUPS_ENABLED", True)
    spec = LagunaForCausalLM.get_kv_cache_spec(_vllm_config())
    assert list(spec) == [f"model.layers.{layer}.self_attn" for layer in range(40)]
    names = [type(value).__name__ for value in spec.values()]
    assert names.count("FullAttentionSpec") == 10
    assert names.count("SlidingWindowSpec") == 30
    assert all(
        type(spec[f"model.layers.{layer}.self_attn"]).__name__
        == ("FullAttentionSpec" if layer % 4 == 0 else "SlidingWindowSpec")
        for layer in range(40)
    )


def test_get_kv_cache_spec_rejects_prefix_overlap_and_checkpoint_drift(monkeypatch, expect_error):
    monkeypatch.setattr(LagunaForCausalLM, "_HYBRID_KV_CACHE_GROUPS_ENABLED", True)
    monkeypatch.setattr(LagunaForCausalLM, "_PREFIX_CACHE_ENABLED", True)
    with expect_error(RuntimeError, "cannot be combined"):
        LagunaForCausalLM.get_kv_cache_spec(_vllm_config())

    monkeypatch.setattr(LagunaForCausalLM, "_PREFIX_CACHE_ENABLED", False)
    changed = _vllm_config().model_config.hf_config.layer_types.copy()
    changed[17] = "full_attention"
    with expect_error(ValueError, "exact 40-layer"):
        LagunaForCausalLM.get_kv_cache_spec(_vllm_config(changed))

    unknown = _vllm_config().model_config.hf_config.layer_types.copy()
    unknown[0] = "mystery_attention"
    with expect_error(ValueError, "unknown layer_types"):
        LagunaForCausalLM.get_kv_cache_spec(_vllm_config(unknown))
