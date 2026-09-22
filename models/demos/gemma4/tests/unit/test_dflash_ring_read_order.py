# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks for contract-owned chronological packed ring reads.

Actual decoder, target routing, packed attention, and cleanup methods execute
with lower device operations stubbed. Physical membership and buffer ownership
are checked here; finite-precision device equivalence needs hardware tests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tests.unit.test_dflash_attention_policy import Tensor as AttentionTensor
from models.demos.gemma4.tests.unit.test_dflash_attention_policy import adapter as _adapter_fixture
from models.demos.gemma4.tests.unit.test_dflash_attention_policy import attention as _attention_fixture
from models.demos.gemma4.tests.unit.test_dflash_attention_policy import config, prepare_forward
from models.demos.gemma4.tests.unit.test_dflash_capture_cleanup import _model
from models.demos.gemma4.tests.unit.test_dflash_capture_cleanup import production as _cleanup_fixture
from models.demos.gemma4.tests.unit.test_dflash_width_prepare import Target, _drafter
from models.demos.gemma4.tests.unit.test_dflash_width_prepare import generator_module as _generator_fixture
from models.demos.gemma4.tests.unit.test_dflash_width_prepare import production as _production_fixture

adapter = _adapter_fixture
attention = _attention_fixture
cleanup = _cleanup_fixture
generator_module = _generator_fixture
production = _production_fixture


@pytest.fixture
def expect_error():
    return pytest.raises  # allow-pytest.raises: root conftest requires hardware runtime


def decoder_for(production, monkeypatch, *, selected=True, bounded=True):
    cls = production.module.DFlashFusedDecoder
    monkeypatch.setattr(cls, "_pv_upload", production.real_pv_upload)
    target = Target(production.runtime)
    target.hf_config = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention", "sliding_attention"], sliding_window=1024
    )
    target.layers = [
        SimpleNamespace(self_attn=SimpleNamespace(config=SimpleNamespace(cache_position_modulo=ring)))
        for ring in (2048 if bounded else None, None, 2048 if bounded else None)
    ]
    pages = torch.tensor([[1 + (13 * index) % 32 for index in range(32)]], dtype=torch.int32)
    flat = torch.arange(101, 165, dtype=torch.int32).reshape(1, -1)
    target._active_page_tables_per_layer = [pages, flat, pages]
    decoder = cls(target, _drafter(production.runtime), object(), flat, ctx_cap=32, rotate_ring_reads=selected)
    return decoder


def membership(table, mask):
    columns = table.reshape(-1).repeat_interleave(64)
    rows = torch.arange(columns.numel()) % 64
    return [
        {(int(block), int(row)) for block, row in zip(columns[query == 0], rows[query == 0])}
        for query in mask.reshape(6, -1)
    ]


@pytest.mark.parametrize("start", [0, 194, 1023, 2043, 2047, 2048, 2049, 2079, 2110, 2111, 2120, 2124, 4094])
def test_upload_preserves_physical_membership_for_all_six_queries(production, monkeypatch, start):
    decoder = decoder_for(production, monkeypatch)
    decoder._pv_ring_meta()
    record = decoder._pv_width_install(4096)
    natural = record["cache_by_type"]["sliding_attention"].data.clone()
    decoder._pv_upload(start)
    read = record["read_cache_by_type"]["sliding_attention"].data
    mask = record["pv_mask_slide"].data
    rotation = (max(0, start - 1024 + 1) // 64) % 32
    assert torch.equal(read, natural.roll(-rotation, -1))
    assert torch.equal(record["cache_by_type"]["sliding_attention"].data, natural)
    assert torch.equal(decoder._pv_host_masks(start)[1], mask)
    assert torch.equal(decoder.pv_widx_all.data, torch.arange(start, start + 6, dtype=torch.int32))
    visible = membership(read, mask)
    for offset, actual in enumerate(visible):
        query = start + offset
        expected = {
            (int(natural[0, (position % 2048) // 64]), position % 64)
            for position in range(max(0, query - 1024 + 1), query + 1)
        }
        assert actual == expected
    assert record["read_tables"][0] is record["read_tables"][2]
    assert record["read_tables"][1] is None
    assert record["read_tables"][0] is not record["pv_tables"][0]
    assert torch.isfinite(mask).all()
    assert set(mask.reshape(-1).tolist()) <= {0.0, float(torch.tensor(-1e9, dtype=torch.bfloat16))}


@pytest.mark.parametrize("selected,bounded", [(False, True), (False, False), (True, False)])
def test_legacy_and_unbounded_paths_have_no_read_allocation(production, monkeypatch, selected, bounded):
    decoder = decoder_for(production, monkeypatch, selected=selected, bounded=bounded)
    decoder._pv_ring_meta()
    record = decoder._pv_width_install(3072)
    assert record["read_tables"] is None and record["read_cache_by_type"] == {}
    before = record["cache_by_type"]["sliding_attention"].data.clone()
    decoder._pv_upload(2124)
    assert torch.equal(record["cache_by_type"]["sliding_attention"].data, before)
    if bounded:
        natural_mask = decoder._pv_host_masks(2124)[1]
        assert torch.equal(record["pv_mask_slide"].data, natural_mask)


def test_all_read_buffers_exist_before_any_body_or_capture(production, monkeypatch):
    decoder = decoder_for(production, monkeypatch)
    production.runtime.events.clear()
    decoder.prepare_widths([3072, 4096])
    first_body = next(i for i, event in enumerate(production.runtime.events) if event[0] == "body")
    allocations = [event[1] for event in production.runtime.events[:first_body] if event[0] == "allocate"]
    buffers = [record["read_tables"][0] for record in decoder._pv_widths.values()]
    assert len({id(buffer) for buffer in buffers}) == 2
    assert all(buffer in allocations for buffer in buffers)
    production.runtime.events.clear()
    decoder.capture_widths([3072, 4096])
    assert not any(event[0] == "allocate" for event in production.runtime.events)
    assert [record["read_tables"][0] for record in decoder._pv_widths.values()] == buffers


def test_owner_refresh_and_committed_width_change_use_current_natural_snapshot(production, monkeypatch):
    decoder = decoder_for(production, monkeypatch)
    decoder.capture_widths([3072, 4096])
    decoder.select_width(3001)
    decoder.start, decoder.anchor, decoder.ctx_len, decoder.win_first = 3001, 17, 3001, 2991
    new_row = torch.arange(301, 333, dtype=torch.int32).reshape(1, -1)
    decoder.target._active_page_tables_per_layer[0] = new_row
    decoder.target._active_page_tables_per_layer[2] = new_row
    new_flat = torch.arange(501, 565, dtype=torch.int32).reshape(1, -1)
    decoder.refresh_page_tables(new_flat)
    decoder._pv_upload(3001)
    assert torch.equal(decoder._pv_host_rows["sliding_attention"], new_row)
    new_row.add_(1000)
    assert not torch.equal(decoder._pv_host_rows["sliding_attention"], new_row)
    decoder._pv_upload(3001)
    assert set(decoder.pv_read_tables[0].data.flatten().tolist()) == set(range(301, 333))
    decoder.contract_commit(3, 42)
    assert decoder.start == 3004 and decoder.anchor == 42
    decoder.select_width(decoder.start)
    assert decoder.pv_sk == 4096
    production.runtime.events.clear()
    decoder._pv_upload(decoder.start)
    assert not any(event[0] == "allocate" for event in production.runtime.events)
    rotation = (3004 - 1024 + 1) // 64 % 32
    assert torch.equal(decoder.pv_read_tables[0].data, new_row.roll(-rotation, -1))
    assert torch.equal(decoder.pv_tables[0].data, new_row)
    assert torch.equal(decoder._pv_cache_by_type["full_attention"].data, new_flat)
    assert torch.equal(decoder.pv_mask_slide.data, decoder._pv_host_masks(3004)[1])


def test_failed_read_allocation_keeps_width_owned_and_retries_without_reallocating_writes(
    production, monkeypatch, expect_error
):
    decoder = decoder_for(production, monkeypatch)
    decoder._pv_ring_meta()
    native = production.tt.from_torch

    def fail_read(data, **kwargs):
        if kwargs.get("device") is not None and 3072 in decoder._pv_widths:
            raise RuntimeError("read allocation failed")
        return native(data, **kwargs)

    monkeypatch.setattr(production.tt, "from_torch", fail_read)
    with expect_error(RuntimeError, match="read allocation failed"):
        decoder.prepare_widths([3072])
    record = decoder._pv_widths[3072]
    writes = tuple(record["pv_tables"])
    assert record["read_cache_by_type"] == {} and record["trace"] is None
    assert not getattr(decoder, "_prepared_widths", set())
    monkeypatch.setattr(production.tt, "from_torch", native)
    decoder.prepare_widths([3072])
    assert tuple(record["pv_tables"]) == writes
    assert record["read_tables"][0] is not None


@pytest.mark.parametrize("destination", ["read", "mask", "positions"])
def test_upload_failure_frees_temporary_and_prevents_replay(production, monkeypatch, expect_error, destination):
    decoder = decoder_for(production, monkeypatch)
    decoder.capture_widths([3072])
    targets = {"read": decoder.pv_read_tables[0], "mask": decoder.pv_mask_slide, "positions": decoder.pv_widx_all}
    native = production.tt.copy_host_to_device_tensor
    created, replayed = [], []
    original_from = production.tt.from_torch

    def create(data, **kwargs):
        value = original_from(data, **kwargs)
        created.append(value)
        return value

    def copy(source, target):
        if target is targets[destination]:
            raise RuntimeError("upload failed")
        native(source, target)

    monkeypatch.setattr(production.tt, "from_torch", create)
    monkeypatch.setattr(production.tt, "copy_host_to_device_tensor", copy)
    monkeypatch.setattr(production.tt, "execute_trace", lambda *a, **kw: replayed.append(True), raising=False)
    monkeypatch.setattr(production.module.DFlashFusedDecoder, "_upload_iter_inputs", production.real_upload_iter_inputs)
    decoder.anchor, decoder.start = 3, 2124
    with expect_error(RuntimeError, match="upload failed"):
        decoder.contract_replay(first=True)
    assert created and all(value.freed for value in created)
    assert replayed == []


def test_read_buffers_survive_session_release_and_teardown_frees_once_after_traces(cleanup):
    model = _model(cleanup, gemma=True)
    events = cleanup.runtime.events

    class ReadBuffer:
        def deallocate(self, force):
            events.append(("free_read", self))

    first, second = ReadBuffer(), ReadBuffer()
    decoder = model._spec_decoder
    decoder._pv_widths[3072]["read_cache_by_type"] = {"sliding_attention": first, "alias": first}
    decoder._pv_widths[4096]["read_cache_by_type"] = {"sliding_attention": second}
    model._spec_release_decoder()
    assert events == []
    model.release_persistent_capture()
    assert events[:2] == [("release", "root-mesh", 30, False), ("release", "root-mesh", 31, False)]
    assert [event for event in events if event[0] == "free_read"] == [("free_read", first), ("free_read", second)]
    before = list(events)
    model.release_persistent_capture()
    cleanup.runtime.close("root-mesh")
    model.__del__()
    assert events == before + [("close", "root-mesh")]


@pytest.mark.parametrize("contract", [False, True])
def test_adapter_preparation_selects_read_order_only_for_contract(production, generator_module, contract):
    cls = generator_module.Gemma4DFlashContractForCausalLM if contract else generator_module.Gemma4DFlashForCausalLM
    model = cls.__new__(cls)
    model._spec_decoder = model._spec_width_ladder = None
    model._spec_horizon = 2048
    model.model_args = [SimpleNamespace(max_seq_len=4096)]
    model.model = [Target(production.runtime)]
    model._spec_get_drafter = lambda: _drafter(production.runtime)
    model._spec_capture_width_set(object(), 64, prepare_only=True)
    assert model._spec_decoder.rotate_ring_reads is contract


@pytest.mark.parametrize("dim", [256, 512])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("write_mode", ["shared", "fallback", "staging"])
def test_packed_attention_routes_only_sdpa_through_read_table(attention, monkeypatch, dim, selected, write_mode):
    inputs = prepare_forward(attention, monkeypatch, dim, 1, 6, packed=True)
    inputs["config"] = config(attention, dim, True)
    inputs["config"].cache_position_modulo = 2048 if dim == 256 else None
    api = attention.api
    api.slice = lambda value, starts, ends: AttentionTensor(
        value.name + ".slice", [b - a for a, b in zip(starts, ends)]
    )
    api.pad = lambda tensor, padding, **kw: AttentionTensor(
        tensor.name, [s + a + b for s, (a, b) in zip(tensor.shape, padding)]
    )
    api.permute = lambda tensor, order, **kw: AttentionTensor(tensor.name, [tensor.shape[index] for index in order])
    reads = AttentionTensor("rotated-pages", [1, 32]) if selected else None
    writes, fills = inputs["page_table"], []
    hot = object()
    if write_mode == "staging":
        inputs["kv_staging"] = inputs["kv_cache"]
        inputs["embed_idx"], inputs["hot_pt"] = object(), hot
        monkeypatch.setattr(attention.decode, "_packed_fill_kv_loopfree_embed", lambda *a: fills.append(a))
    attention.module.packed_decode_forward(
        **inputs,
        kv_write_idxs=[object() for _ in range(6)],
        attn_mask=AttentionTensor("mask", [1, 1, 24, 2048]),
        packed_p=6,
        is_kv_shared=write_mode == "shared",
        rope_packed=(inputs["cos_cache"], inputs["sin_cache"]),
        read_page_table=reads,
    )
    assert api.sdpa_calls[0][2]["page_table_tensor"] is (reads if selected else writes)
    assert all(kwargs["page_table"] is writes for args, kwargs in api.updates)
    assert len(api.updates) == (12 if write_mode == "fallback" else 0)
    assert len(fills) == (2 if write_mode == "staging" else 0)
    assert all(call[-1] is hot for call in fills)


@pytest.fixture
def model_module(attention, monkeypatch):
    for name, members in {
        "tracy": {"signpost": lambda *a, **kw: None},
        "models.common.sampling.generator": {"SamplingGenerator": object},
        "models.demos.gemma4.tt.attention": {"Gemma4AttentionConfig": object, "flush_deferred_bounded_fills": None},
        "models.demos.gemma4.tt.layer": {"Gemma4DecoderLayer": object},
        "models.demos.gemma4.tt.rms_norm": {"RMSNorm": object},
        "models.demos.gemma4.utils.general_utils": {"cast_host_for_ttnn": None, "get_cache_file_name": None},
        "models.demos.gemma4.utils.substate": {"substate": None},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[2] / "tt/model.py"
    spec = importlib.util.spec_from_file_location("gemma4_ring_read_model_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("selected", [False, True])
def test_target_forward_keeps_read_tables_out_of_ordinary_conversion_and_selects_each_layer(
    model_module, attention, selected
):
    target = model_module.Gemma4Model.__new__(model_module.Gemma4Model)
    target.mesh_device = object()
    target.hf_config = SimpleNamespace(layer_types=["sliding_attention", "full_attention"])
    target.kv_shared_layer_map, target.rope_caches_2d = {}, {}
    target.tt_kv_cache, target._dflash_tap_layers = None, None
    target._compute_per_layer_inputs = lambda *a: None
    target._get_rope_mats = lambda *a, **kw: (object(), object())
    target.embed_tokens = lambda x: AttentionTensor("embeds", [1, 1, 6, 5376])
    target.norm = SimpleNamespace(forward=lambda value: value)
    target._apply_lm_head = lambda value, **kw: value
    attention.api.clone = lambda value: AttentionTensor("clone", value.shape)
    converted, calls = [], []
    target._page_tables_to_ttnn = lambda tables: converted.append(tables) or tables
    target.layers = [lambda value, **kwargs: calls.append(kwargs) or value for _ in range(2)]
    writes = [object(), object()]
    reads = [object(), None] if selected else None
    target.ttnn_packed_verify_forward(
        x=object(),
        position_idx=object(),
        attn_mask_full=object(),
        attn_mask_sliding=object(),
        packed_p=6,
        kv_cache=[object(), object()],
        page_tables_per_layer=writes,
        read_page_tables_per_layer=reads,
    )
    assert converted == [writes]
    for index, call in enumerate(calls):
        assert call["page_table"] is writes[index]
        assert call["packed"]["read_page_table"] is (reads[index] if selected else None)


def test_attention_dispatch_forwards_read_table_without_replacing_write_table(attention, monkeypatch):
    cls = attention.module.Gemma4Attention
    instance = cls.__new__(cls)
    for name in ("weights", "config", "mesh_config", "mesh_device", "ccl_manager", "kv_staging"):
        setattr(instance, name, object())
    calls = []
    monkeypatch.setattr(attention.module, "packed_decode_forward", lambda **kwargs: calls.append(kwargs))
    writes, reads, cache = object(), object(), object()
    instance(
        object(),
        rope_mats=(object(), object()),
        page_table=writes,
        kv_cache=cache,
        packed={"packed_p": 6, "position_idx": object(), "attn_mask": object(), "read_page_table": reads},
    )
    assert calls[0]["page_table"] is writes and calls[0]["read_page_table"] is reads


def test_partial_second_width_failure_releases_owned_read_buffer_before_mesh_close(
    production, generator_module, monkeypatch, expect_error
):
    decoder = decoder_for(production, monkeypatch)
    native = production.tt.from_torch

    def fail_second_read(data, **kwargs):
        if kwargs.get("device") is not None and 4096 in decoder._pv_widths:
            raise RuntimeError("second read allocation failed")
        return native(data, **kwargs)

    monkeypatch.setattr(production.tt, "from_torch", fail_second_read)
    with expect_error(RuntimeError, match="second read allocation failed"):
        decoder.prepare_widths([3072, 4096])
    first = decoder._pv_widths[3072]["read_cache_by_type"]["sliding_attention"]
    assert not first.freed and decoder._pv_widths[4096]["read_cache_by_type"] == {}
    assert not any(event[0] in ("body", "begin") for event in production.runtime.events)
    model = generator_module.Gemma4DFlashContractForCausalLM.__new__(generator_module.Gemma4DFlashContractForCausalLM)
    model._spec_decoder, model._spec_width_set = decoder, True
    model.mesh_device, model.model = production.runtime, [decoder.target]
    model._spec_release_decoder(teardown=True)
    assert model._spec_decoder is None and first.freed
    assert all(buffer.freed for record in decoder._pv_widths.values() for buffer in record["cache_by_type"].values())
    model._spec_release_decoder(teardown=True)
