# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks for reconstructing drafter taps without rewriting serving KV.

The production allocator and reconstruction context run against explicit tensor
handles. These tests check ownership and ordering; TT execution needs a device
witness with the same source revision.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

# Pytest discovers these imported fixtures by name.
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import adapter  # noqa: F401
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import decoder_width_for  # noqa: F401
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import expect_error  # noqa: F401
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import model  # noqa: F401
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import (
    DeviceResult,
    ScratchTensor,
    _complete,
    _ordinary,
    _prefill,
    _serving_config,
    _tensor,
)
from models.demos.gemma4.tt.dflash_contract import ContractRequest, ContractStep


def _step(model, position, slot=7):
    owner = ContractRequest(55, slot, list(range(position + 1)))
    table = _tensor([[55, 56, 57, 58]])
    return owner, ContractStep((owner,), table, [table.clone(), table.clone()], model.kv_cache)


def test_storage_uses_configured_capacity_null_page_and_both_batches(model):
    storage = model._ct_rebuild_storage
    assert storage["capacity"] == 1024
    assert set(storage["by_batch"]) == {1, 32}
    assert storage["kv_cache"][0][0][0].shape == (17, 1, 64, 8)
    assert storage["host_tables"][0].tolist() == [list(range(1, 17))]
    assert storage["by_batch"][32][0].shape == (32, 16)
    assert all(buffer.releases == 0 for buffer in storage["allocated"])


def test_allocator_supports_device_shapes_without_slice_indexing(model, expect_error):
    serving = model.kv_cache[0][0][0]
    with expect_error(TypeError, match="only supports integer indices"):
        serving.shape[1:]
    model._contract_release_rebuild_storage()
    model._contract_prepare_rebuild_storage(model.kv_cache)
    scratch = model._ct_rebuild_storage["kv_cache"][0][0][0]
    assert scratch.shape == (17, 1, 64, 8)
    assert scratch.padded_shape[1] == serving.padded_shape[1]
    assert scratch.padded_shape[-1] == serving.padded_shape[-1]


def test_non_power_of_two_context_and_effective_block_sizes(model):
    model._contract_release_rebuild_storage()
    model.model_args[0].max_seq_len = 1500
    model.model[0].layers[1].self_attn.config.head_dim = 16
    model._contract_prepare_rebuild_storage(model.kv_cache)
    storage = model._ct_rebuild_storage
    assert storage["capacity"] == 2048
    assert storage["kv_cache"][0][0][0].shape[0] == 33
    assert storage["kv_cache"][0][1][0].shape[0] == 65
    assert storage["host_tables"][1].shape == (1, 64)


@pytest.mark.parametrize("entry", ["allocate_kv_cache", "allocate_kv_cache_per_layer"])
def test_both_allocation_entries_prepare_before_return(model, adapter, monkeypatch, entry):
    model._contract_release_rebuild_storage()
    monkeypatch.setattr(adapter.Gemma4ForCausalLM, entry, lambda self, *a, **k: self.kv_cache)
    assert getattr(model, entry)([]) is model.kv_cache
    assert model._ct_rebuild_storage is not None
    assert not any(event[0] == "replay" for event in model.events)


def test_reconstruction_preserves_kv_and_bootstraps_original_storage(model, monkeypatch):
    target = model.model[0]
    serving = model.kv_cache[0][0][0]
    serving.value.fill_(73)
    allocated = list(model._ct_rebuild_storage["allocated"])
    original = model._contract_target_prefill
    real_bootstrap = model._spec_bootstrap
    bootstraps = []

    def prefill(**kwargs):
        assert kwargs["kv_cache"] is model._ct_rebuild_storage["kv_cache"]
        assert kwargs["empty_slots"] == [0]
        for layer in kwargs["kv_cache"][0]:
            for tensor in layer:
                tensor.value.fill_(-9)
        return original(**kwargs)

    def bootstrap(anchor, position, tables, cache, **kwargs):
        assert cache is model.kv_cache
        assert model.mode == "DECODE"
        bootstraps.append((position, tables.clone(), kwargs["page_tables_per_layer"]))
        real_bootstrap(anchor, position, tables, cache, **kwargs)

    monkeypatch.setattr(model, "_contract_target_prefill", prefill)
    monkeypatch.setattr(model, "_spec_bootstrap", bootstrap)
    for position in (158, 193):
        owner, step = _step(model, position)
        model._contract_rebuild(owner, step, 0, position, position)
        assert model._spec_owner_slot == 7
        assert model._ct_force_reload is True
        assert torch.all(serving.value == 73)
        assert model._ct_rebuild_storage["allocated"] == allocated
        assert target.tap_layers is None
    assert [item[0] for item in bootstraps] == [158, 193]
    assert all(item[1].tolist() == [[55, 56, 57, 58]] for item in bootstraps)
    assert all(buffer.releases == 0 for buffer in allocated)


@pytest.mark.parametrize("fail", [False, True])
def test_context_preserves_device_tables_tails_validity_and_absent_attributes(model, monkeypatch, expect_error, fail):
    target = model.model[0]
    attention = target.layers[0].self_attn
    config = attention.config
    serving_pt = ScratchTensor(_tensor([[91, 92]]))
    original_by_batch = {1: [serving_pt]}
    original_last = {1: [_tensor([[91, 92]])]}
    target._persistent_pt_by_batch = original_by_batch
    target._last_host_pt_by_batch = original_last
    target._active_page_tables_per_layer = original_last[1]
    target.bounded_sliding_kv_cache = model._bounded_sliding_kv_cache = True
    target.prefill_valid_len_dev = ScratchTensor(_tensor([193]))
    config.prefill_valid_len_dev = target.prefill_valid_len_dev
    config.cache_position_modulo = 1024
    config.sliding_window = 1024
    live_tail = ScratchTensor(_tensor([42]))
    attention._tail_pool = [[live_tail, live_tail]]
    attention._tail_pool_map = {55: 0}
    attention._sliding_tails_by_key = {55: [live_tail, live_tail]}
    config.sliding_prefill_tail_persistent = [live_tail, live_tail]
    config._g4_active_req_key = 55
    attention._last_kv = [live_tail, live_tail]
    markers = {3, 7}
    model._slots_prefilled_since_decode = markers
    objects = [model, target, attention, config]
    before = [dict(vars(obj)) for obj in objects]
    temporary = ScratchTensor(_tensor([99]))

    def release(**kwargs):
        assert kwargs == {"clear_persistent": True, "all_keys": True}
        assert attention._tail_pool is None
        assert attention._sliding_tails_by_key == {2: [temporary]}
        temporary.deallocate(True)

    monkeypatch.setattr(attention, "_release_sliding_prefill_tail", release)
    before[2]["_release_sliding_prefill_tail"] = release

    def exercise():
        with model._contract_rebuild_context(193) as storage:
            assert target._persistent_pt_by_batch is storage["by_batch"]
            assert target._last_host_pt_by_batch == {}
            assert model._ct_eager_prefill is True
            assert not model._bounded_sliding_kv_cache and not target.bounded_sliding_kv_cache
            assert target.prefill_valid_len_dev is config.prefill_valid_len_dev is None
            assert config.cache_position_modulo is None
            assert attention._last_kv is None
            assert attention._tail_pool is None
            attention._sliding_tails_by_key[2] = [temporary]
            config._g4_active_req_key = 2
            target._last_host_pt_by_batch[1] = [_tensor([[1]])]
            model.mode = "PREFILL"
            model._slots_prefilled_since_decode.add(0)
            if fail:
                raise ValueError("prefill failure")

    if fail:
        with expect_error(ValueError, match="prefill failure"):
            exercise()
    else:
        exercise()
    for obj, saved in zip(objects, before):
        assert vars(obj).keys() == saved.keys()
        assert all(getattr(obj, name) is value for name, value in saved.items())
    assert model._slots_prefilled_since_decode is markers and markers == {3, 7}
    assert serving_pt.value.tolist() == [[91, 92]]
    assert live_tail.releases == 0 and live_tail.value.tolist() == [42]
    assert temporary.releases == 1


@pytest.mark.parametrize("shape", [(2, 16), (1, 17), (32, 17)])
def test_context_rejects_page_table_growth_before_conversion(model, shape, expect_error):
    with model._contract_rebuild_context(158):
        with expect_error(RuntimeError, match="not prepared"):
            model.model[0]._page_tables_to_ttnn([torch.zeros(shape, dtype=torch.int32)])


@pytest.mark.parametrize("field", ["_deferred_bounded_fill", "_deferred_bounded_fill_batched"])
def test_context_rejects_pending_bounded_fill_without_mutation(model, field, expect_error):
    setattr(model.model[0].layers[1].self_attn.config, field, object())
    original = dict(vars(model))
    with expect_error(RuntimeError, match="pending bounded fills"):
        with model._contract_rebuild_context(158):
            pytest.fail("pending fill reached reconstruction")
    assert all(getattr(model, name) is value for name, value in original.items())


def test_capacity_decline_keeps_ordinary_completion(model):
    _prefill(model)
    owner = model._ct_requests[10]
    owner.tokens = list(range(1025))
    model.model_args[0].max_seq_len = 2048
    _ordinary(model, [1025], [1025], [10], DeviceResult(_tensor([[42]])))
    model.events.clear()
    output = _complete(model, [[42]], [[1026]])
    assert output.num_valid.tolist() == [0]
    assert owner.tokens[-1] == 42
    assert not any(event[0] in ("sync", "prefill", "commit", "replay") for event in model.events)


def test_partial_allocation_and_repeated_release(model, monkeypatch, expect_error):
    model._contract_release_rebuild_storage()
    allocated = []

    def allocate(value, **kwargs):
        if len(allocated) == 2:
            raise ValueError("allocation failed")
        tensor = ScratchTensor(value)
        allocated.append(tensor)
        return tensor

    monkeypatch.setattr(sys.modules["ttnn"], "from_torch", allocate)
    monkeypatch.setattr(model.model[0], "_page_table_torch_to_ttnn", allocate)
    with expect_error(ValueError, match="allocation failed"):
        model._contract_prepare_rebuild_storage(model.kv_cache)
    assert model._ct_rebuild_storage is None
    assert [tensor.releases for tensor in allocated] == [1, 1]
    model._contract_release_rebuild_storage()
    assert [tensor.releases for tensor in allocated] == [1, 1]


def test_final_cleanup_releases_scratch_after_serving_capture(model, adapter, monkeypatch):
    storage = model._ct_rebuild_storage
    calls = []
    monkeypatch.setattr(
        adapter.Gemma4DFlashForCausalLM, "release_persistent_capture", lambda self: calls.append("serving")
    )
    model.release_persistent_capture()
    model.release_persistent_capture()
    assert calls == ["serving", "serving"]
    assert model._ct_rebuild_storage is None
    assert all(tensor.releases == 1 for tensor in storage["allocated"])
    assert model.kv_cache[0][0][0].releases == 0


def test_spec_plan_accounts_configured_31b_scratch_per_chip(adapter, monkeypatch):
    monkeypatch.setenv("GEMMA4_DFLASH_DRAFTER", "/unused")
    monkeypatch.setattr(
        adapter,
        "_dflash_drafter_config",
        lambda path: dict(num_hidden_layers=5, hidden_size=128, head_dim=32, num_key_value_heads=8, block_size=16),
    )
    monkeypatch.setattr(adapter, "_dflash_mesh_tp", lambda: 8)
    config = _serving_config()
    config.model_config.hf_config = SimpleNamespace(
        layer_types=["sliding_attention"] * 50 + ["full_attention"] * 10,
        num_key_value_heads=16,
        head_dim=256,
        num_global_key_value_heads=4,
        global_head_dim=512,
    )
    base = adapter.Gemma4DFlashForCausalLM.spec_plan(config, 1, 5)
    plan = adapter.Gemma4DFlashContractForCausalLM.spec_plan(config, 32, 5)
    assert plan.extra_bytes_per_seq - base.extra_bytes_per_seq == 511180800 + 506880
    assert plan.extra_bytes_per_token == 0


def test_shared_layers_keep_single_owned_allocation(model):
    model._contract_release_rebuild_storage()
    model.model[0].kv_shared_layer_map = {1: 0}
    model._contract_prepare_rebuild_storage(model.kv_cache)
    storage = model._ct_rebuild_storage
    assert storage["kv_cache"][0][1] is storage["kv_cache"][0][0]
    assert len(storage["allocated"]) == 6
    model._contract_release_rebuild_storage()
    assert all(tensor.releases == 1 for tensor in storage["allocated"])


def test_cleanup_attempts_every_buffer_when_one_release_fails(model, monkeypatch, expect_error):
    storage = model._ct_rebuild_storage
    failed = storage["allocated"][2]

    def release(force):
        failed.releases += 1
        raise RuntimeError("release failed")

    monkeypatch.setattr(failed, "deallocate", release)
    with expect_error(RuntimeError, match="release failed"):
        model._contract_release_rebuild_storage()
    assert all(tensor.releases == 1 for tensor in storage["allocated"])
    model._contract_release_rebuild_storage()
    assert all(tensor.releases == 1 for tensor in storage["allocated"])


def test_failed_prefill_releases_captured_taps_and_restores_stashes(model, monkeypatch, expect_error):
    owner, step = _step(model, 158)
    target = model.model[0]
    original_ids = _tensor([[71]])
    target._prefill_input_ids_torch = original_ids
    target._prefill_embeds_torch = object()
    original_embeds = target._prefill_embeds_torch
    markers = model._slots_prefilled_since_decode
    model._perf_decode_tokens, model._perf_decode_s = 47, 2.5
    tap = ScratchTensor(_tensor([9]))
    serving_pt = {1: [ScratchTensor(_tensor([[7]]))]}
    target._persistent_pt_by_batch = serving_pt

    def fail(**kwargs):
        target.taps.append(tap)
        target._prefill_input_ids_torch = _tensor([[2]])
        target._prefill_embeds_torch = None
        target._prefill_batch_size = 1
        target._prefill_seq_len_per_user = 158
        model._perf_decode_tokens, model._perf_decode_s = 0, 0.0
        model.mode = "PREFILL"
        model._slots_prefilled_since_decode.add(0)
        raise RuntimeError("target prefill failed")

    monkeypatch.setattr(model, "_contract_target_prefill", fail)
    with expect_error(RuntimeError, match="target prefill failed"):
        model._contract_rebuild(owner, step, 0, 158, 158)
    assert tap.releases == 1
    assert target.tap_layers is None
    assert target._prefill_input_ids_torch is original_ids
    assert target._prefill_embeds_torch is original_embeds
    assert not hasattr(target, "_prefill_batch_size")
    assert not hasattr(target, "_prefill_seq_len_per_user")
    assert target._persistent_pt_by_batch is serving_pt
    assert model._slots_prefilled_since_decode is markers
    assert model.mode == "DECODE"
    assert (model._perf_decode_tokens, model._perf_decode_s) == (47, 2.5)
    assert model._spec_pending is None and model._ct_decoder_owner is None
    assert not any(event[0] in ("ingest", "reseed", "replay") for event in model.events)


def test_direct_rebuild_capacity_check_precedes_device_work(model, expect_error):
    owner, step = _step(model, 1025)
    model.events.clear()
    with expect_error(RuntimeError, match="scratch capacity"):
        model._contract_rebuild(owner, step, 0, 1025, 1025)
    assert model.events == []


@pytest.mark.parametrize("fail", [False, True])
def test_real_page_table_updates_keep_scratch_addresses_and_restore_inherited_converter(
    model, monkeypatch, expect_error, fail
):
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "tt/model.py"
    tree = ast.parse(source.read_text())
    names = {
        "_host_page_tables_batch",
        "_page_tables_to_ttnn",
        "_pad_page_table_host_to_shape",
        "update_persistent_per_layer_page_tables",
    }
    methods = [
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(methods) == len(names)
    namespace = {"torch": torch, "ttnn": sys.modules["ttnn"]}
    # Execute the four production methods without importing TT model weights.
    selected = ast.ClassDef(name="PageTables", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    target = model.model[0]
    monkeypatch.setattr(target, "__class__", type("TargetWithTables", (type(target), namespace["PageTables"]), {}))
    del target._page_tables_to_ttnn
    runtime = sys.modules["ttnn"]
    monkeypatch.setattr(runtime, "Tensor", ScratchTensor, raising=False)
    monkeypatch.setattr(runtime, "int32", torch.int32, raising=False)
    monkeypatch.setattr(runtime, "ROW_MAJOR_LAYOUT", object(), raising=False)
    monkeypatch.setattr(
        runtime, "copy_host_to_device_tensor", lambda host, device: device.value.copy_(host.value), raising=False
    )
    target._replicate_to_mesh_mapper = lambda: None
    original = {1: [ScratchTensor(_tensor([[71, 72]])) for _ in target.layers]}
    target._persistent_pt_by_batch = original
    target._last_host_pt_by_batch = {1: [_tensor([[71, 72]]) for _ in target.layers]}
    saved_shadow = target._last_host_pt_by_batch
    scratch_ids = {
        batch: tuple(id(t) for t in tensors) for batch, tensors in model._ct_rebuild_storage["by_batch"].items()
    }

    def allocate_forbidden(*args, **kwargs):
        pytest.fail("persistent scratch page table allocated during reconstruction")

    monkeypatch.setattr(target, "_page_table_torch_to_ttnn", allocate_forbidden)

    def update():
        with model._contract_rebuild_context(158) as storage:
            target.update_persistent_per_layer_page_tables(storage["host_tables"])
            wide = [
                torch.cat((table, torch.zeros(31, table.shape[1], dtype=table.dtype)))
                for table in storage["host_tables"]
            ]
            target.update_persistent_per_layer_page_tables(wide)
            for batch, tensors in storage["by_batch"].items():
                assert tuple(id(t) for t in tensors) == scratch_ids[batch]
                assert tensors[0].value[0].tolist() == list(range(1, 17))
            if fail:
                raise ValueError("after page upload")

    if fail:
        with expect_error(ValueError, match="after page upload"):
            update()
    else:
        update()
    assert "_page_tables_to_ttnn" not in vars(target)
    assert target._persistent_pt_by_batch is original
    assert target._last_host_pt_by_batch is saved_shadow
    assert all(t.value.tolist() == [[71, 72]] for t in original[1])


@pytest.mark.parametrize("verified_first", [False, True])
def test_bounded_pending_capture_declines_without_touching_kv_and_ordinary_progresses(
    model, monkeypatch, verified_first
):
    from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import _start_solo, _verify

    if verified_first:
        _start_solo(model)
        completion = _verify(model, blocks=[[4, 21, 22, 23, 24, 25]], positions=[3])
        hidden = completion.hidden
        position = 4
    else:
        _prefill(model)
        _ordinary(model, [3], [2], [10], DeviceResult(_tensor([[4]])))
        hidden = None
        position = 3
    decoder = model._spec_decoder
    decoder._pv_widths = {1024: {"trace": None}}
    decoder._prepared_widths = (1024,)
    config = model.model[0].layers[1].self_attn.config
    config.cache_position_modulo = 128
    config.sliding_window = 128
    serving = model.kv_cache[0][0][0]
    serving.value.fill_(73)

    def corrupting_capture(widths):
        serving.value.fill_(-1)
        model.events.append(("capture_widths", widths))

    monkeypatch.setattr(decoder, "capture_widths", corrupting_capture)
    model.events.clear()
    output = _complete(model, [[42]], [[position]], hidden=hidden)
    assert output.num_valid.tolist() == [0]
    assert model._ct_requests[10].tokens[-1] == 42
    assert torch.all(serving.value == 73)
    assert not any(event[0] in ("sync", "capture_widths", "prefill", "commit", "replay") for event in model.events)
    _ordinary(model, [42], [position], [10], DeviceResult(_tensor([[43]])))
    output = _complete(model, [[43]], [[position + 1]])
    assert output.num_valid.tolist() == [0]
    assert model._ct_requests[10].tokens[-2:] == [42, 43]
    assert torch.all(serving.value == 73)
    assert any(event[0] == "decode" for event in model.events)
    assert not any(event[0] in ("capture_widths", "prefill", "commit", "replay") for event in model.events)


def test_direct_bounded_rebuild_rejects_pending_capture_before_device_work(model, expect_error):
    owner, step = _step(model, 158)
    model._spec_decoder._pv_widths = {1024: {"trace": None}}
    model._spec_decoder._prepared_widths = (1024,)
    model.model[0].layers[1].self_attn.config.cache_position_modulo = 1024
    model.events.clear()
    with expect_error(RuntimeError, match="capture before serving requests"):
        model._contract_rebuild(owner, step, 0, 158, 158)
    assert model.events == []
