# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the production contract's bounded ring ownership on the host."""

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
    _tensor,
)


@pytest.fixture
def bounded(model, adapter, monkeypatch):
    model._bounded_sliding_kv_cache = True
    model.model_args[0].max_batch_size = 3
    target = model.model[0]
    target.hf_config = SimpleNamespace(sliding_window=1024, layer_types=["sliding_attention", "full_attention"])
    target.layers[0].self_attn.config.cache_position_modulo = 2048
    target.layers[0].self_attn.config.sliding_window = 1024
    for layer in target.layers:
        layer.self_attn.mesh_config = target.mesh_config
    cache = ScratchTensor(torch.zeros(96, 1, 64, 8, dtype=torch.bfloat16))
    model.kv_cache = [[(cache, cache), (cache, cache)]]
    monkeypatch.setattr(
        model,
        "_pad_sliding_page_tables_for_bounded",
        adapter.Gemma4ForCausalLM._pad_sliding_page_tables_for_bounded.__get__(model),
    )
    original = model._contract_target_prefill

    def prefill(*args, page_tables_per_layer=None, **kwargs):
        tables = model._build_per_layer_page_tables(page_tables_per_layer, kwargs.get("page_table"))
        remapped = model._pad_sliding_page_tables_for_bounded(tables, kwargs.get("kv_cache"))
        model.events.append(("ring_prefill", remapped, tuple(model._ct_requests)))
        return original(*args, page_tables_per_layer=page_tables_per_layer, **kwargs)

    monkeypatch.setattr(model, "_contract_target_prefill", prefill)
    return model


def _ring_blocks(model, key):
    table = _tensor([[key, key + 1]])
    return model._pad_sliding_page_tables_for_bounded([table, table], model.kv_cache, authoritative=True)[0]


def test_released_peer_ring_is_reused_while_off_batch_owner_survives(bounded):
    _prefill(bounded, keys=(101, 201, 301))
    a, b, c = [bounded._ct_requests[key] for key in (101, 201, 301)]
    original_a = _ring_blocks(bounded, 101).clone()
    bounded.release_request(2)
    step = bounded._contract_step(_tensor([[3]]), _tensor([[2]]), _tensor([[201, 202]]), None, bounded.kv_cache)
    bounded._contract_refresh(step, 0)
    _prefill(bounded, keys=(401,), slots=[2])
    assert bounded._bounded_ring_slot_map == {101: 0, 201: 1, 401: 2}
    assert torch.equal(_ring_blocks(bounded, 101), original_a)
    assert a.live and b.live and not c.live
    assert _ring_blocks(bounded, 401).tolist() == [list(range(64, 96))]


def test_arriving_batch_reserves_distinct_rings_before_owner_registration(bounded):
    _prefill(bounded, keys=(101, 201, 301))
    ring_prefill = next(event for event in bounded.events if event[0] == "ring_prefill")
    assert ring_prefill[2] == ()
    assert ring_prefill[1][0].tolist() == [list(range(index * 32, (index + 1) * 32)) for index in range(3)]
    assert {key: owner.bounded_ring_key for key, owner in bounded._ct_requests.items()} == {
        101: 101,
        201: 201,
        301: 301,
    }


def test_admission_exhaustion_cannot_evict_or_partially_assign(bounded, expect_error):
    _prefill(bounded, keys=(101, 201))
    before = bounded._bounded_ring_slot_map.copy()
    with expect_error(RuntimeError, match="bounded rings are full"):
        bounded._bounded_ring_slots(_tensor([[301, 302], [401, 402]]), 3, False)
    assert bounded._bounded_ring_slot_map == before
    _prefill(bounded, keys=(301,), slots=[2])
    bounded._bounded_ring_slots(_tensor([[201, 202]]), 3, True)
    before = bounded._bounded_ring_slot_map.copy()
    with expect_error(RuntimeError, match="bounded rings are full"):
        _prefill(bounded, keys=(401,), slots=[2])
    assert bounded._bounded_ring_slot_map == before
    assert set(bounded._ct_requests) == {101, 201, 301}


def test_completed_requests_do_not_accumulate_beyond_physical_pool(bounded):
    bounded.model_args[0].max_batch_size = 32
    for index in range(20):
        key = 101 + index * 100
        _prefill(bounded, keys=(key,))
        assert bounded._bounded_ring_slot_map == {key: 0}
        bounded.release_request(0)
        assert bounded._bounded_ring_slot_map == {}
    assert bounded._ct_requests == {}


def test_release_follows_state_slot_permutation_not_ring_slot(bounded):
    _prefill(bounded, keys=(101, 201, 301))
    bounded.note_state_slots_moved({0: 2, 1: 0, 2: 1})
    bounded.release_request(0)
    assert bounded._bounded_ring_slot_map == {101: 0, 301: 2}
    assert set(bounded._ct_requests) == {101, 301}
    assert bounded._ct_requests[101].slot == 2


def test_release_all_is_idempotent_and_drops_every_owned_ring(bounded):
    _prefill(bounded, keys=(101, 201, 301))
    owners = list(bounded._ct_requests.values())
    bounded.release_request(None)
    bounded.release_request(None)
    bounded.release_request(0)
    assert bounded._ct_requests == {}
    assert bounded._bounded_ring_slot_map == {}
    assert all(not owner.live for owner in owners)
    assert not any(event[0] == "sync" for event in bounded.events)


def test_release_uses_full_attention_key_not_legacy_identity(bounded):
    legacy = _tensor([[101, 102], [201, 202]])
    full = _tensor([[501, 502], [601, 602]])
    bounded.prefill_forward(
        tokens=_tensor([[1, 2], [5, 6]]),
        prompt_lens=[2, 2],
        empty_slots=[2, 0],
        page_table=legacy,
        page_tables_per_layer=[legacy, full],
        kv_cache=bounded.kv_cache,
    )
    assert bounded._ct_requests[101].bounded_ring_key == 501
    assert bounded._bounded_ring_slot_map == {501: 0, 601: 1}
    bounded.release_request(2)
    assert bounded._bounded_ring_slot_map == {601: 1}
    assert set(bounded._ct_requests) == {201}


@pytest.mark.parametrize("warmup_method", ["warmup_model_prefill", "warmup_model_decode"])
def test_warmup_ring_reservations_do_not_reach_first_request(bounded, adapter, monkeypatch, warmup_method):
    def warmup(instance):
        instance._bounded_ring_slots(_tensor([[901, 902], [911, 912], [921, 922]]), 3, False)

    monkeypatch.setattr(adapter.Gemma4DFlashForCausalLM, warmup_method, warmup, raising=False)
    getattr(bounded, warmup_method)()
    assert not hasattr(bounded, "_bounded_ring_slot_map")
    assert bounded._ct_requests == {}
    _prefill(bounded, keys=(101,))
    assert bounded._bounded_ring_slot_map == {101: 0}


def test_direct_warmup_prefill_preserves_existing_ring_assignments(bounded):
    _prefill(bounded, keys=(101,))
    bounded.prefill_forward(
        tokens=_tensor([[0, 0]]),
        prompt_lens=[2],
        page_table=_tensor([[901, 902]]),
        kv_cache=bounded.kv_cache,
        warmup_prefill=True,
    )
    assert bounded._bounded_ring_slot_map == {101: 0}
    assert set(bounded._ct_requests) == {101}


def test_warmup_exception_restores_ring_assignments(bounded, adapter, monkeypatch, expect_error):
    _prefill(bounded, keys=(101,))

    def warmup(instance):
        instance._bounded_ring_slots(_tensor([[901, 902]]), 3, False)
        raise ValueError("warmup failed")

    monkeypatch.setattr(adapter.Gemma4DFlashForCausalLM, "warmup_model_decode", warmup, raising=False)
    with expect_error(ValueError, match="warmup failed"):
        bounded.warmup_model_decode()
    assert bounded._bounded_ring_slot_map == {101: 0}
    assert bounded._ct_warmup_depth == 0


def test_failed_prefill_synchronizes_before_rolling_back_new_ring(bounded, monkeypatch, expect_error):
    _prefill(bounded, keys=(101,))
    original = bounded._contract_target_prefill

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise ValueError("prefill failed")

    monkeypatch.setattr(bounded, "_contract_target_prefill", fail)
    with expect_error(ValueError, match="prefill failed"):
        _prefill(bounded, keys=(201,), slots=[1])
    assert bounded.events[-1] == ("sync",)
    assert bounded._bounded_ring_slot_map == {101: 0}
    assert set(bounded._ct_requests) == {101}


def test_failed_prefill_keeps_reserved_ring_when_synchronization_fails(bounded, monkeypatch, expect_error):
    original = bounded._contract_target_prefill

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise ValueError("prefill failed")

    def sync():
        raise RuntimeError("sync failed")

    monkeypatch.setattr(bounded, "_contract_target_prefill", fail)
    monkeypatch.setattr(bounded, "_contract_synchronize", sync)
    with expect_error(RuntimeError, match="sync failed"):
        _prefill(bounded, keys=(101,))
    assert bounded._bounded_ring_slot_map == {101: 0}
    assert bounded._ct_requests == {}


def test_release_waits_for_pending_generation_before_page_id_reuse(bounded):
    _prefill(bounded, keys=(101,))
    pending = _ordinary(bounded, [3], [2], [101], DeviceResult(_tensor([[4]])))
    old = pending.step.owners[0]
    bounded.events.clear()
    bounded.release_request(0)
    assert bounded.events == [("sync",)]
    assert not old.live and bounded._bounded_ring_slot_map == {}
    _prefill(bounded, keys=(101,), prompts=[[50, 51]])
    new = bounded._ct_requests[101]
    assert new is not old
    stale = _complete(bounded, [[4]], [[3]])
    assert stale.num_valid.tolist() == [0]
    assert new.tokens == [50, 51]
    assert bounded._bounded_ring_slot_map == {101: 0}


def test_pending_release_sync_failure_keeps_live_owner_and_ring(bounded, monkeypatch, expect_error):
    _prefill(bounded, keys=(101,))
    _ordinary(bounded, [3], [2], [101], DeviceResult(_tensor([[4]])))
    owner = bounded._ct_requests[101]

    def sync():
        raise RuntimeError("sync failed")

    monkeypatch.setattr(bounded, "_contract_synchronize", sync)
    with expect_error(RuntimeError, match="sync failed"):
        bounded.release_request(0)
    assert bounded._ct_requests[101] is owner and owner.live
    assert bounded._bounded_ring_slot_map == {101: 0}


def test_superseding_prefill_waits_for_pending_old_generation(bounded):
    _prefill(bounded, keys=(101,))
    pending = _ordinary(bounded, [3], [2], [101], DeviceResult(_tensor([[4]])))
    bounded.events.clear()
    _prefill(bounded, keys=(101,), prompts=[[50, 51]])
    names = [event[0] for event in bounded.events]
    assert names.index("sync") < names.index("ring_prefill")
    assert pending.step.owners[0].live is False
    assert bounded._bounded_ring_slot_map == {101: 0}
    _complete(bounded, [[4]], [[3]])
    assert bounded._ct_requests[101].tokens == [50, 51]


def test_outstanding_proposal_survives_peer_ring_reuse(bounded):
    _prefill(bounded, keys=(101,))
    _ordinary(bounded, [3], [2], [101], DeviceResult(_tensor([[4]])))
    assert _complete(bounded, [[4]], [[3]]).num_valid.tolist() == [5]
    proposal = bounded._ct_proposal
    _prefill(bounded, keys=(201,), slots=[1])
    bounded.release_request(1)
    _prefill(bounded, keys=(301,), slots=[1])
    assert bounded._ct_proposal is proposal
    assert proposal.owner.live
    assert bounded._bounded_ring_slot_map == {101: 0, 301: 1}
    bounded.release_request(0)
    assert bounded._ct_proposal is None
    assert bounded._ct_decoder_owner is None
    assert bounded._bounded_ring_slot_map == {301: 1}


@pytest.mark.parametrize("collision", ["incoming_rows", "live_peer", "changed_owner_key"])
def test_ambiguous_full_attention_ring_ownership_fails_before_native_prefill(bounded, expect_error, collision):
    _prefill(bounded, keys=(101,))
    legacy = _tensor([[201, 202], [301, 302]]) if collision == "incoming_rows" else _tensor([[201, 202]])
    full = _tensor([[501, 502], [501, 502]]) if collision == "incoming_rows" else _tensor([[101, 102]])
    if collision == "changed_owner_key":
        legacy, full = _tensor([[101, 102]]), _tensor([[501, 502]])
    before = len(bounded.events)
    with expect_error(ValueError, match="share a bounded ring|bounded ring ownership"):
        bounded.prefill_forward(
            tokens=_tensor([[1, 2]] * legacy.shape[0]),
            prompt_lens=[2] * legacy.shape[0],
            page_table=legacy,
            page_tables_per_layer=[legacy, full],
            kv_cache=bounded.kv_cache,
        )
    assert len(bounded.events) == before
    assert bounded._bounded_ring_slot_map == {101: 0}


def test_unbounded_pending_release_does_not_synchronize(model):
    _prefill(model)
    _ordinary(model, [3], [2], [10], DeviceResult(_tensor([[4]])))
    model.events.clear()
    model.release_request(0)
    assert model.events == []


def test_physical_capacity_failure_rolls_back_provisional_ring(bounded, expect_error):
    bounded.model_args[0].max_batch_size = 32
    _prefill(bounded, keys=(101, 201, 301))
    before = bounded._bounded_ring_slot_map.copy()
    with expect_error(ValueError, match="slot 3 needs 128 blocks.*96 blocks"):
        _prefill(bounded, keys=(401,), slots=[3])
    assert bounded._bounded_ring_slot_map == before
    assert set(bounded._ct_requests) == {101, 201, 301}
    assert bounded.events[-1] == ("sync",)
    bounded.release_request(1)
    _prefill(bounded, keys=(501,), slots=[1])
    assert bounded._bounded_ring_slot_map == {101: 0, 301: 2, 501: 1}


def test_superseding_prefill_sync_failure_prevents_native_writes(bounded, monkeypatch, expect_error):
    _prefill(bounded, keys=(101,))
    _ordinary(bounded, [3], [2], [101], DeviceResult(_tensor([[4]])))
    owner = bounded._ct_requests[101]
    before = len(bounded.events)

    def sync():
        raise RuntimeError("sync failed")

    monkeypatch.setattr(bounded, "_contract_synchronize", sync)
    with expect_error(RuntimeError, match="sync failed"):
        _prefill(bounded, keys=(101,), prompts=[[50, 51]])
    assert len(bounded.events) == before
    assert bounded._ct_requests[101] is owner and owner.live
    assert bounded._bounded_ring_slot_map == {101: 0}


def test_release_without_own_pending_work_does_not_wait_for_another_owner(bounded):
    _prefill(bounded, keys=(101, 201))
    _ordinary(bounded, [3], [2], [101], DeviceResult(_tensor([[4]])))
    bounded.events.clear()
    bounded.release_request(1)
    assert bounded.events == []
    assert bounded._bounded_ring_slot_map == {101: 0}


def test_full_nominal_slot_reproduction_preserves_unscheduled_owner(bounded):
    bounded.model_args[0].max_batch_size = 32
    cache = ScratchTensor(torch.zeros(1024, 1, 64, 8, dtype=torch.bfloat16))
    bounded.kv_cache = [[(cache, cache), (cache, cache)]]
    keys = tuple(101 + index * 100 for index in range(32))
    _prefill(bounded, keys=keys)
    for slot in range(2, 32):
        bounded.release_request(slot)
    assert _ring_blocks(bounded, 201).tolist() == [list(range(32, 64))]
    _prefill(bounded, keys=(9001,), slots=[2])
    assert bounded._bounded_ring_slot_map == {101: 0, 201: 1, 9001: 2}
    assert _ring_blocks(bounded, 101).tolist() == [list(range(32))]
