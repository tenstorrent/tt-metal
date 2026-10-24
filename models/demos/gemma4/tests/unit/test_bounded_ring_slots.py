# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bounded sliding rings: slot recycling must never evict a request that a prefill
bound since the last authoritative decode step."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM


def _rows(first_blocks):
    # one page-table row per request, keyed by its first global block id
    return torch.tensor([[b, b + 1, b + 2] for b in first_blocks], dtype=torch.int32)


def _generator():
    # Only the slot bookkeeping is exercised: a bare namespace with the row-key helper.
    return SimpleNamespace(_bounded_row_key=Gemma4ForCausalLM._bounded_row_key)


def _slots(self, first_blocks, authoritative):
    return Gemma4ForCausalLM._bounded_ring_slots(self, _rows(first_blocks), 32, authoritative)


def test_full_map_evicts_the_finished_request_not_a_fresh_arrival():
    self = _generator()
    # A long request decodes alone: the last authoritative running set is {long}.
    (long_slot,) = _slots(self, [5000], authoritative=True)
    # It finishes; 32 new requests arrive through prefill (non-authoritative), one per call.
    new_slots = [_slots(self, [100 + i], authoritative=False)[0] for i in range(32)]
    assert len(set(new_slots)) == 32, "two live requests were handed the same ring"
    assert long_slot in new_slots, "the departed request's ring was not the one recycled"
    assert 5001 not in self._bounded_ring_slot_map


def test_stale_keys_from_the_last_running_set_are_recycled_oldest_first():
    self = _generator()
    old = [_slots(self, [b], authoritative=False)[0] for b in range(10, 42)]
    _slots(self, list(range(10, 42)), authoritative=True)  # all 32 ran together
    # They all finish; a new burst must recycle them without colliding.
    new = [_slots(self, [900 + i], authoritative=False)[0] for i in range(32)]
    assert sorted(new) == sorted(old)
    assert len(set(new)) == 32


def test_authoritative_step_resets_the_fresh_set():
    self = _generator()
    a = _slots(self, [1], authoritative=False)[0]
    _slots(self, [1], authoritative=True)
    assert self._g4_ring_keys_since_authoritative == set()
    b = _slots(self, [2], authoritative=False)[0]
    assert a != b
