# SPDX-License-Identifier: Apache-2.0
"""Host ownership/boundary metadata checks; no devices are opened."""
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_8_27b.tt.generator_vllm import Qwen38ForCausalLM


class PoisonPositions:
    def __getitem__(self, key):
        raise AssertionError('Resident snapshot tracking consulted stale host positions')


def adapter_with_partial_owner():
    adapter = Qwen38ForCausalLM.__new__(Qwen38ForCausalLM)
    adapter.batch_size = 2
    adapter.scheduler_block_size = 4096
    adapter.slot_ends = [4094, 128]
    adapter.request_seeds = [1234, 5678]
    adapter.generator = SimpleNamespace(_decode_active=torch.tensor([True, False]))
    adapter.saved = []
    adapter.state_pool = SimpleNamespace(
        pools={0: object(), 21: object(), 42: object()},
        save=lambda slot, ids: adapter.saved.append((slot, ids.copy())),
    )
    tables = [torch.zeros(2, 64, dtype=torch.int32) for _ in range(64)]
    tables[0][0, 0] = 11
    tables[21][0, 0] = 29
    tables[42][0, 0] = 47
    return adapter, tables


def test_dormant_partial_prefill_owner_survives_authoritative_and_resident_decode():
    adapter, tables = adapter_with_partial_owner()
    adapter._save_decode_boundary(torch.tensor([4094, -1]), True, tables)
    assert adapter.slot_ends == [4095, 128]
    assert adapter.saved == []

    adapter._save_decode_boundary(PoisonPositions(), False, tables)
    assert adapter.slot_ends == [4096, 128]
    assert adapter.saved == [(0, {0: 11, 21: 29, 42: 47})]

    adapter._save_decode_boundary(PoisonPositions(), False, tables)
    assert adapter.slot_ends == [4097, 128]
    assert len(adapter.saved) == 1
    # A resumes from its retained state at128, without an impossible restore
    # from a non-aligned prefix checkpoint.
    assert adapter.slot_ends[1] == 128


def test_snapshot_tracking_uses_authoritative_active_mask_not_position_values():
    adapter, tables = adapter_with_partial_owner()
    # Deliberately plausible but stale/nonnegative scheduler position for a
    # dormant owner must not cause snapshot writes or ownership advancement.
    adapter._save_decode_boundary(torch.tensor([4095, 4095]), True, tables)
    assert adapter.slot_ends == [4096, 128]
    assert adapter.saved == [(0, {0: 11, 21: 29, 42: 47})]


def test_release_clears_only_the_released_owner():
    adapter, _ = adapter_with_partial_owner()
    adapter.decode_slots = {0}
    adapter._event = lambda *args, **kwargs: None
    adapter.release_request(0)
    assert adapter.slot_ends == [None, 128]
    assert adapter.decode_slots == set()
    assert adapter.request_seeds[1] == 5678
    assert 0 <= adapter.request_seeds[0] < 2**31
