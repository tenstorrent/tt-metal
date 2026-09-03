# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The 24-of-93 map, and the two numberings it keeps apart. Host only.

Kimi-K3 puts a full-attention layer every fourth position and a KDA layer everywhere else, so a
rank's KV cache holds one slot per *MLA* layer, not one per layer. Two numberings therefore exist —
the global model layer index that names weights, and the rank-local KV slot that indexes the cache —
and the failure mode when they are confused is silent: `ttMLA._cache_batch_idx` computes
`cache_user_id * layer_num + cache_layer_idx` against a cache sized to this rank's MLA count, so a
global slot on a rank with `first_layer_idx > 0` is a plausible integer pointing at another user's
rows, or past the end.

`KimiK3Config.mla_kv_slot()` is the model-wide map and returns the global slot. The
`NotImplementedError` in `adapters/kimi_k3.py` names it as the thing to use when wiring the cache
up, which is right for a single-rank run and wrong for every other one. The test below pins that
difference rather than leaving it to be rediscovered.
"""

import pytest

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule

# The depth ladder the bring-up gates on. `expected_mla` is what each depth actually exercises:
# a single-layer run has NO full-attention layer at all, so its KV cache is empty.
DEPTHS = [(1, 0), (5, 1), (12, 3), (24, 6), (93, 24)]


@pytest.mark.parametrize("num_layers, expected_mla", DEPTHS, ids=[f"L{n}" for n, _ in DEPTHS])
def test_depth_ladder_mla_counts(num_layers, expected_mla):
    """How many KV slots each bring-up depth needs — zero is a legal answer."""
    schedule = KimiK3LayerSchedule.build(KimiK3Config, 0, num_layers)
    assert schedule.num_mla_layers == expected_mla
    assert sum(slot is not None for slot in schedule.kv_slot_of_local) == expected_mla


def test_slots_are_dense_and_ordered():
    """Slots number 0..n-1 in layer order, with no gaps — the cache has no holes to skip."""
    schedule = KimiK3LayerSchedule.build(KimiK3Config, 0, KimiK3Config.NUM_LAYERS)
    assigned = [slot for slot in schedule.kv_slot_of_local if slot is not None]
    assert assigned == list(range(schedule.num_mla_layers))


def test_slots_are_rank_local_not_global():
    """The whole point. A rank starting mid-model must renumber from zero.

    Layers 24..47 hold MLA layers 27, 31, 35, 39, 43, 47. Their GLOBAL slots are 6..11; their
    rank-local slots must be 0..5, because this rank allocated a six-slot cache.
    """
    schedule = KimiK3LayerSchedule.build(KimiK3Config, first_layer_idx=24, num_layers=24)
    assert schedule.num_mla_layers == 6

    local_slots = [
        (schedule.global_index(local), schedule.kv_slot(local))
        for local in range(schedule.num_layers)
        if schedule.local_is_mla(local)
    ]
    assert local_slots == [(27, 0), (31, 1), (35, 2), (39, 3), (43, 4), (47, 5)]

    # And the model-wide map disagrees with every one of them, by exactly this rank's offset.
    for global_idx, local_slot in local_slots:
        assert KimiK3Config.mla_kv_slot(global_idx) == local_slot + 6


def test_kda_layers_have_no_slot():
    schedule = KimiK3LayerSchedule.build(KimiK3Config, 0, 24)
    for local in range(schedule.num_layers):
        assert schedule.local_is_mla(local) is (schedule.global_index(local) in KimiK3Config.mla_layer_ids())
        if not schedule.local_is_mla(local):
            assert schedule.kv_slot(local) is None


@pytest.mark.parametrize("num_layers", [4, 12, 24, 93], ids=lambda n: f"L{n}")
def test_kv_only_last_layer_is_legal_at_depths_ending_on_mla(num_layers):
    """Full-attention layers sit at 0-based `4k+3`, so a depth divisible by 4 ends on one.

    That covers the 12- and 24-layer bring-up depths, and the full 93-layer model — whose last
    layer, 92, is full-attention because 92 and 93 are adjacent in the checkpoint's schedule and
    break the otherwise-uniform 3-KDA-to-1-MLA pattern.
    """
    KimiK3LayerSchedule.build(KimiK3Config, 0, num_layers).validate_kv_only_last_layer()


@pytest.mark.parametrize("num_layers", [1, 5], ids=lambda n: f"L{n}")
def test_kv_only_last_layer_is_rejected_at_depths_ending_on_kda(num_layers, expect_error):
    """The shallow bring-up depths end on a KDA layer, and must say so rather than build.

    1 ends on layer 0 and 5 ends on layer 4, both KDA. A kv_only block there computes a full
    recurrence, discards it, and writes no KV — pure cost with no output. `PREFILL_KV_ONLY_LAST_LAYER`
    defaults to on in serving, so a shallow K3 run has to turn it off explicitly rather than
    inherit it.
    """
    with expect_error(ValueError, "is KDA"):
        KimiK3LayerSchedule.build(KimiK3Config, 0, num_layers).validate_kv_only_last_layer()


def test_slice_bounds_are_checked(expect_error):
    with expect_error(ValueError, "runs past"):
        KimiK3LayerSchedule.build(KimiK3Config, first_layer_idx=90, num_layers=10)
    with expect_error(ValueError, "invalid slice"):
        KimiK3LayerSchedule.build(KimiK3Config, first_layer_idx=0, num_layers=0)
