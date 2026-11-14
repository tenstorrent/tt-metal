# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the vLLM adapter's bookkeeping (no device, no checkpoint)."""

from types import SimpleNamespace

import torch

from models.demos.muse_glimmer_30b.tt import generator_vllm as gv
from models.demos.muse_glimmer_30b.tt.generator_vllm import MuseGlimmerForConditionalGeneration, prefill_buckets
from models.demos.muse_glimmer_30b.tt.model import DECODE_B_PAD


def _adapter(sliding_kv_blocks=3, n_blocks=7):
    a = MuseGlimmerForConditionalGeneration()
    a.args = SimpleNamespace(sliding_kv_blocks=sliding_kv_blocks, max_prefill_len=131072, block_size=64)
    a.n_blocks = n_blocks
    return a


def test_prefill_buckets_default_step():
    b = prefill_buckets(131072, step_div=4)
    assert b[:3] == [128, 256, 512]
    assert b[-1] == 131072
    assert len(b) == 33
    assert b == sorted(set(b))


def test_prefill_buckets_pad_bound():
    for div in (1, 4, 8):
        b = prefill_buckets(131072, step_div=div)
        for lo, hi in zip(b[1:], b[2:]):
            assert hi - lo <= max(256, lo // div)


def test_prefill_buckets_short_max_len():
    assert prefill_buckets(128, step_div=4) == [128]
    assert prefill_buckets(1000, step_div=4)[-1] == 1000


def test_prefill_buckets_power_of_two():
    assert prefill_buckets(2048, step_div=1) == [128, 256, 512, 1024, 2048]


def test_bucket_lookup_and_overflow(expect_error):
    a = _adapter()
    a.buckets = [128, 256, 512]
    assert [a._bucket(n) for n in (1, 128, 129, 512)] == [128, 128, 256, 512]
    with expect_error(ValueError, "exceeds max_prefill_len"):
        a._bucket(513)


def test_ring_table_wraps_inside_slot():
    a = _adapter(sliding_kv_blocks=3, n_blocks=7)
    t = a._ring_table([0, 2, DECODE_B_PAD])
    assert t.dtype == torch.int32 and t.shape == (3, 7)
    assert t[0].tolist() == [0, 1, 2, 0, 1, 2, 0]
    assert t[1].tolist() == [6, 7, 8, 6, 7, 8, 6]
    assert t[2, 0].item() == DECODE_B_PAD * 3


def _check_permutation(a):
    assert sorted(a._row_ring) == list(range(DECODE_B_PAD))


def test_remap_identity_keeps_rings():
    a = _adapter()
    a._remap_rings(list(range(DECODE_B_PAD)))
    assert a._row_ring == list(range(DECODE_B_PAD))


def test_remap_moves_ring_with_request():
    a = _adapter()
    remap = list(range(DECODE_B_PAD))
    remap[0], remap[5] = 5, 0  # swap rows 0 and 5
    a._remap_rings(remap)
    assert a._row_ring[0] == 5 and a._row_ring[5] == 0
    _check_permutation(a)


def test_remap_compaction_gives_vacated_row_an_unowned_ring():
    a = _adapter()
    remap = list(range(DECODE_B_PAD))
    remap[1] = 3  # row 1 now holds the request that was at row 3; row 3 is vacated
    a._remap_rings(remap)
    assert a._row_ring[1] == 3
    assert a._row_ring[3] == 1  # the ring row 1 gave up
    _check_permutation(a)


def test_remap_random_sequences_stay_permutations():
    g = torch.Generator().manual_seed(0)
    a = _adapter()
    for _ in range(200):
        remap = torch.randperm(DECODE_B_PAD, generator=g).tolist()
        owner = {i: a._row_ring[i] for i in range(DECODE_B_PAD)}
        a._remap_rings(remap)
        _check_permutation(a)
        for i, j in enumerate(remap):
            assert a._row_ring[i] == owner[j]


def test_slot_bucket_uses_highest_live_row(monkeypatch):
    monkeypatch.setattr(gv, "SLOT_TRACES", True)
    sb = MuseGlimmerForConditionalGeneration._slot_bucket
    pos = torch.full((DECODE_B_PAD,), -1)
    assert sb(pos) == 1
    pos[0] = 5
    assert sb(pos) == 1
    pos[2] = 5
    assert sb(pos) == 4
    pos[9] = 5
    assert sb(pos) == 16
    pos[DECODE_B_PAD - 1] = 5
    assert sb(pos) == DECODE_B_PAD


def test_slot_bucket_disabled(monkeypatch):
    monkeypatch.setattr(gv, "SLOT_TRACES", False)
    assert MuseGlimmerForConditionalGeneration._slot_bucket(torch.zeros(4)) == DECODE_B_PAD


def test_max_tokens_all_users(monkeypatch):
    monkeypatch.delenv("MUSE_VLLM_KV_TOKENS", raising=False)
    f = MuseGlimmerForConditionalGeneration.get_max_tokens_all_users
    assert f(max_model_len=131072, max_num_seqs=32) == gv.DEFAULT_KV_TOKENS
    assert f(max_model_len=1000, max_num_seqs=2) == 2000
    monkeypatch.setenv("MUSE_VLLM_KV_TOKENS", "4096")
    assert f(max_model_len=131072, max_num_seqs=32) == 4096


def test_initialize_rejects_unsupported_configs(expect_error):
    mesh = SimpleNamespace(shape=(1, 4))
    init = MuseGlimmerForConditionalGeneration.initialize_vllm_model
    with expect_error(ValueError, "tt_data_parallel"):
        init(None, mesh, 1, tt_data_parallel=2)
    with expect_error(ValueError, "optimization"):
        init(None, mesh, 1, optimizations="x")
    with expect_error(ValueError, "max_num_seqs"):
        init(None, mesh, DECODE_B_PAD + 1)
    with expect_error(ValueError, "1, TP"):
        init(None, SimpleNamespace(shape=(2, 4)), 1)


def test_prefill_rejects_nonzero_start(expect_error):
    a = _adapter()
    with expect_error(ValueError, "position 0"):
        a.prefill_forward(torch.zeros(1, 8, dtype=torch.int64), torch.zeros(1, 1), None, [8], start_pos=[4])


def test_capabilities_route_penalties_to_host():
    assert MuseGlimmerForConditionalGeneration.model_capabilities["supports_device_penalties"] is False
