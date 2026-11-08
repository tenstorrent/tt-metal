# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) of the host logic of the load-based decode buckets (DSV41_LOAD_TRACES): per-bucket round-time tables of the adaptive k (spec_adapt.AdaptiveVerify), and the slot placement that
keeps the live users inside the smallest bucket (vllm_state.SlotTable.claim_rowaware + users_needed + pick_bucket). Run (worktree root):
  python_env/bin/python -m pytest --noconftest -c /dev/null --rootdir=/tmp models/demos/blackhole/deepseek_v41_flash/tests/test_load_traces_cpu.py -q"""

import torch

from models.demos.blackhole.deepseek_v41_flash.tt import vllm_state as VS
from models.demos.blackhole.deepseek_v41_flash.tt.spec_adapt import AdaptiveVerify, bucket_route, nearest_runner_k


class FakeRunner:
    bucket_ok = True

    def bucket_for(self, pos):
        return 0


class FakeAd:
    def __init__(self):
        self.runners = {3: FakeRunner(), 5: FakeRunner()}
        self.times = {3: 41.6, 5: 53.5}
        self.policy = "adapt"
        self.hyst = 0.0
        self.seedr = None

    def expected_tokens(self, conf, active):
        S = torch.cumprod(conf[active], dim=1).mean(0)
        return {k: 1.0 + float(S[:k].sum()) for k in (3, 5)}


def mk():
    av = AdaptiveVerify(FakeAd(), log=lambda m: None)
    av.probe_every = 0
    return av


def test_bucket_tables_are_separate():
    av = mk()
    av.set_bucket_times(1, {3: 25.3, 5: 25.9})
    conf = torch.full((4, 5), 0.8)
    # full trace: k=5 costs 12 ms more than k=3; B=4 trace: only 0.6 ms more -> the cheaper trace prefers the longer block
    k_full = av.choose(conf, max_pos=1000)
    k_b4 = av.choose(conf, max_pos=1000, bucket=1)
    E = FakeAd().expected_tokens(conf, torch.ones(4, dtype=torch.bool))
    assert k_full == max((3, 5), key=lambda k: E[k] / {3: 41.6, 5: 53.5}[k])
    assert k_b4 == max((3, 5), key=lambda k: E[k] / {3: 25.3, 5: 25.9}[k])
    assert k_b4 == 5


def test_note_round_updates_only_its_bucket():
    av = mk()
    av.set_bucket_times(2, {3: 32.6, 5: 38.4})
    av.note_round(3, 30.0, bucket=2)
    assert av.btimes[2][3] != 32.6 and av.ad.times[3] == 41.6
    av.note_round(5, 60.0)
    assert av.ad.times[5] != 53.5 and av.btimes[2][5] == 38.4
    assert av.k_hist[(2, 3)] == 1 and av.k_hist[5] == 1


def test_unknown_bucket_falls_back_to_the_full_table():
    av = mk()
    assert av._tab(1)[0] is av.ad.times and av._tab(None)[0] is av.ad.times


def place(n_users, U=4, rows=4):
    """claim n_users one after another as the interleaved prefill does (each prompt alone in flight: busy rows empty) and return the live set"""
    st = VS.SlotTable(rows * U, rows * U)
    for i in range(n_users):
        p, _ = st.claim_rowaware(i, set(), U)
    return st


def test_packing_keeps_c4_in_the_b4_bucket_and_c8_in_b8():
    buckets = [1, 2, 4]
    for n, want in [(1, 1), (2, 1), (3, 1), (4, 1), (5, 2), (8, 2), (9, 4), (16, 4)]:
        st = place(n)
        assert VS.pick_bucket(st.users_needed(4), buckets) == want, (n, sorted(st.live))


def test_straggler_pins_the_bigger_bucket_until_it_finishes():
    st = place(8)
    for p in sorted(st.live):
        if p % 4 == 0:  # the four users at u0 finish
            st.live.discard(p)
    assert st.users_needed(4) == 2  # u1 of every row still live: the B=8 trace
    for p in sorted(st.live):
        st.live.discard(p)
    assert st.users_needed(4) == 1


def test_nearest_runner_rule():
    assert nearest_runner_k([3], 0) == 3  # shorter block: padded up to the only runner
    assert nearest_runner_k([3], 3) == 3
    assert nearest_runner_k([3], 5) == 3  # longer block: cut
    assert nearest_runner_k([3, 5], 4) == 5
    assert nearest_runner_k([5, 3], 2) == 3
    assert nearest_runner_k([3, 5], 5) == 5


def test_bucket_route_block_k_below_equal_above():
    """the 2026-11-08 crash: bucket holds k=3 only; a block of k=5 (offered on the full trace) must go to the full runner, never to the bucket"""
    assert bucket_route([3], 0) == 3  # below: padded
    assert bucket_route([3], 2) == 3
    assert bucket_route([3], 3) == 3  # equal
    assert bucket_route([3], 5) is None  # above: full trace
    assert bucket_route([3, 5], 4) == 5
    assert bucket_route([3, 5], 5) == 5
    assert bucket_route([5], 3) == 5


def test_choose_in_a_bucket_offers_only_its_ks_and_mixed_users_pin_the_bucket():
    """per-bucket adaptive choice offers only the bucket's k; users at in-row index >= Ub (a straggler) pin the full width (no bucket => the full set of k's)"""
    ad = AdaptiveVerify.__new__(AdaptiveVerify)  # (no device: only tables and the pure rule are used)
    ad.ks = [3, 5]
    ad.btimes = {1: {3: 30.0}}
    ad.bsince = {}
    ad.ad = type("A", (), {"times": {3: 50.0, 5: 52.0}})()
    ad.since = {}
    ad.candidates = lambda max_pos=None: [3, 5]
    ad.runners = {}
    ad.probe_every = 0
    ad._cur = None
    ad.policy = None
    import torch

    conf = torch.full((2, 5), 0.9)
    k_b = AdaptiveVerify.choose(ad, conf, torch.ones(2, dtype=torch.bool), max_pos=100, bucket=1)
    assert k_b == 3  # the bucket only has k=3
    # without a bucket (full trace, or a straggler pinned the full width) both k's are candidates
    ad.btimes = {}
    assert AdaptiveVerify.candidates(ad) == [3, 5]
