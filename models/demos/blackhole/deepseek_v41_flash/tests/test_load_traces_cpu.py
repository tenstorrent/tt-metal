# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) of the host logic of the load-based decode buckets (DSV41_LOAD_TRACES): per-bucket round-time tables of the adaptive k (spec_adapt.AdaptiveVerify), and the slot placement that
keeps the live users inside the smallest bucket (vllm_state.SlotTable.claim_rowaware + users_needed + pick_bucket). Run (worktree root):
  python_env/bin/python -m pytest --noconftest -c /dev/null --rootdir=/tmp models/demos/blackhole/deepseek_v41_flash/tests/test_load_traces_cpu.py -q"""

import torch

from models.demos.blackhole.deepseek_v41_flash.tt import vllm_state as VS
from models.demos.blackhole.deepseek_v41_flash.tt.spec_adapt import AdaptiveVerify, nearest_runner_k


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
