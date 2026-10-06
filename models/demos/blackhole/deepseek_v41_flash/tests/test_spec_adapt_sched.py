# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-free test of the adaptive verification-length scheduler (AdaptiveSpec.choose / expected_tokens / conf_report)."""
import torch

from models.demos.blackhole.deepseek_v41_flash.tt.spec_model import AdaptiveSpec


def _mk(times, policy="adapt", hyst=0.0):
    a = AdaptiveSpec.__new__(AdaptiveSpec)
    a.ks, a.times, a.policy, a.cal_a, a.cal_b, a.hyst = sorted(times), dict(times), policy, 1.0, 0.0, hyst
    a.cal_records = []
    return a


def test_choice_follows_confidence():
    t = {1: 70.0, 3: 83.0, 5: 100.0}
    a = _mk(t)
    act = torch.ones(16, dtype=torch.bool)
    hi = torch.full((16, 5), 0.97)  # near-certain drafts: longest block wins
    lo = torch.full((16, 5), 0.30)  # hopeless drafts: shortest block wins
    mid = torch.tensor([0.9, 0.8, 0.7, 0.5, 0.4]).repeat(16, 1)
    assert a.choose(hi, act) == 5
    assert a.choose(lo, act) == 1
    assert a.choose(mid, act) == 3
    E = a.expected_tokens(mid, act)
    assert abs(E[1] - 1.9) < 1e-6 and abs(E[3] - (1 + 0.9 + 0.72 + 0.504)) < 1e-6


def test_fixed_and_masking():
    a = _mk({1: 70.0, 3: 83.0}, policy="k3")
    conf = torch.full((4, 5), 0.1)
    assert a.choose(conf, torch.ones(4, dtype=torch.bool)) == 3
    a.policy = "adapt"
    act = torch.tensor([True, False, False, False])
    conf = torch.tensor([[0.95] * 5] + [[0.0] * 5] * 3)  # inactive users must not influence the choice
    assert a.choose(conf, act) == 3
