# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the V4.1 routed-expert placement (tt/v41/expert_placement.py, beads 8y7.9.8, 8y7.9.12): the placement
method on hand-checkable loads, the placement of the checkpoint layers, relabelling invariance of the MoE, and
the weight-cache key."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.tests.v41 import weight_cache as wc
from models.demos.deepseek_v3_d_p.tt.v41 import expert_placement as P
from models.demos.deepseek_v3_d_p.tt.v41.moe import _to_slot_order

EXPERTS, CHIPS, MESH = 384, 8, (2, 4)  # MESH: (dispatch group size, dispatch groups)
PROFILED = (0, 2, 3, 20, 21, 24)


def test_chip_cost_model():
    # 2 chips, expert 1 without tokens: chip 0 = 1 active expert, 10 pairs; chip 1 = 2 active, 6 pairs
    counts, chip = np.array([[10.0, 0.0, 4.0, 2.0]]), np.array([0, 0, 1, 1])
    a, p = P.ACTIVE_EXPERT_US, P.PAIR_US
    np.testing.assert_allclose(P.chip_costs(counts, chip, 2), [[a + 10 * p, 2 * a + 6 * p]])
    np.testing.assert_allclose(P.chip_costs(counts * 2, chip, 2), [[a + 20 * p, 2 * a + 12 * p]])  # pairs scale


def test_active_experts_are_spread():
    # 4 experts with one pair each and 4 without: every chip reads the weights of 2 active experts
    chip = P.place(np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]), 2)
    assert sorted(chip[:4]) == [0, 0, 1, 1] and sorted(chip) == [0] * 4 + [1] * 4, chip


def test_placement_balances_every_prompt():
    # two prompts with opposite hot experts (0 in the first, 1 in the second) and two warm ones: only the hot
    # pair on one chip balances both prompts (per prompt 1 + 2 active experts, 6000 + 6000 pairs); any other pairing
    # puts 9000 pairs on one chip
    chip = P.place(np.array([[6000.0, 0.0, 3000.0, 3000.0], [0.0, 6000.0, 3000.0, 3000.0]]), 2)
    assert chip[0] == chip[1] != chip[2] == chip[3], chip


@pytest.mark.parametrize("layer", PROFILED)
def test_checkpoint_layer_placement(layer):
    order = P.expert_order(layer, EXPERTS, *MESH)
    assert sorted(order) == list(range(EXPERTS))
    per_chip = EXPERTS // CHIPS
    chips = [order[c * per_chip : (c + 1) * per_chip] for c in range(CHIPS)]
    assert all(list(c) == sorted(c) for c in chips)  # ascending ids within a chip
    P.expert_order.cache_clear()
    assert P.expert_order(layer, EXPERTS, *MESH) == order  # deterministic
    # a lower modelled cost of the most expensive chip on the training prompts than the checkpoint order
    counts = P.profile_counts(layer)
    chip = np.empty(EXPERTS, dtype=int)
    chip[list(order)] = np.arange(EXPERTS) // per_chip
    worst = lambda assign: P.chip_costs(counts, assign, CHIPS).max(1).mean()
    assert worst(chip) < worst(np.arange(EXPERTS) // per_chip)


def test_profile_rows_scale_to_the_chunk():
    # every profile row is scaled from its own token count (a prompt shorter than the long row keeps all its tokens)
    profile = P.load_profile()
    rows, raw = profile["rows"], np.asarray(profile["counts"]["21"], dtype=np.float64)
    assert raw.shape[0] == len(rows) and {r["tokens"] for r in rows} >= {2048, 5120}
    k = next(i for i, r in enumerate(rows) if r["tokens"] == 2048)
    np.testing.assert_allclose(P.profile_counts(21)[k], raw[k] * 2.5)
    np.testing.assert_allclose(P.profile_counts(21).sum(1), P.CHUNK_TOKENS * 6)  # 6 routed experts per token


def test_checkpoint_order_without_profile():
    assert P.expert_order(1, EXPERTS, *MESH) is None  # no checkpoint shards, no profile
    assert P.expert_order(2, 128, *MESH) is None  # SmallV41Config expert count
    assert P.expert_order(2, EXPERTS, 7, 1) is None  # chips that do not divide the experts


def _moe(weights, x):
    """The reference MoE routing (Gate.forward: sqrtsoftplus, bias-steered top-k, normalized weights) + routed sum."""
    gate = weights["gate_weights"]
    scores = F.softplus(x @ gate["weight"].T).sqrt()
    idx = (scores + gate["e_score_correction_bias"]).topk(2, dim=-1)[1]
    w = scores.gather(1, idx)
    w = w / w.sum(-1, keepdim=True)
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(idx.shape[1]):
            e = weights["routed_expert_weights"][idx[t, k]]
            h = F.silu(x[t] @ e["gate_proj"].T) * (x[t] @ e["up_proj"].T)
            out[t] += w[t, k] * (h @ e["down_proj"].T)
    return out, idx


def test_relabelling_keeps_the_moe_output():
    g = torch.Generator().manual_seed(0)
    n, d, h = 8, 16, 8
    weights = {
        "gate_weights": {
            "weight": torch.randn(n, d, generator=g),
            "e_score_correction_bias": torch.randn(n, generator=g),
        },
        "gate_bias_vl": torch.randn(n, generator=g),
        "routed_expert_weights": [
            {
                k: torch.randn(*s, generator=g)
                for k, s in (("gate_proj", (h, d)), ("up_proj", (h, d)), ("down_proj", (d, h)))
            }
            for _ in range(n)
        ],
        "shared_expert_weights": {"gate_proj": torch.randn(h, d, generator=g)},
    }
    order = torch.randperm(n, generator=g).tolist()
    relabelled = _to_slot_order(weights, order)
    assert torch.equal(relabelled["gate_bias_vl"], weights["gate_bias_vl"][order])
    assert relabelled["shared_expert_weights"] is weights["shared_expert_weights"]
    x = torch.randn(5, d, generator=g)
    out, idx = _moe(weights, x)
    out_slots, idx_slots = _moe(relabelled, x)
    assert torch.equal(torch.tensor(order)[idx_slots], idx)  # slot s holds checkpoint expert order[s]
    assert torch.equal(out_slots, out)


def test_profile_data_is_in_the_weight_cache_key(monkeypatch, tmp_path):
    spec = orc.real_spec((2, 3, 20, 21, 24), 2048, candidate_topk_blocks=96, checkpoint=orc.HF_SNAPSHOT)
    base = wc.weight_cache_dir(spec, (2, 4))
    edited = tmp_path / "profile.json"
    edited.write_bytes(P.PROFILE_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(wc, "CONVERSION_DATA", (edited,))
    assert wc.weight_cache_dir(spec, (2, 4)) != base
