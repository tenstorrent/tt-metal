# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the logprobs payload of an argmax_ids verify (tt/spec_logprobs.py,
vllm-tt-plugin supports_spec_device_logprobs).

* The candidates (each vocabulary shard's stable top-32, gathered in shard order) hold the global top-32 of every row
  under (value descending, id ascending), ties included, so the payload's top-20 is the vocabulary's.
* The log-sum-exp from per-shard float32 sums relative to the global maximum, as the device adds them.
* The payload of a column (logprob, rank and top-N of the token the verify answers there) equals what vLLM's Sampler
  computes from the full logits, for the argmax and for any token of the top-32 (a device draw).
* PendingVerify reads the logprobs trace's outputs after the ids and draws and before the one-enqueue outputs.
"""

from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen36.tt.spec_engine import PendingVerify
from models.demos.blackhole.qwen36.tt.spec_logprobs import (
    TOP_LOGPROBS,
    gather_logprobs_from_candidates,
    lse_from_stats,
    spec_logprobs_payload,
    stable_order,
    stats_from_logits,
)
from models.demos.blackhole.qwen36.tt.spec_sample import SAMPLER_TOP_K


def _logits(g, R, V, scale=3.0, coarse=0.0, peaked=False):
    """bfloat16-valued logits [R, V]; ``coarse`` rounds them to a grid (many exact ties)."""
    x = torch.randn(R, V, generator=g) * scale
    if coarse:
        x = torch.round(x / coarse) * coarse
    if peaked:
        for r in range(R):
            idx = torch.randint(0, V, (5,), generator=g)
            x[r, idx] += torch.tensor([12.0, 10.0, 9.0, 8.0, 7.0])
    return x.to(torch.bfloat16).float()


def _device_outputs(x, shards):
    """What the logprobs trace writes for full logits [R, V]: the candidates (per shard the stable top-32 with global
    ids, in shard order) and the float32 per-shard sums of exp(x - M) added over the shards, M the row maximum."""
    R, V = x.shape
    per = V // shards
    vals, ids, parts = [], [], []
    m = x.max(dim=1, keepdim=True).values
    for s in range(shards):
        sh = x[:, s * per : (s + 1) * per]
        o = torch.sort(sh, dim=1, descending=True, stable=True).indices[:, :SAMPLER_TOP_K]
        vals.append(sh.gather(1, o))
        ids.append(o + s * per)
        parts.append((sh - m).exp().sum(1))
    return torch.cat(vals, 1).to(torch.bfloat16), torch.cat(ids, 1), torch.stack(parts).to(torch.float32).sum(0)


@pytest.mark.parametrize("shards", [1, 2, 3, 8])
def test_the_candidates_hold_the_global_top_32_with_ties(shards):
    g = torch.Generator().manual_seed(100 + shards)
    for coarse in (0.0, 0.25, 1.0):
        x = _logits(g, 12, shards * 240, scale=2.0, coarse=coarse)
        vals, ids, _ = _device_outputs(x, shards)
        order = stable_order(vals.float(), ids)
        want = torch.sort(x, dim=1, descending=True, stable=True).indices[:, :SAMPLER_TOP_K]
        assert torch.equal(torch.gather(ids, 1, order)[:, :SAMPLER_TOP_K], want), (shards, coarse)


@pytest.mark.parametrize("peaked", [False, True])
def test_the_lse_of_float32_shard_sums_is_the_vocabulary_lse(peaked):
    """At the Qwen3.8 shape (8 shards of 31040), float32 sums relative to the global maximum give the float64
    log-sum-exp to 1e-5."""
    g = torch.Generator().manual_seed(7 + peaked)
    x = _logits(g, 4, 8 * 31040, scale=3.0, peaked=peaked)
    _, _, sums = _device_outputs(x, 8)
    lse = lse_from_stats(x.max(dim=1).values, sums)
    assert (lse - torch.logsumexp(x.double(), dim=-1)).abs().max() <= 1e-5
    m, l_raw, _ = stats_from_logits(x, torch.ones(4))
    assert torch.allclose(lse_from_stats(m, l_raw), torch.logsumexp(x.double(), dim=-1))


@pytest.mark.parametrize("top", [0, 1, 5, TOP_LOGPROBS])
def test_the_payload_equals_vllms_logprobs_of_the_same_logits(top):
    """For the argmax and for tokens deeper in the top-32 (device draws), the payload's logprob is vLLM's raw logprob
    to 2e-5, its rank vLLM's rank (or a lower bound when the token ties a shard's last candidate), and its top-N ids
    and logprobs vLLM's, the lowest id first among equal values."""
    pytest.importorskip("vllm")
    from vllm.v1.sample.sampler import Sampler

    g = torch.Generator().manual_seed(300 + top)
    R, shards = 24, 4
    for coarse in (0.0, 0.5, 2.0):
        x = _logits(g, R, shards * 400, scale=2.5, coarse=coarse)
        vals, ids, sums = _device_outputs(x, shards)
        order = torch.sort(x, dim=1, descending=True, stable=True).indices
        depth = torch.randint(0, SAMPLER_TOP_K, (R, 1), generator=g)
        depth[: R // 3] = 0  # argmax rows
        chosen = order.gather(1, depth).reshape(-1)
        lp, rank, top_lp, top_ids = spec_logprobs_payload(vals, ids, sums, chosen, top=top)
        ref = Sampler.gather_logprobs(Sampler.compute_logprobs(x), top, chosen)
        assert torch.allclose(lp, ref.logprobs[:, 0], atol=2e-5, rtol=0), float((lp - ref.logprobs[:, 0]).abs().max())
        want_rank = ref.selected_token_ranks.to(torch.int32)
        assert bool((rank <= want_rank).all())
        cval = x.gather(1, chosen.reshape(-1, 1))
        shard_last = vals.float().reshape(R, shards, SAMPLER_TOP_K)[:, :, -1]
        exact = ~(shard_last == cval).any(dim=1)
        assert torch.equal(rank[exact], want_rank[exact])
        if top:
            assert torch.equal(top_ids.to(torch.int64), order[:, :top])
            assert torch.allclose(top_lp, ref.logprobs[:, 1:], atol=2e-5, rtol=0)


def test_a_live_row_whose_token_is_no_candidate_raises():
    g = torch.Generator().manual_seed(1)
    x = _logits(g, 2, 2 * 200)
    vals, ids, sums = _device_outputs(x, 2)
    outside = torch.sort(x, dim=1, descending=True, stable=True).indices[:, -1]
    with pytest.raises(RuntimeError, match=r"verify rows \[1\]"):
        spec_logprobs_payload(vals, ids, sums, outside, live=torch.tensor([False, True]))
    lp, rank, _, _ = spec_logprobs_payload(vals, ids, sums, outside, live=torch.tensor([False, False]))
    assert int(rank[0]) == 2 * SAMPLER_TOP_K + 1  # a padding row: nothing reads it


def test_the_missing_and_lower_bound_flags():
    ids = torch.arange(64).reshape(1, 64)
    vals = torch.full((1, 64), 5.0)  # a 64-way tie: every chosen is a lower bound
    tok, lp, rank, lower, missing = gather_logprobs_from_candidates(
        ids, vals, torch.tensor([6.0]), torch.tensor([10]), 2
    )
    assert bool(lower[0]) and not bool(missing[0]) and int(rank[0]) == 64
    tok, lp, rank, lower, missing = gather_logprobs_from_candidates(
        ids, vals, torch.tensor([6.0]), torch.tensor([999]), 0
    )
    assert bool(missing[0]) and int(rank[0]) == 65 and float(lp[0, 0]) == -1.0
    with pytest.raises(ValueError):
        gather_logprobs_from_candidates(ids, vals, torch.tensor([6.0]), torch.tensor([1]), 65)


def test_a_pending_verify_reads_the_payload_after_ids_and_draws():
    """Read order of a one-enqueue verify with logprobs: ids, the logprobs trace's (candidates, ids, sums), then the
    drafter pack and the drafts. The payload describes the ids wait() returns, the unmasked ones."""
    B, K = 2, 3
    T = K + 1
    g = torch.Generator().manual_seed(3)
    x = _logits(g, B * T, 2 * 300)
    vals, ids, sums = _device_outputs(x, 2)
    argmax = x.argmax(dim=1)
    lanes = torch.zeros(1, 1, 1, 32, dtype=torch.int64)
    lanes[0, 0, 0, : B * T] = argmax
    pack, drafts = torch.zeros(4, 8), [torch.full((1, 1, 1, 32), 9) for _ in range(K)]
    host = [
        lanes,
        vals.reshape(1, 1, B * T, -1),
        ids.reshape(1, 1, B * T, -1),
        sums.reshape(1, 1, -1, 1),
        pack,
        *drafts,
    ]
    reads = SimpleNamespace(event="event", result=lambda: host)
    plan = SimpleNamespace(rows=None, p0=[0, 0], declined=[False, False], keep=[False, False])
    step = PendingVerify(B, K, [1, 3], [True, True], None, 1, set(), plan=plan, reads=reads, logprobs=True)

    lp, rank, top_lp, top_ids = step.logprobs_result()

    assert step.result().tolist() == [
        [int(argmax[0]), int(argmax[1]), -1, -1],
        [int(argmax[4]), int(argmax[5]), int(argmax[6]), int(argmax[7])],
    ]
    assert lp.shape == rank.shape == (B, T) and top_ids.shape == top_lp.shape == (B, T, TOP_LOGPROBS)
    want = torch.log_softmax(x.double(), dim=-1).gather(1, argmax.reshape(-1, 1)).reshape(B, T)
    assert torch.allclose(lp.double(), want, atol=2e-5)
    want_rank = (x >= x.max(dim=1, keepdim=True).values).sum(dim=1).reshape(B, T)
    assert torch.equal(rank.to(torch.int64), want_rank)
    assert torch.equal(step.pack, pack.to(torch.int64)) and step.drafts.shape == (B, K)
    plain = PendingVerify(B, K, [1, 3], [True, True], None, 1, set(), reads=SimpleNamespace(result=lambda: [lanes]))
    assert plain.logprobs_result() is None
