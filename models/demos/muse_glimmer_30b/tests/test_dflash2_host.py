# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the DFlash2 host side: sampler, rejection sampling, candidate selector and checkpoint lookup."""

import os

import pytest
import torch
from safetensors import safe_open

from models.demos.muse_glimmer_30b.reference.dflash2 import CandidateSelector, DFlash2Config
from models.demos.muse_glimmer_30b.tt.dflash.sampling import rejection_sample
from models.demos.muse_glimmer_30b.tt.dflash.selector import HostCandidateSelector, merge_device_topk
from models.demos.muse_glimmer_30b.tt.model import prefill_buckets, prefill_len
from models.demos.muse_glimmer_30b.tt.model_config import PREFILL_COARSE_MAX, PREFILL_MAX_PAD, dflash_checkpoint_dir
from models.demos.muse_glimmer_30b.tt.sampling import filtered_probs, normalize_sampling


@pytest.fixture(scope="module")
def ckpt():
    path = os.path.join(dflash_checkpoint_dir(), "model.safetensors")
    if not os.path.exists(path):
        pytest.skip("DFlash2 checkpoint not found")
    return path


def _ids(rows, cols):
    return torch.arange(cols, dtype=torch.long).repeat(rows, 1)


def test_filtered_probs_rules():
    # (a) top_k=1 -> one-hot on argmax
    v = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    p, i = filtered_probs(v, _ids(1, 4), 1.0, 1.0, 1)
    assert p.shape == (1, 1) and p[0, 0] == 1.0 and i[0, 0] == 1
    # (b) T=1, top_p=1, k=4 -> softmax of top-4
    v = torch.tensor([[4.0, 1.0, 3.0, 2.0, 0.0, -1.0]])
    p, i = filtered_probs(v, _ids(1, 6), 1.0, 1.0, 4)
    assert i[0].tolist() == [0, 2, 3, 1]
    torch.testing.assert_close(p[0], torch.softmax(torch.tensor([4.0, 3.0, 2.0, 1.0]), -1))
    # (c) cumulative crosses top_p exactly at index 2 -> 3 kept
    base = torch.tensor([0.5, 0.25, 0.15, 0.06, 0.04])
    v = base.log().unsqueeze(0)
    p, i = filtered_probs(v, _ids(1, 5), 1.0, 0.85, 5)
    assert int((p[0] > 0).sum()) == 3
    assert p[0, 3:].sum() == 0
    # (d) unsorted columns and -inf entries
    v = torch.tensor([[float("-inf"), 2.0, 7.0, float("-inf"), 5.0]])
    ids = torch.tensor([[10, 11, 12, 13, 14]])
    p, i = filtered_probs(v, ids, 1.0, 1.0, 3)
    assert i[0].tolist() == [12, 14, 11]
    # (e) rows sum to 1
    g = torch.Generator().manual_seed(0)
    v = torch.randn(6, 40, generator=g) * 3
    for tp in (0.3, 0.9, 1.0):
        p, _ = filtered_probs(v, _ids(6, 40), 0.7, tp, 8)
        torch.testing.assert_close(p.sum(-1), torch.ones(6), atol=1e-6, rtol=0)
    # (f) T=0.5 == softmax(values * bf16(2.0)) over top-k
    v = torch.randn(3, 20, generator=g).to(torch.bfloat16).float()
    p, i = filtered_probs(v, _ids(3, 20), 0.5, 1.0, 5)
    top = torch.topk(v, 5, dim=-1).values
    torch.testing.assert_close(p, torch.softmax(top * 2.0, -1))


def test_normalize_sampling(expect_error):
    assert normalize_sampling(None, 32) is None
    assert normalize_sampling({"temperature": 0.0}, 32) is None
    assert normalize_sampling({"temperature": 1.0, "top_p": 0.95, "top_k": 64}, 32) == (1.0, 0.95, 32)
    assert normalize_sampling({"temperature": 1.0, "top_p": 0.95, "top_k": 0}, 32) == (1.0, 0.95, 32)
    assert normalize_sampling({"temperature": 0.6, "top_k": 10}, 32) == (0.6, 1.0, 10)
    for bad in (0.0, 1.5, -0.1):
        with expect_error(ValueError, "top_p must be"):
            normalize_sampling({"temperature": 1.0, "top_p": bad}, 32)
    with expect_error(ValueError, "temperature must be"):
        normalize_sampling({"temperature": -0.5}, 32)


def _tv(hist, probs, ids, vocab):
    ref = torch.zeros(vocab)
    ref[ids] = probs
    return 0.5 * (hist / hist.sum() - ref).abs().sum().item()


def test_rejection_sample_is_lossless():
    vocab, length, kt, kq, trials = 40, 3, 8, 16, 60000
    g = torch.Generator().manual_seed(0)
    t_probs, t_ids = filtered_probs(
        torch.randn(length + 1, vocab, generator=g) * 2, _ids(length + 1, vocab), 1.0, 0.9, kt
    )
    q_rows = []
    for i in range(length):  # 5 ids inside the target top-k plus 11 random others (may overlap further)
        rest = torch.tensor([t for t in torch.randperm(vocab, generator=g).tolist() if t not in t_ids[i, :5].tolist()])
        q_rows.append(torch.cat([t_ids[i, :5], rest[: kq - 5]]))
    q_ids = torch.stack(q_rows)
    q_scores = torch.randn(length, kq, generator=g)
    q_scores[:, :5] += 2.5  # favour the shared ids so full-length acceptance is common
    q_probs = torch.softmax(q_scores, -1)
    cdf = q_probs.cumsum(-1)
    first = torch.zeros(vocab)
    second = torch.zeros(vocab)
    third = torch.zeros(vocab)
    n_second = n_bonus = 0
    for _ in range(trials):
        r = torch.rand(length, generator=g)
        pos = (r.unsqueeze(-1) > cdf).sum(-1).clamp(max=kq - 1)
        drafts = [int(q_ids[i, pos[i]]) for i in range(length)]
        acc, bonus = rejection_sample(drafts, q_probs, q_ids, t_probs, t_ids, g)
        assert 0 <= acc <= length
        seq = drafts[:acc] + [bonus]
        first[seq[0]] += 1
        if len(seq) >= 2:
            second[seq[1]] += 1
            n_second += 1
        if acc == length:
            third[seq[length]] += 1
            n_bonus += 1
    tv0 = _tv(first, t_probs[0], t_ids[0], vocab)
    tv1 = _tv(second, t_probs[1], t_ids[1], vocab)
    print(f"TV0={tv0:.4f} TV1={tv1:.4f} n_second={n_second} n_bonus={n_bonus}")
    assert tv0 < 0.015
    assert tv1 < 0.02
    assert n_bonus >= 3000, n_bonus
    tv3 = _tv(third, t_probs[3], t_ids[3], vocab)
    print(f"TV3={tv3:.4f}")
    assert tv3 < 0.03


def test_selector_sample():
    g = torch.Generator().manual_seed(1)
    vocab, rank, length, k = 50, 8, 5, 16
    sel = HostCandidateSelector(torch.randn(vocab, rank, generator=g), torch.randn(vocab, rank, generator=g))
    scores = torch.randn(length, k, generator=g)
    cands = torch.stack([torch.randperm(vocab, generator=g)[:k] for _ in range(length)])
    hp = torch.randn(length, rank, generator=g)
    path, q = sel.sample(scores, cands, hp, 3, 1.0, torch.Generator().manual_seed(7))
    assert q.shape == (length, k) and q.dtype == torch.float32
    torch.testing.assert_close(q.sum(-1), torch.ones(length), atol=1e-5, rtol=0)
    cold, _ = sel.sample(scores, cands, hp, 3, 1e-4, torch.Generator().manual_seed(7))
    assert cold == sel.select(scores, cands, hp, 3)
    again, _ = sel.sample(scores, cands, hp, 3, 1.0, torch.Generator().manual_seed(7))
    assert again == path


def test_rejection_sample_greedy_limit():
    t_ids = torch.tensor([[5], [6], [7]])
    t_probs = torch.ones(3, 1)
    q_ids = torch.tensor([[5, 1], [6, 2]])
    q_probs = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
    acc, bonus = rejection_sample([5, 6], q_probs, q_ids, t_probs, t_ids, torch.Generator().manual_seed(0))
    assert acc == 2 and bonus == 7


VOCAB = 202_048
VOCAB_LOCAL = 50_688


def test_merge_device_topk():
    torch.manual_seed(0)
    full = torch.randn(32, 4 * VOCAB_LOCAL) * 5
    # Only a few padding columns are hot: if all 704 were, shard 3's top-32 would hold no valid ids at all.
    full[:, VOCAB : VOCAB + 8] = 100.0
    vals, ids = zip(*(torch.topk(full[:, d * VOCAB_LOCAL : (d + 1) * VOCAB_LOCAL], 32) for d in range(4)))
    scores, cands = merge_device_topk(
        torch.stack(vals), torch.stack(ids), vocab_local=VOCAB_LOCAL, vocab_size=VOCAB, k=16
    )
    exp_scores, exp_cands = torch.topk(full[:, :VOCAB], 16)
    assert scores.dtype == torch.float32 and cands.dtype == torch.int64
    for r in range(32):
        assert set(cands[r].tolist()) == set(exp_cands[r].tolist())
    assert torch.allclose(scores, exp_scores)


@pytest.fixture(scope="module")
def tensors(ckpt):
    with safe_open(ckpt, framework="pt") as f:
        return {
            "pred": f.get_tensor("candidate_selector.predecessor_codebook"),
            "succ": f.get_tensor("candidate_selector.successor_codebook"),
            "hp": f.get_tensor("candidate_selector.hidden_projection.weight"),
        }


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_selector_matches_reference(tensors, seed):
    ref = CandidateSelector(DFlash2Config())
    ref.predecessor_codebook.weight.data = tensors["pred"].clone()
    ref.successor_codebook.weight.data = tensors["succ"].clone()
    ref.hidden_projection.weight.data = tensors["hp"].clone()
    ref = ref.float().eval()

    torch.manual_seed(seed)
    hidden = torch.randn(1, 15, 6656)
    logits = torch.randn(1, 15, VOCAB) * 5
    anchor = 1234
    with torch.no_grad():
        ref_path = ref.select(hidden, logits, torch.tensor([anchor]))[0][0].tolist()

    hp = hidden[0] @ tensors["hp"].float().T
    scores, cands = torch.topk(logits[0], 16)
    path = HostCandidateSelector(tensors["pred"], tensors["succ"]).select(scores, cands, hp, anchor)
    assert path == ref_path


def test_dflash_checkpoint_dir(monkeypatch, tmp_path):
    import huggingface_hub

    from models.demos.muse_glimmer_30b.tt import model_config as mc

    monkeypatch.setenv("MUSE_DFLASH_MODEL", str(tmp_path))
    assert mc.dflash_checkpoint_dir() == str(tmp_path)
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id: calls.append(repo_id) or "/snap")
    mc._hf_snapshot.cache_clear()
    monkeypatch.setenv("MUSE_DFLASH_MODEL", mc.DFLASH_REPO_ID)
    try:
        assert mc.dflash_checkpoint_dir() == "/snap"
        assert calls == [mc.DFLASH_REPO_ID]
    finally:
        mc._hf_snapshot.cache_clear()


def test_batch_layout_policy(expect_error):
    from models.demos.muse_glimmer_30b.tt.dflash.batch_generator import plan_layout, rows_for_users

    assert [rows_for_users(n) for n in (1, 2, 3, 4, 5, 8)] == [16, 16, 8, 8, 4, 4]
    with expect_error(ValueError, "use plain decode"):
        rows_for_users(9)
    lay = plan_layout([0, 1, 2], 8)
    assert lay.slots == (0, 1, 2, None)
    # same R: slots keep their groups; a new slot takes the lowest free group
    lay = plan_layout([0, 2, 5], 8, lay)
    assert lay.slots == (0, 5, 2, None)
    # new R: slots keep their old group order, new slots after them (by slot id)
    lay = plan_layout([2, 5, 0, 7, 6], 4, lay)
    assert lay.slots == (0, 5, 2, 6, 7, None, None, None)
    lay = plan_layout([5, 2], 16, lay)
    assert lay.slots == (5, 2)


def test_prefill_buckets():
    """Coarse prefill buckets: few, 256-aligned (or 128), capped by max_len, every pad below PREFILL_MAX_PAD."""
    b = prefill_buckets(PREFILL_COARSE_MAX)
    assert len(b) == 33 and b[0] == 128 and b[-1] == PREFILL_COARSE_MAX and b == sorted(set(b))
    assert all(x == 128 or x % 256 == 0 for x in b)
    for s in range(1, PREFILL_COARSE_MAX + 1):
        p = prefill_len(s)
        assert p in b and 0 <= p - s < PREFILL_MAX_PAD
    assert prefill_len(20000) == 20480 and prefill_len(140000) == 140032
    assert prefill_buckets(8448)[-1] == 8448 and prefill_len(8300, 8448) == 8448
    assert prefill_len(100, 100) == 128
