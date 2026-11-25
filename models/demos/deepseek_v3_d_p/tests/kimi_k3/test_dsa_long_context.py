# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host contracts of ``dsa_long_context.py``, the scoring of ``test_k31_dsa_long_context.py``: hand-checked FP64
attention, the near-tie baseline's selection, scored rows, fail-closed bars, and the recompute check of a whole run
(``RunScorer``, on a stub golden). CPU only, tiny dimensions."""

from __future__ import annotations

import math

import pytest
import torch

from models.demos.deepseek_v3_d_p.tests.accuracy.windowed import RowSeries, check
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import dsa_long_context as dsa
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import k31_chain_golden

ONE = dsa.Dims(heads=1, nope=1, rope=1, v=1, lora=1, topk=4)
F64 = torch.float64


def unit_weights() -> dict[str, torch.Tensor]:
    """W_k = W_v = 1 and o_proj = 1: the output is the gate times the softmax-weighted mean of the latents."""
    return {
        "self_attn.kv_b_proj.weight": torch.ones(2, 1, dtype=F64),
        "self_attn.o_proj.weight": torch.ones(1, 1, dtype=F64),
    }


def hand_query(score_on_kpe: float) -> dict[str, torch.Tensor]:
    """One query row: no latent score, ``score_on_kpe`` on the rope part, gate 0.5."""
    return {
        "q_latent": torch.zeros(1, 1, 1, dtype=F64),
        "q_pe": torch.full((1, 1, 1), score_on_kpe, dtype=F64),
        "gate": torch.full((1, 1), 0.5, dtype=F64),
    }


LATENT = torch.tensor([[2.0], [4.0], [10.0]], dtype=F64)


def attend(q, latent, kpe, selection, w, dims):
    """Test oracle, one query row at a time in the unbatched form: gated absorbed MLA of ``q`` (``dsa.queries``) over
    each row's selected keys (negative entries select nothing)."""
    kvb = w["self_attn.kv_b_proj.weight"].reshape(dims.heads, dims.nope + dims.v, dims.lora)
    out = []
    for i, sel in enumerate(selection):
        keys = sel[sel >= 0]
        scores = (q["q_latent"][i] @ latent[keys].T + q["q_pe"][i] @ kpe[keys].T) * (dims.nope + dims.rope) ** -0.5
        mixed = scores.softmax(-1) @ latent[keys]  # [heads, lora]
        heads = torch.einsum("hl,hvl->hv", mixed, kvb[:, dims.nope :]).reshape(-1)
        out.append((heads * q["gate"][i]) @ w["self_attn.o_proj.weight"].T)
    return torch.stack(out)


def production(q, latent, kpe, selection, w, dims):
    """The batched path ``score_rows`` computes its reference with."""
    scores, lat = dsa._scores(q, 0, latent, kpe, selection, dims)
    return dsa._output(q, dsa._heads(scores.softmax(-1), lat, w, dims), w)


@pytest.mark.parametrize("path", [attend, production], ids=["oracle", "production"])
def test_attention_by_hand(path):
    kpe = torch.tensor([[1.0], [0.0], [0.0]], dtype=F64)
    w = unit_weights()
    # Equal scores: 0.5 * mean(2, 4) = 1.5; a -1 entry selects nothing.
    assert path(hand_query(0.0), LATENT, kpe, torch.tensor([[0, 1, -1]]), w, ONE).item() == pytest.approx(1.5)
    assert path(hand_query(0.0), LATENT, kpe, torch.tensor([[0, 2]]), w, ONE).item() == pytest.approx(3.0)
    # Score sqrt(2) ln 3 on key 0 at scale 2^-0.5: weights 3/4, 1/4, so 0.5 * (0.75 * 2 + 0.25 * 4) = 1.25.
    q = hand_query(math.sqrt(2) * math.log(3))
    assert path(q, LATENT, kpe, torch.tensor([[0, 1]]), w, ONE).item() == pytest.approx(1.25)


def test_tail_dropped_removes_the_lowest_ranked_valid_keys():
    want = torch.tensor([[5, 3, 1, -1], [7, 6, 4, 2]])
    assert dsa.tail_dropped(want, torch.tensor([1, 0])).tolist() == [[5, 3, -1, -1], [7, 6, 4, 2]]
    assert dsa.tail_dropped(want, torch.tensor([3, 2])).tolist() == [[-1, -1, -1, -1], [7, 6, -1, -1]]


def test_device_selection_and_future_keys():
    assert dsa.device_selection(torch.tensor([[0xFFFFFFFF, 3]])).tolist() == [[-1, 3]]
    # rows 2 and 3: row 2 selects key 5 (future), row 3 selects keys 0..3 (none in its future)
    assert dsa.future_keys(torch.tensor([[0, 1, 5], [0, 3, -1]]), 2).tolist() == [1, 0]


def test_scored_rows():
    # one partial window [2048, 5120): round(4 * 3072 / 16384) = 1 row (its first), plus the sampled row 4088
    rows = dsa.scored_rows(5120, torch.tensor([0, 2044, 4088]), dsa.Dims(), per_window=4)
    assert rows.tolist() == [2048, 4088]
    # a full window past top-k: 4 rows evenly spread over [16384, 32768), and none below top-k
    rows = dsa.scored_rows(32768, torch.tensor([], dtype=torch.long), dsa.Dims(), per_window=4)
    assert rows.tolist()[-4:] == [16384, 21845, 27306, 32767] and rows.min() >= 2048


def small_case(seed: int = 0):
    """Random weights and keys of a small layer: 2 heads, 6 keys, top-3."""
    dims = dsa.Dims(heads=2, nope=4, rope=2, v=3, lora=5, topk=3)
    g = torch.Generator().manual_seed(seed)
    hidden, q_lora = 8, 6
    w = {
        "input_layernorm.weight": torch.ones(hidden, dtype=F64),
        "self_attn.q_a_proj.weight": torch.randn(q_lora, hidden, generator=g, dtype=F64),
        "self_attn.q_a_layernorm.weight": torch.ones(q_lora, dtype=F64),
        "self_attn.q_b_proj.weight": torch.randn(dims.heads * (dims.nope + dims.rope), q_lora, generator=g, dtype=F64),
        "self_attn.kv_b_proj.weight": torch.randn(dims.heads * (dims.nope + dims.v), dims.lora, generator=g, dtype=F64),
        "self_attn.g_proj.weight": torch.randn(dims.heads * dims.v, hidden, generator=g, dtype=F64),
        "self_attn.o_proj.weight": torch.randn(hidden, dims.heads * dims.v, generator=g, dtype=F64),
    }
    latent = torch.randn(6, dims.lora, generator=g, dtype=F64)
    kpe = torch.randn(6, dims.rope, generator=g, dtype=F64)
    normed = dsa.normed_input(torch.randn(2, hidden, generator=g), w, dims)
    return dims, w, latent, kpe, normed


def test_score_rows_arms():
    dims, w, latent, kpe, normed = small_case()
    rows = torch.tensor([4, 5])
    want = torch.tensor([[3, 1, 0], [5, 2, 4]])
    q = dsa.queries(normed, w, dims)
    ref = attend(q, latent, kpe, want, w, dims)
    # Same selection, exact output: every arm is exact and nothing is missed.
    arms, got_ref = dsa.score_rows(rows, normed, ref, want, want, latent, kpe, w, dims)
    assert torch.allclose(got_ref, ref)
    for sums in (arms.device, arms.sel, arms.tail):
        assert sums[:, 6].abs().max() < 1e-24  # SUM_EE: squared error
    assert arms.missed_key_max.tolist() == [0.0, 0.0]
    # Row 4 misses key 0 (its lowest-ranked) for key 2: the selection arm is FP64 attention over [3, 1, 2], the tail
    # baseline over [3, 1]; row 5 is unchanged.
    got = torch.tensor([[3, 1, 2], [5, 2, 4]])
    arms, _ = dsa.score_rows(rows, normed, ref, got, want, latent, kpe, w, dims)
    sel = attend(q, latent, kpe, got, w, dims)
    tail = attend(q, latent, kpe, torch.tensor([[3, 1, -1], [5, 2, 4]]), w, dims)
    assert arms.sel[:, 6].tolist() == pytest.approx((sel - ref).square().sum(1).tolist())
    assert arms.tail[:, 6].tolist() == pytest.approx((tail - ref).square().sum(1).tolist())
    assert arms.sel[0, 6] > 0 and arms.missed_key_max[0] > 0 and arms.missed_key_max[1] == 0


def windows_of(device_out: torch.Tensor, got: torch.Tensor):
    """``measure``'s windows for the small case's two rows, scored as rows 4 and 5 of a 6-row run."""
    dims, w, latent, kpe, normed = small_case()
    rows, want = torch.tensor([4, 5]), torch.tensor([[3, 1, 0], [5, 2, 4]])
    arms, _ = dsa.score_rows(rows, normed, device_out, got, want, latent, kpe, w, dims)
    series = RowSeries()
    series.append(
        0,
        recall=torch.ones(6),
        misses_all=torch.zeros(6, dtype=torch.long),
        misses_top_half=torch.zeros(6, dtype=torch.long),
        future=torch.zeros(6, dtype=torch.long),
    )
    series.add_events(
        "scored",
        row=rows,
        device=arms.device,
        sel=arms.sel,
        tail=arms.tail,
        missed_key_max=arms.missed_key_max,
    )
    floor = dsa.floor_of(arms.device)
    return series.windows(6, lambda win: dsa.measure(win, floor, dims)), dims, w, latent, kpe, normed, want


def test_bars_fail_closed():
    dims, w, latent, kpe, normed = small_case()
    want = torch.tensor([[3, 1, 0], [5, 2, 4]])
    ref = attend(dsa.queries(normed, w, dims), latent, kpe, want, w, dims)
    bars = dsa.bars({"sel_nrmse_median": 1e-9, "attn_nrmse": 1e-9, "attn_rms_err": 1e-9})
    exact, *_ = windows_of(ref, want)
    assert check(exact, bars) == []
    # A non-finite device output fails its window, not passes it.
    broken = ref.clone()
    broken[1, 0] = math.nan
    failures = check(windows_of(broken, want)[0], bars)
    assert any("attn_nrmse nan is not finite" in f for f in failures), failures
    # A different selection on both rows fails the selection bar, even with the device output exact.
    swapped = torch.tensor([[3, 1, 2], [5, 2, 1]])
    failures = check(windows_of(ref, swapped)[0], bars)
    assert any(f.startswith("sel_nrmse_median") for f in failures), failures


def test_unset_bar_is_refused(expect_error):
    """A NaN bar would pass every window (no comparison with NaN fails), so a rung without values is refused."""
    with expect_error(ValueError, "no bar value"):
        dsa.bars({"sel_nrmse_median": math.nan, "attn_nrmse": 0.1})


class StubGolden:
    """What ``RunScorer`` reads of a chain golden, for ``read_rows`` / ``read_sampled`` patched to serve ``data``."""

    golden_id = "stub"

    def __init__(self, data: dict, sampled: dict):
        self.data, self.sampled, self.rows = data, sampled, data["latent"].shape[0]
        self.manifest = {"products": {"index_key": {"shape": [data["index_key"].shape[1]]}}}


def stub_run(monkeypatch, sampled_rows: list[int], scale: float = 1.0):
    """An 8-row run of the small case's layer (top-3, causal best-first selections), its golden stubbed: the golden's
    stored attention on ``sampled_rows`` is the oracle's times ``scale``. Returns the scorer, ready for chunks."""
    dims, w, _, _, _ = small_case()
    g = torch.Generator().manual_seed(1)
    n, hidden = 8, w["input_layernorm.weight"].numel()
    want = torch.tensor([[r - j if r - j >= 0 else -1 for j in range(dims.topk)] for r in range(n)])
    data = {
        "dsa_input_3": torch.randn(n, hidden, generator=g),
        "latent": torch.randn(n, dims.lora, generator=g, dtype=F64),
        "kpe": torch.randn(n, dims.rope, generator=g, dtype=F64),
        "index_key": torch.randn(n, 2, generator=g, dtype=F64),
        "topk_indices_3": want.int(),
        "topk_scores_3": torch.tensor([[3.0, 2.0, 1.0]]).expand(n, -1).masked_fill(want < 0, math.nan),
    }
    rows = torch.tensor(sampled_rows, dtype=torch.long)
    q = dsa.queries(dsa.normed_input(data["dsa_input_3"][rows], w, dims), w, dims)
    sampled = {
        "sampled_rows": rows,
        "attn_output_3": attend(q, data["latent"], data["kpe"], want[rows], w, dims) * scale,
    }
    golden = StubGolden(data, sampled)
    monkeypatch.setattr(
        k31_chain_golden, "read_rows", lambda gold, names, lo, hi: {k: gold.data[k][lo:hi] for k in names}
    )
    monkeypatch.setattr(k31_chain_golden, "read_sampled", lambda gold, end: gold.sampled)
    return dsa.RunScorer(golden, n, w, dims), data, w, dims


def run_exact(scorer, data, w, dims):
    """Feed every row in chunks of 4 with a device equal to the oracle, then finish."""
    for lo in (0, 4):
        normed = scorer.chunk_input(lo, lo + 4)
        want = data["topk_indices_3"][lo : lo + 4].long()
        out = attend(dsa.queries(normed, w, dims), data["latent"], data["kpe"], want, w, dims)
        scorer.add(lo, out, want)
    return scorer.finish(torch.cat([data["latent"], data["kpe"]], 1), data["index_key"])


def test_run_scorer_checks_its_recompute(monkeypatch, expect_error):
    # Rows 5 and 7 are sampled rows past top-k, in one chunk: scored, and their recompute matches the golden's stored
    # attention row for row.
    windows, summary = run_exact(*stub_run(monkeypatch, [5, 7]))
    assert summary["scored_rows"] == 3  # row 3 (the one evenly spread row of [3, 8)) and the sampled rows 5, 7
    assert summary["recompute_nrmse"] < 1e-12
    assert windows[0]["attn_nrmse"] < 1e-12 and windows[0]["sel_nrmse_median"] == 0 and windows[0]["future_keys"] == 0
    # A stored attention 1e-3 off: the reference is not trusted, nothing is gated.
    with expect_error(AssertionError, "recompute differs"):
        run_exact(*stub_run(monkeypatch, [5, 7], scale=1.001))
    # The only sampled row is below top-k, so never scored: the recompute is unchecked.
    with expect_error(AssertionError, "unchecked"):
        run_exact(*stub_run(monkeypatch, [1]))


def test_run_scorer_refuses_a_chunk_it_did_not_read(monkeypatch, expect_error):
    scorer, data, w, dims = stub_run(monkeypatch, [5])
    scorer.chunk_input(0, 4)
    with expect_error(ValueError, "does not match the last chunk_input"):
        scorer.add(4, torch.zeros(4, w["input_layernorm.weight"].numel()), data["topk_indices_3"][4:].long())
