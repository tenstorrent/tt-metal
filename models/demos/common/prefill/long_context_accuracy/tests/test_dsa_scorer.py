# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``DsaScorer`` (``dsa/scorer.py``): scored rows, causality, gain, the per-chunk contracts, and a whole run on a stub
golden: metric tiers, fail-closed bars and the recompute check. CPU only, tiny dimensions."""

from __future__ import annotations

import math

import pytest
import torch

from models.demos.common.prefill.long_context_accuracy.acceptance_bars import AcceptanceBar, check
from models.demos.common.prefill.long_context_accuracy.block import METRIC_TIERS, BlockOutputs, ChunkScore
from models.demos.common.prefill.long_context_accuracy.dsa import fp64_attention as fa
from models.demos.common.prefill.long_context_accuracy.dsa import scorer as ds
from models.demos.common.prefill.long_context_accuracy.tests.test_dsa_fp64_attention import attend, small_case

F64 = torch.float64
NAMES = ds.GoldenNames(
    input="x",
    latent="latent",
    kpe="kpe",
    index_key="index_key",
    selection="topk",
    selection_scores="topk_scores",
    sampled_rows="sampled_rows",
    sampled_output="attn",
)


def test_scored_rows() -> None:
    # one partial window [2048, 5120): round(4 * 3072 / 16384) = 1 row (its first), plus the sampled row 4088
    rows = ds.scored_rows(5120, torch.tensor([0, 2044, 4088]), 2048, per_window=4)
    assert rows.tolist() == [2048, 4088]
    # a full window past top-k: 4 rows evenly spread over [16384, 32768), and none below top-k
    rows = ds.scored_rows(32768, torch.tensor([], dtype=torch.long), 2048, per_window=4)
    assert rows.tolist()[-4:] == [16384, 21845, 27306, 32767] and rows.min() >= 2048


def test_future_keys() -> None:
    # rows 2 and 3: row 2 selects key 5 (future), row 3 selects keys 0..3 (none in its future)
    assert ds.future_keys(torch.tensor([[0, 1, 5], [0, 3, -1]]), 2).tolist() == [1, 0]


class StubGolden:
    """What ``DsaScorer`` reads of a golden when built: its id, rows, the index key's shape and the sampled rows."""

    golden_id = "stub"

    def __init__(self, rows: int, index_dim: int, sampled: dict):
        self.rows, self.sampled = rows, sampled
        self.manifest = {"products": {"index_key": {"shape": [index_dim]}}}

    def read_sampled(self, end):
        return self.sampled


def stub_run(sampled_rows: list[int], scale: float = 1.0):
    """An 8-row run of the small layer (top-3, causal best-first selections): its golden rows, and a scorer whose stub
    golden stores, on ``sampled_rows``, the oracle's attention times ``scale``."""
    dims, w, _, _, _ = small_case()
    g = torch.Generator().manual_seed(1)
    n, hidden = 8, w["input_layernorm.weight"].numel()
    want = torch.tensor([[r - j if r - j >= 0 else -1 for j in range(dims.topk)] for r in range(n)])
    data = {
        "x": torch.randn(n, hidden, generator=g),
        "latent": torch.randn(n, dims.lora, generator=g, dtype=F64),
        "kpe": torch.randn(n, dims.rope, generator=g, dtype=F64),
        "index_key": torch.randn(n, 2, generator=g, dtype=F64),
        "topk": want.int(),
        "topk_scores": torch.tensor([[3.0, 2.0, 1.0]]).expand(n, -1).masked_fill(want < 0, math.nan),
    }
    rows = torch.tensor(sampled_rows, dtype=torch.long)
    q = fa.queries(fa.normed_input(data["x"][rows], w, dims), w, dims)
    sampled = {"sampled_rows": rows, "attn": attend(q, data["latent"], data["kpe"], want[rows], w, dims) * scale}
    scorer = ds.DsaScorer(StubGolden(n, 2, sampled), n, NAMES, w, dims)
    return scorer, data, w, dims


def exact_outputs(scorer, data, w, dims, lo: int, hi: int) -> BlockOutputs:
    """A device equal to the oracle on rows ``[lo, hi)``."""
    want = data["topk"][lo:hi].long()
    q = fa.queries(fa.normed_input(data["x"][lo:hi], w, dims), w, dims)
    return BlockOutputs(attend(q, data["latent"], data["kpe"], want, w, dims), {"topk": want})


def run_exact(scorer, data, w, dims):
    """Feed every row in chunks of 4 with a device equal to the oracle, check each chunk, then score the run."""
    for lo in (0, 4):
        rows = {k: v[lo : lo + 4] for k, v in data.items()}
        outputs = exact_outputs(scorer, data, w, dims, lo, lo + 4)
        assert scorer.score_chunk(lo, outputs, rows) == ChunkScore()
    carried = {"kv_cache": torch.cat([data["latent"], data["kpe"]], 1), "index_cache": data["index_key"]}
    return scorer.score_run(carried)


def test_device_input_is_the_normed_input_rounded_once() -> None:
    scorer, data, w, dims = stub_run([5])
    x = scorer.device_input({"x": data["x"][:4]})
    assert x.dtype == torch.bfloat16
    assert torch.equal(x, fa.normed_input(data["x"][:4], w, dims).to(torch.bfloat16))


def test_exact_run_tiers() -> None:
    # Rows 5 and 7 are sampled rows past top-k, in one chunk: scored, and their recompute matches the golden's stored
    # attention row for row.
    tiers = run_exact(*stub_run([5, 7]))
    assert set(tiers) == set(METRIC_TIERS)
    names = [{k for m in tiers[t] for k in m} - {"start", "end", "rows"} for t in METRIC_TIERS]
    assert not (names[0] & names[1] or names[0] & names[2] or names[1] & names[2]), names
    assert all({"start", "end", "rows"} <= set(m) for t in METRIC_TIERS for m in tiers[t])
    (headline,) = tiers[0]
    assert (headline["start"], headline["end"]) == (0, 8)
    assert headline["run_attn_nrmse"] < 1e-12 and headline["run_sel_nrmse"] == 0
    (window,) = tiers[1]
    assert window["scored_rows"] == 3  # row 3 (the one evenly spread row of [3, 8)) and the sampled rows 5, 7
    assert window["attn_nrmse"] < 1e-12 and window["sel_nrmse_median"] == 0 and window["future_keys"] == 0
    assert window.keys() - {"start", "end", "rows"} == ds.WINDOW_TIER_1
    window_2, run_2 = tiers[2]
    assert window_2["missed_key_max_max"] == 0 and window_2["kv_latent_nrmse"] == 0 and window_2["index_key_nrmse"] == 0
    assert run_2["run_scored_rows"] == 3 and run_2["recompute_nrmse"] < 1e-12
    assert run_2["attn_gain"] == pytest.approx(1.0) and run_2["nrmse_floor"] > 0


def test_run_bars_fail_closed() -> None:
    bars = [AcceptanceBar(m, "max", 1e-9) for m in ("sel_nrmse_median", "attn_nrmse", "attn_rms_err")]
    tiers = run_exact(*stub_run([5, 7]))
    measurements = [m for t in METRIC_TIERS for m in tiers[t]]
    assert check(measurements, bars) == []
    # A selection that misses keys on the scored rows fails the selection bar, even with the device output exact.
    scorer, data, w, dims = stub_run([5, 7])
    for lo in (0, 4):
        rows = {k: v[lo : lo + 4] for k, v in data.items()}
        outputs = exact_outputs(scorer, data, w, dims, lo, lo + 4)
        got = outputs.extras["topk"].clone()
        r = torch.arange(lo, lo + 4)
        got[r >= 3, -1] = r[r >= 3] - 3  # rows r >= 3 select r - 3 for their lowest-ranked key r - 2 (still causal)
        scorer.score_chunk(lo, BlockOutputs(outputs.output, {"topk": got}), rows)
    carried = {"kv_cache": torch.cat([data["latent"], data["kpe"]], 1), "index_cache": data["index_key"]}
    tiers = scorer.score_run(carried)
    failures = check([m for t in METRIC_TIERS for m in tiers[t]], bars)
    assert any(f.startswith("sel_nrmse_median") for f in failures), failures


def test_score_chunk_reports_contract_breaks() -> None:
    scorer, data, w, dims = stub_run([5])
    scorer.score_chunk(0, exact_outputs(scorer, data, w, dims, 0, 4), {k: v[0:4] for k, v in data.items()})
    rows = {k: v[4:8] for k, v in data.items()}
    outputs = exact_outputs(scorer, data, w, dims, 4, 8)
    broken = outputs.output.clone()
    broken[1, 0] = math.nan
    future = outputs.extras["topk"].clone()
    future[2, 0] = 7  # row 6 selects key 7
    violations = scorer.score_chunk(4, BlockOutputs(broken, {"topk": future}), rows).violations
    assert violations == ["non-finite device attention output", "1 selected keys lie after their row (first on row 6)"]


def test_recompute_is_checked(expect_error) -> None:
    # A stored attention 1e-3 off: the reference is not trusted, nothing is scored.
    with expect_error(AssertionError, "recompute differs"):
        run_exact(*stub_run([5, 7], scale=1.001))
    # The only sampled row is below top-k, so never scored: the recompute is unchecked.
    with expect_error(AssertionError, "unchecked"):
        run_exact(*stub_run([1]))


def test_score_chunk_refuses_mismatched_rows(expect_error) -> None:
    scorer, data, w, dims = stub_run([5])
    rows = {k: v[0:4] for k, v in data.items()}
    outputs = exact_outputs(scorer, data, w, dims, 0, 3)
    with expect_error(ValueError, "differ from the golden's 4"):
        scorer.score_chunk(0, outputs, rows)
