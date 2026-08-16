# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The roofline describes the stages the MODEL has, not the two an LLM has.

WHAT THE REPORT SHOWED, 2026-08-16. The stage breakdown listed three stages, measured:

    prefill  110.12 ms  ████████████████████
    encode    12.79 ms  ██
    decode     6.11 ms  █

The Roofline section under it listed two. The audio encoder -- twice the cost of decode, and the
second largest stage in the model -- was measured, printed once, and then absent from the analysis
with nothing to say it was missing. A reader comparing the two tables is looking at 116 of 129 ms
and cannot tell.

The cause was a literal:

    stages = [("decode", 1)]
    if prefill_tokens: stages.insert(0, ("prefill", ...))

which is wrong in both directions. A model with a THIRD stage could not have it priced; and a model
with NO decode -- a classifier, a vision tower, a vocoder, a diffusion denoiser -- was still handed a
DECODE row it does not have. perf_target has never been LLM-only: it reasons explicitly about
MULTI-TOWER ("a token reads the language backbone, not the audio encoder"), CONV-HEAVY weights reused
across spatial positions, and diffusion priced per denoise step. Only this table assumed two stages.

stage_ms is written from the model's own PIPELINE_STAGES by the run that measured them, so it is the
authority on what exists, and its order is the order the pipeline runs in.

A STAGE WITHOUT A KNOWN READ SET GETS ITS ROW AND NO ROOF. active_bytes prices an autoregressive
step: a separate tower streams its own weights and none of the backbone's, so borrowing that divisor
would be wrong rather than approximate. The honest output is the measurement plus a missing ceiling,
which is visible, instead of a missing stage, which is not.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

_BW = 512.0
_BYTES = 1718081696


def _stress():
    """The stress file's _render builds a full report; import it by path so this works whether or not
    `tests` is an importable package in the current rootdir."""
    import importlib.util as _u

    spec = _u.spec_from_file_location("_rt_stress", Path(__file__).with_name("test_roofline_table_stress.py"))
    mod = _u.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _roofs(stage_ms, unit="tok/s/u", profile=None):
    from cc_optimize.summary import _stage_roofs

    return _stage_roofs(_BYTES, _BW, 1, unit, profile, stage_ms)


def test_a_third_stage_the_model_declares_is_priced():
    """THE BUG: encode measured, encode absent from the roofline."""
    assert list(_roofs({"encode": 12.79, "prefill": 110.12, "decode": 6.11})) == ["encode", "prefill", "decode"]


def test_declared_order_is_kept():
    """It is the order the pipeline runs in, and the reader follows the model down the table."""
    assert list(_roofs({"prefill": 1.0, "encode": 2.0, "decode": 3.0})) == ["prefill", "encode", "decode"]


def test_a_model_with_no_decode_is_not_given_one():
    """A vision tower or a classifier has no autoregressive step. The old literal invented one."""
    got = list(_roofs({"vision": 40.0, "classify": 2.0}, unit="img/s"))
    assert got == ["vision", "classify"]
    assert "decode" not in got


def test_nothing_declared_falls_back_to_the_recurring_unit():
    """Unchanged behaviour when the model never reported its stages: every model has one."""
    assert list(_roofs(None)) == ["prefill", "decode"]


def test_a_stage_with_no_known_read_set_gets_a_row_but_no_roof():
    """Borrowing the backbone's byte count for a separate tower would be wrong, not approximate."""
    r = _roofs({"encode": 12.79, "decode": 6.11})
    assert r["encode"]["memory_ms"] is None
    assert r["encode"]["bytes"] == 0
    assert r["decode"]["memory_ms"] is not None


def test_a_stage_with_measured_bytes_does_get_a_roof():
    """When the profile records what that stage read, the same formula applies to it."""
    prof = {"buckets": [{"id": "matmul", "stage": "encode", "bytes": 512_000_000}]}
    r = _roofs({"encode": 12.79, "decode": 6.11}, profile=prof)
    assert r["encode"]["bytes"] == 512_000_000
    assert abs(r["encode"]["memory_ms"] - 1.0) < 1e-6  # 0.512 GB / 512 GB/s = 1.0 ms


def test_the_backbone_stages_are_unchanged():
    """prefill and decode keep their own byte model -- this adds stages, it does not re-price them."""
    r = _roofs({"prefill": 110.12, "decode": 6.11})
    assert r["decode"]["bytes"] == _BYTES
    assert r["decode"]["memory_ms"] is not None
    assert r["prefill"]["memory_ms"] is not None


def test_the_stage_list_is_no_longer_a_literal():
    src = (_PA / "cc_optimize" / "summary.py").read_text()
    i = src.index("def _stage_roofs(")
    body = src[i : src.index("\ndef ", i + 1)]
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    assert "_declared" in code, "the stage list is not built from what the model declared"
    j = code.index("_declared")
    assert 'stages = [("decode", 1)]' in code[j:], "the literal is no longer the fallback-only path"


def test_every_stage_gets_a_unit_and_a_title():
    """A declared stage with no unit rendered as a blank row -- present in the data, invisible."""
    src = (_PA / "cc_optimize" / "summary.py").read_text()
    assert "_STAGE_UNIT.setdefault" in src
    assert "_STAGE_TITLE.setdefault" in src


# ------------------------------------------------------------------ what the numbers are PER


def test_the_batch_is_stated_above_the_table(monkeypatch):
    """Every figure in the table is per unit, and batch decides how many units a step retires -- so
    the same measurement reads eight ways on an eight-user run. Voxtral serves 8 and the table said
    nothing, leaving no way to tell a per-user figure from an aggregate one."""
    T = _stress()  # its _render already builds a full report

    monkeypatch.setenv("TT_PERF_BATCH", "8")
    line = next(l for l in T._render(stage_ms={"prefill": 30.0}).splitlines() if "batch:" in l)
    assert "8" in line and "PER unit" in line, line


def test_an_unresolved_batch_says_so_instead_of_printing_one(monkeypatch):
    """TT_PERF_BATCH carries 0 for "ask the pipeline", so a 1 here would be a guess dressed as a
    fact -- and it is exactly the guess that priced an 8-user step as a 1-user step."""
    T = _stress()

    monkeypatch.setenv("TT_PERF_BATCH", "0")
    monkeypatch.delenv("PERF_MCP_BATCH", raising=False)
    monkeypatch.delenv("TT_PERF_BATCH_SIZE", raising=False)
    line = next(l for l in T._render(stage_ms={"prefill": 30.0}).splitlines() if "batch:" in l)
    assert "not reported" in line, line
    assert "batch: 1" not in line, "a sentinel was rendered as a measured batch of 1"


def test_an_unpriced_stage_still_shows_its_measurement():
    """A declared stage with no modelled read set has a real time. Blanking that column made the row
    look like a failed measurement rather than an unpriced one."""
    T = _stress()

    out = T._render(stage_ms={"encode": 35.80, "generate": 138.49})
    row = next(l for l in out.splitlines() if "memory" in l and "35.80" in l)
    assert "not modelled" in row, row


# ------------------------------------------------------------------ batch moves the THEORETICAL too


_MF = {
    "total_params": 3611483136,
    "dominant_dtype": "bfloat16",
    "layers": 30,
    "kv_heads": 8,
    "head_dim": 128,
    "hidden_size": 3072,
    "intermediate_size": 8192,
}


def _roofs_at_batch(monkeypatch, batch):
    """Patch the facts and the prompt length directly, as the stress suite does.

    Not via the environment: this file imports the stress module to reach its _render, and executing
    it leaves module attributes patched, so an env-driven variant passed alone and failed in file
    order -- the least useful kind of test.
    """
    import cc_optimize.summary as S

    monkeypatch.setattr(S, "_model_facts", lambda: _MF)
    monkeypatch.setattr(S, "_prefill_tokens", lambda: 128)
    monkeypatch.setattr(S, "_prefill_batch", lambda: batch)
    return S._stage_roofs(_BYTES, _BW, 1, "tok/s/u", None, {"prefill": 110.1, "decode": 6.11})


def test_batch_raises_the_prefill_ceiling(monkeypatch):
    """Eight users in flight carry eight sets of activations and eight KV writes. Costing that as one
    user made batch free, so the ceiling came out too high and every at-floor verdict inherited it."""
    one = _roofs_at_batch(monkeypatch, 1)["prefill"]["bytes"]
    eight = _roofs_at_batch(monkeypatch, 8)["prefill"]["bytes"]
    assert eight > one * 1.5, (one, eight)


def test_batch_raises_the_decode_ceiling_too(monkeypatch):
    """Decode used to return the anchor untouched -- weights only, no KV -- so it had no per-user term
    for batch to scale. Every user re-reads their whole history on every token."""
    one = _roofs_at_batch(monkeypatch, 1)["decode"]["bytes"]
    eight = _roofs_at_batch(monkeypatch, 8)["decode"]["bytes"]
    assert eight > one, (one, eight)


def test_decode_still_carries_the_agreed_weights(monkeypatch):
    """The addition is a DIFFERENCE against the anchor, never a second opinion on the weights -- that
    is what kept decode excluded in the first place (two ceilings 2.18x apart)."""
    assert _roofs_at_batch(monkeypatch, 8)["decode"]["bytes"] >= _BYTES


def test_the_theoretical_moves_not_just_the_measured(monkeypatch):
    """A comparison is only correct if BOTH sides describe the same workload."""
    one = _roofs_at_batch(monkeypatch, 1)
    eight = _roofs_at_batch(monkeypatch, 8)
    assert eight["prefill"]["memory_ms"] > one["prefill"]["memory_ms"]
    assert eight["decode"]["memory_ms"] > one["decode"]["memory_ms"]
