# SPDX-License-Identifier: Apache-2.0
"""RUN_REPORT 'Roofline & utilization' table: an adaptive per-optimize block (tok/s/u form for LLM
decode, roofline-floor ms form otherwise). MEASURED values are computed from the ms being reported,
against a STATIC target snapshot — so nothing stale leaks in and missing inputs render 'n/a', never a
fabricated 0.0 (the fix for the old '+0.0%'-style readout)."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "cc_summary", str(Path(__file__).resolve().parents[1] / "cc_optimize" / "summary.py")
)
S = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(S)

_LLM = {
    "is_llm_decode": True,
    "theoretical_tok_s": 64.0,
    "band": [38.4, 51.2],
    "active_bytes": 8_000_000_000,
    "tp_degree": 1,
    "modeled_floor_ms": 15.6,
}


def test_llm_decode_form_matches_bandwidth_math():
    out = "\n".join(S._roofline_lines(_LLM, 19.4))
    assert "theoretical ceiling : 64.0 tok/s/u" in out
    assert "38.4 - 51.2 tok/s/u" in out
    assert "measured            : 51.5 tok/s/u" in out  # 1000 / 19.4
    assert "412 GB/s" in out  # 8 GB / 19.4 ms
    assert "%" in out and "utilization" in out


def test_module_floor_form_when_not_llm():
    out = "\n".join(S._roofline_lines({"is_llm_decode": False, "scope": "module", "modeled_floor_ms": 8.90}, 11.82))
    assert "modeled floor       : 8.90 ms" in out
    assert "measured            : 11.82 ms" in out
    assert "at-floor            : 75%" in out  # 8.90 / 11.82
    assert "tok/s/u — N/A" in out


def test_stale_guard_renders_na_not_zero():
    # invalid forward ms must NOT produce a fake 0.0 tok/s/u / 0% — it renders n/a
    out = S._roofline_lines(_LLM, 0.0)
    joined = "\n".join(out)
    assert "measured            : n/a" in joined
    assert "utilization         : n/a" in joined
    # the ONLY place a "0" appears is the static band label; no fabricated measured/util zero
    assert "0.0 tok/s/u   (1000" not in joined
    assert "0%   (measured / ceiling)" not in joined


def test_none_throughput_skips_table():
    assert S._roofline_lines(None, 19.4) == []  # no snapshot -> no table at all
    # an empty dict is a (degenerate) snapshot -> floor form with n/a floor, never crashes
    out = "\n".join(S._roofline_lines({}, 19.4))
    assert out.startswith("Roofline & utilization") and "modeled floor       : n/a" in out


def test_render_summary_accepts_throughput_kwarg(tmp_path):
    # the new kwarg must be optional and not break the existing summary render
    log = tmp_path / "k.json"
    log.write_text("[]")
    txt = S.render_summary(str(log), 19.4, model="m", task="main", throughput=_LLM, final_override_ms=19.4)
    assert "Roofline & utilization" in txt and "51.5 tok/s/u" in txt
    # and with no throughput it still renders (table just absent)
    txt2 = S.render_summary(str(log), 19.4, model="m", task="main")
    assert "Roofline & utilization" not in txt2
