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
    "has_unit_ceiling": True,
    "theoretical_rate": 51.2,  # (512 * 0.80) / 8 GB -- sustained, not spec
    "band": [38.4, 51.2],
    "active_bytes": 8_000_000_000,
    "peak_bw_gbps": 512.0,
    "bw_fraction": 0.80,
    "tp_degree": 1,
    "modeled_floor_ms": 15.6,
}


def test_llm_decode_form_matches_bandwidth_math():
    out = "\n".join(S._roofline_lines(_LLM, 19.4))
    assert "ceiling (sustained) : 51.2 tok/s/u" in out
    assert "38.4 - 51.2 tok/s/u" in out
    assert "measured            : 51.5 tok/s/u" in out  # 1000 / 19.4
    assert "412 GB/s" in out  # 8 GB / 19.4 ms
    assert "%" in out and "utilization" in out


def test_module_floor_form_when_not_llm():
    out = "\n".join(S._roofline_lines({"has_unit_ceiling": False, "scope": "module", "modeled_floor_ms": 8.90}, 11.82))
    assert "modeled floor       : 8.90 ms" in out
    assert "measured            : 11.82 ms" in out
    assert "at-floor            : 75%" in out  # 8.90 / 11.82
    # No tok/s figure in the ms form. The reason line used to assert "not an LLM decode pipeline"
    # unconditionally, which was false for Llama-3.1-8B; with no active_bytes it now says the
    # numerator is missing instead of inventing a property of the model.
    assert "rate ceiling" in out and "tok/s/u   (1000 /" not in out
    assert "no weight-bytes input" in out


def test_floor_form_shows_the_achievable_band_not_just_the_floor():
    """The floor is unreachable BY CONSTRUCTION (full-grid compute term, DRAM fallback for L1), and it
    is NOT a bandwidth ceiling -- so it publishes no 60-80% band. Deriving one from 1000/floor made a
    range the hardware has no peak behind, which the report showed as "achievable" and the stop gate
    treated as done. What the operator gets instead is at-floor% with no fabricated goal."""
    out = "\n".join(S._roofline_lines({"has_unit_ceiling": False, "modeled_floor_ms": 8.90}, 11.82))
    assert "achievable (60-80%)" not in out
    assert "status              : NO_BAND" in out
    assert "at-floor            : 75%" in out  # 8.90 / 11.82


def test_floor_form_below_band_keeps_optimizing():
    out = "\n".join(S._roofline_lines({"has_unit_ceiling": False, "modeled_floor_ms": 8.90}, 40.0))
    assert "status              : NO_BAND" in out and "keep optimizing" in out
    assert "at-floor            : 22%" in out  # 8.90 / 40.0


def test_stale_guard_renders_na_not_zero():
    # invalid forward ms must NOT produce a fake 0.0 tok/s/u / 0% — it renders n/a
    out = S._roofline_lines(_LLM, 0.0)
    joined = "\n".join(out)
    assert "measured            : n/a" in joined
    assert "utilization         : n/a" in joined
    # the ONLY place a "0" appears is the static band label; no fabricated measured/util zero
    assert "0.0 tok/s/u   (1000" not in joined
    assert "0%   (measured / ceiling)" not in joined


def test_floor_form_flags_stale_when_measured_below_floor():
    # measured faster than the modeled floor => floor is from a different (stale) profile; the table
    # must say so, NOT print a bogus >100% at-floor. (Caught live on ace_step_audio_tokenizer: a
    # baseline-profile floor of 14.81 ms paired with the optimized 11.82 ms.)
    out = "\n".join(S._roofline_lines({"has_unit_ceiling": False, "modeled_floor_ms": 14.81}, 11.82))
    assert "stale/suspect" in out
    assert "125%" not in out and "at-floor            : 1" not in out  # no >100% number
    # It must NOT assert WHICH side is stale. On llama3_1_8b_p150 the FLOOR was the fresh number
    # (16-layer profile) and the MEASUREMENT was the leftover (2-layer window), i.e. the opposite of
    # the ace_step case above; the verdict comes from perf_target.score, which only knows the pair
    # is inconsistent.
    assert "ABOVE_BAND" in out
    assert "floor stale" not in out


def test_llm_form_flags_stale_when_measured_exceeds_ceiling():
    out = "\n".join(
        S._roofline_lines(
            {
                "has_unit_ceiling": True,
                "theoretical_rate": 40.0,
                "band": [24.0, 32.0],
                "active_bytes": 8_000_000_000,
                "tp_degree": 1,
            },
            19.4,
        )
    )
    # 1000/19.4 = 51.5 tok/s > 40 ceiling => flag, not "129%"
    assert "EXCEEDS ceiling" in out and "129%" not in out


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
