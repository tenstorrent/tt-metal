"""Stress test: RUN_REPORT Roofline & utilization and Block-level timing render DETERMINISTICALLY —
never gated on the throughput temp file existing or the agent passing stages_json. Self-contained
(synthetic profiles), so it holds in a fresh checkout / CI with no runtime artifacts.
"""

from __future__ import annotations

import re

import importlib.util
import json
import sys
from pathlib import Path


def _flat(text):
    """A column-width-agnostic view of the table.

    The roofline pads a number and its unit into fixed sub-fields, so a published figure reads
    "64.0      tok/s/u". These assertions are about the PAIRING -- that a value is published carrying
    its unit -- not about the geometry, and pinning the geometry is how a column-width change becomes
    a test failure with nothing wrong behind it. Collapsing runs of spaces keeps the claim and drops
    the layout.
    """
    return re.sub(r"[ \t]+", " ", str(text))


_PA = Path(__file__).resolve().parents[1]
if str(_PA) not in sys.path:
    sys.path.insert(0, str(_PA))
_CC = _PA / "cc_optimize"

_PROF = {
    "buckets": [
        {"id": "datamove", "device_ms": 70.0},
        {"id": "matmul", "device_ms": 50.0},
        {"id": "reduction", "device_ms": 14.0},
    ]
}


def _summary():
    spec = importlib.util.spec_from_file_location("cc_summary_test", str(_CC / "summary.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tp(floor_ms):
    theo = 1000.0 / floor_ms
    return {
        "scope": "model",
        "has_unit_ceiling": False,
        "theoretical_rate": theo,
        "band": [0.6 * theo, 0.8 * theo],
        "active_bytes": 0,
        "peak_bw_gbps": 0.0,
        "tp_degree": 1,
        "modeled_floor_ms": floor_ms,
    }


def _render(sm, tmp_path, *, baseline_profile=None, throughput=None, final_ms=100.0, attempts=None):
    kl = tmp_path / "kl.json"
    kl.write_text(json.dumps(attempts or []))
    return sm.render_summary(
        str(kl),
        final_ms,
        model="s",
        task="main",
        metric="device_ms",
        baseline_profile=baseline_profile,
        throughput=throughput,
        final_override_ms=final_ms,
        finalized=True,
    )


def test_stages_from_profile_direct():
    sm = _summary()
    rows = sm._stages_from_profile(_PROF)
    assert rows and rows[0]["name"] == "datamove" and rows[0].get("dominant")


def test_block_level_renders_from_profile_without_stages(tmp_path):
    sm = _summary()
    text = _render(sm, tmp_path, baseline_profile=_PROF, throughput=_tp(100.0), attempts=[])
    assert "Block-level timing (per-stage trace)" in text
    assert "datamove" in text


def test_roofline_renders_when_throughput_none(tmp_path, monkeypatch):
    sm = _summary()
    monkeypatch.setattr(sm, "_throughput_from_profile", lambda bp: _tp(100.0))
    text = _render(sm, tmp_path, baseline_profile=_PROF, throughput=None, final_ms=200.0, attempts=[])
    assert "Roofline" in text


_AGENT_STAGES = [
    {
        "op_signature": "MatmulDeviceOperation",
        "kernel_kind": "dtype",
        "measured_ms": 10.0,
        "beat_baseline": True,
        "stages": [{"name": "matmul", "ms": 9.0, "dominant": True}],
    }
]


def test_the_profile_outranks_the_agents_prose(tmp_path):
    """MEASURED FIRST. This preferred the agent's stages_json -- free text, unvalidated, frozen at
    the moment it was captured -- and consulted the profile's own op-class buckets only when no
    attempt happened to carry any. So the one source with a device measurement behind it was the
    LAST resort: the table carried "QKV, still HiFi2 -- same lever untried" long after that lever
    was tried and won, and summed to 529.43 ms while the op breakdown directly above it summed to
    556.80 -- two totals for one profile, in adjacent sections."""
    sm = _summary()
    text = _render(sm, tmp_path, baseline_profile=_PROF, attempts=_AGENT_STAGES)
    assert "op-class breakdown (same profile as the table above)" in text
    assert "latest lever on Matmul" not in text
    assert "annotation, not measurement" not in text, "measured rows must not be labelled annotation"


def test_the_agents_prose_is_still_shown_when_nothing_measured_it(tmp_path):
    """It remains the more useful view for finding hot spots -- it just stops OUTRANKING a
    measurement, and it says whose words it is."""
    sm = _summary()
    text = _render(sm, tmp_path, baseline_profile={"buckets": []}, attempts=_AGENT_STAGES)
    assert "Block-level timing (per-stage trace) — latest lever on Matmul" in text
    assert "annotation, not measurement" in text


def test_a_floor_target_publishes_no_achievable_band(tmp_path):
    """60-80% is a DRAM-BANDWIDTH statement. 60-80% of 1000/floor is not, and printing it put
    "achievable 671.54 - 895.38 ms" beside a 534 ms measurement -- and the optimize stop gate read
    that same band, so a run could be declared done against a range never derived from the hardware.
    """
    sm = _summary()
    text = _render(sm, tmp_path, baseline_profile=_PROF, throughput=_tp(100.0), final_ms=200.0)
    assert "achievable (60-80%)" not in text
    assert "NO_BAND" in text and "at-floor" in text


def test_a_bandwidth_ceiling_does_publish_the_band(tmp_path):
    sm = _summary()
    llm = {
        "scope": "model",
        "has_unit_ceiling": True,
        "theoretical_rate": 64.0,  # 512 / 8 GB
        "band": [38.4, 51.2],
        "active_bytes": int(8e9),
        "peak_bw_gbps": 512.0,
        "bw_fraction": 0.80,
        "tp_degree": 1,
        "perf_layers": "all",
    }
    text = _render(sm, tmp_path, baseline_profile={"per_token_ms": 19.4}, throughput=llm, final_ms=19.4)
    assert "ACHIEVABLE 60-80%" in text, text
    assert "64.0 tok/s/u" in _flat(text), text  # now a THEORETICAL column, not a labelled line


def test_status_has_no_band_verdict_for_a_floor_target(tmp_path):
    sm = _summary()
    text = _render(sm, tmp_path, baseline_profile=_PROF, throughput=_tp(100.0), final_ms=150.0)
    assert "NO_BAND" in text


def test_status_above_band(tmp_path):
    sm = _summary()
    text = _render(sm, tmp_path, baseline_profile=_PROF, throughput=_tp(100.0), final_ms=90.0)
    assert "ABOVE_BAND" in text


def test_stress_many_profiles(tmp_path):
    sm = _summary()
    for i in range(60):
        prof = {"buckets": [{"id": f"op{j}", "device_ms": (j + 1) * (i % 5 + 1) * 1.0} for j in range(1 + i % 6)]}
        text = _render(sm, tmp_path, baseline_profile=prof, throughput=_tp(50.0 + i), final_ms=100.0 + i, attempts=[])
        assert "Roofline" in text, f"iter {i}: roofline missing"
        assert "Block-level timing (per-stage trace)" in text, f"iter {i}: block-level missing"


def test_block_level_timing_is_rendered_exactly_once():
    """A rebase conflict in summary.py was resolved by keeping BOTH sides, leaving two renders of the
    same section: the report printed the whole Block-level timing table twice, byte-identical. The
    second block already handled the case the first did (plus the no-attempt fallback), so the first
    was pure duplication."""
    import importlib.util
    import json
    import sys
    import tempfile
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("summary_dup_ut", root / "cc_optimize" / "summary.py")
    sm = importlib.util.module_from_spec(spec)
    sys.modules["summary_dup_ut"] = sm
    spec.loader.exec_module(sm)

    kl = Path(tempfile.mktemp(suffix=".json"))
    kl.write_text(
        json.dumps(
            [
                {
                    "op_signature": "MatmulDeviceOperation 128 x 4096 x 14336",
                    "kernel_kind": "dtype",
                    "measured_ms": 648.0,
                    "beat_baseline": True,
                    "stages": [
                        {"name": "prefill ff1/ff3", "ms": 141.85, "dominant": True},
                        {"name": "decode ff1/ff3", "ms": 92.47},
                    ],
                }
            ]
        )
    )
    out = sm.render_summary(kl, model="m", task="main", finalized=True)
    assert out.count("Block-level timing (per-stage trace)") == 1, out.count("Block-level timing (per-stage trace)")
    assert out.count("prefill ff1/ff3") == 1


def test_the_snapshot_carries_every_key_the_report_reads():
    """PRODUCER/CONSUMER KEY PARITY. The roofline section reads its inputs out of the throughput
    snapshot by name, so a key the writer omits silently becomes the reader's default -- and defaults
    are where the model-specific bugs live. `unit` was missing exactly this way: the report fell back
    to "token" for every model, printing a diffusion model's steps/s ceiling as "tok/s/u" and reading
    the byte anchor under the wrong depth, while unit-passing tests kept passing.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    writer = re.search(
        r"def _persist_throughput.*?\n(?=def )", (root / "cc_optimize" / "perf_mcp.py").read_text(), re.S
    ).group(0)
    written = set(re.findall(r'"([a-z_]+)":', writer))

    rl = re.search(r"def _roofline_lines.*?\n(?=def )", (root / "cc_optimize" / "summary.py").read_text(), re.S).group(
        0
    )
    read = set(re.findall(r'throughput(?:\s*or\s*\{\})?\.get\("([a-z_]+)"', rl)) | set(
        re.findall(r'\(throughput or \{\}\)\.get\("([a-z_]+)"', rl)
    )

    # keys the reader accepts only for backward compatibility are allowed to be absent
    legacy = {"theoretical_tok_s", "is_llm_decode"}
    missing = sorted((read - written) - legacy)
    assert not missing, "the report reads keys the snapshot never writes (they silently default): %s" % missing
