# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ONE owner per reported fact -- enforced, not remembered.

Every headline defect this tool has shipped had the same shape: a fact was re-derived at a second
site, the first site got fixed, and the second kept lying.

    "is this a win"      3 renderers in summary.py + the KV-cache GATE in perf_mcp, each deciding
                         for itself. The gate's docstring promised "clears ONLY on a MEASURED
                         reduction" while its code checked the flag alone; the report showed a ✓ in
                         one section and "no gain" in the other for the SAME attempt.

    "the modeled floor"  the throughput snapshot pinned it AND the renderer pinned it, so there were
                         two answers to "what is the target" -- and the snapshot's copy could not be
                         corrected mid-run, because the MCP server loads perf_mcp once at startup.

    "the baseline ms"    a fallback chain of four files; the report took whichever existed, which is
                         how another model's 0.06 ms became an anchor.

Reviewing for the next such site does not scale -- it was missed three times in one file. These tests
fail when a NEW site appears, which is the only version of this that holds.

MODEL-AGNOSTIC BY CONSTRUCTION: the owners key on (model, task) and derive every value from the
profile the model produced. No rule here mentions a model, and the last test proves two models get
independent anchors with no per-model code.
"""
from __future__ import annotations

import re

import ast
import importlib.util
import sys
from pathlib import Path

import pytest


def _flat(text):
    """A column-width-agnostic view of the table.

    The roofline pads a number and its unit into fixed sub-fields, so a published figure reads
    "64.0      tok/s/u". These assertions are about the PAIRING -- that a value is published carrying
    its unit -- not about the geometry, and pinning the geometry is how a column-width change becomes
    a test failure with nothing wrong behind it. Collapsing runs of spaces keeps the claim and drops
    the layout.
    """
    return re.sub(r"[ \t]+", " ", str(text))


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# fact -> (dict-key it is read from, functions ALLOWED to read it and why)
_OWNERS = {
    "is a win": (
        "beat_baseline",
        {
            "cc_optimize/measurements.py": {"is_win"},
            # the writer stamps the flag; it must agree with the measurement at write time
            "cc_optimize/perf_mcp.py": {"record_kernel_attempt", "_record_committed_win", "_autorecord_wedge"},
            "cc_optimize/summary.py": set(),
        },
    ),
    # THE CEILING'S BYTES. One owner (the ledger anchor, written once per model by
    # run._emit_perf_target_inputs) and one cache (the throughput snapshot). _roofline_lines must
    # prefer the anchor -- the snapshot is rewritten from a file inside the model directory the
    # optimize loop reverts, and a 16-layer 3.33 GB vintage came back twice in one run and printed a
    # 153.8 tok/s/u ceiling beside a full-model measurement.
    "the ceiling bytes": (
        "active_bytes",
        {
            "cc_optimize/perf_mcp.py": {"_persist_throughput"},
            "cc_optimize/summary.py": {"_roofline_lines", "_throughput_from_profile"},
        },
    ),
    # THE UNIT OF WORK. Derived once from the model (model_bytes) and then PASSED along the chain --
    # facts json -> ledger anchor depth -> PerfTarget.unit -> snapshot -> report, plus the separate
    # record of which unit the gate's reading counts. No hop may re-derive it; a hop that defaults
    # instead of reading is how "unit" being absent from the snapshot made every model read as
    # per-token while every unit test passed.
    "the unit of work": (
        "unit",
        {
            "agent/model_bytes.py": {"weight_bytes"},
            "agent/perf_target.py": {"compute_target"},
            "cc_optimize/run.py": {"_emit_perf_target_inputs", "_perf_target_inputs"},
            "cc_optimize/perf_mcp.py": {
                "_persist_throughput",
                "_perf_target_status",
                "_reliable_forward_unit",
                "_record_fullpipe_candidate",
                "_establish_fullpipe_baseline",
                # The byte anchor is KEYED by unit, so reading it requires the unit; and once the
                # loop's revert deletes perf_target_inputs.json, the unit the anchor was pinned under
                # is the only surviving record of it. These two read that record instead of
                # re-deriving the unit from a config, and neither DEFAULTS it -- no recoverable unit
                # means no ceiling, which is the rule this registry exists to enforce.
                "_anchored_ceiling_bytes",
                "_anchored_ceiling_facts",
                # SAME GROUND, SAME RULE. The per-stage stop bar prices each stack against its own
                # roof, and _stage_roofs takes the unit because a stage's compute floor is linear in
                # the tokens one unit retires. It reads the anchor's record through the two names
                # above and DEFAULTS NOTHING -- no recoverable unit means no bar, and the run is not
                # blocked by a ceiling nobody could compute.
                "_stages_short_of_achievable",
                # THE OBSERVED UNIT SUPERSEDES THE FILE, and this is the single point every consumer
                # of the facts passes through. perf_target_inputs.json records what a TABLE keyed on
                # the HF pipeline_tag guessed at setup, before anything ran -- and a tag names the
                # TASK, which cannot state whether a model loops: `text-to-speech` covers XTTS, which
                # emits tokens, and Kokoro, which produces a whole waveform in one pass. Once
                # trace_replay has reported what the built pipeline ACTUALLY did, that is a fact, and
                # a fact outranks the lookup it is correcting. This does NOT re-derive the unit from a
                # config -- it reads the observation and applies it, and when nothing has been
                # observed the file's value stands untouched.
                "_load_perf_target_inputs",
            },
            "cc_optimize/summary.py": {"_roofline_lines"},
        },
    ),
    # THE COMPUTE ROOF'S DENOMINATOR. One producer (perf_mcp, which derives the dominant fidelity and
    # pins it) and one reader (summary._stage_roofs, via _pinned_peak_flops). This is the ceiling
    # input that had no anchor at all: _promote_baseline refreshes the profile PICTURE on every
    # profile_model call while ratcheting only device_ms, so a peak read off that picture doubles the
    # moment the `fidelity` rung lands -- 175.5 -> 351.0 TFLOPS on voxtral, from one matmul.
    #
    # It survived because it never printed anything impossible: the memory roof divides by a fixed
    # 512 GB/s and screams when it drifts, while here the peak IS what moves and the measurement
    # moves with it, so a stage that got 2x faster reports the same percentage.
    "the compute peak": (
        "peak_flops",
        {
            "cc_optimize/perf_mcp.py": {"_dominant_peak_flops", "_persist_throughput"},
            # _stage_roofs reads the anchor through _pinned_peak_flops and keeps the current-picture
            # value beside it as peak_flops_now, which is what separates a real fidelity win from a
            # stale reading -- the same pairing _floor_anchor's caller uses.
            # _roofline_tables only RENDERS the value _stage_roofs already put in the roofs dict --
            # it prints the TFLOPS cell and scales the 60-80% band from it. It derives nothing, and
            # the ratio it prints (measured / peak) is the one number a reader checks the ceiling by.
            "cc_optimize/summary.py": {"_stage_roofs", "_pinned_peak_flops", "_roofline_tables"},
        },
    ),
    # THE BYTES ONE UNIT OF WORK READS. Eleven commits have been about this single quantity, and each
    # fixed the consumer that happened to look wrong: the census total, the anchor's key, the checkpoint
    # inference, the pinning, the per-stage split. None was preceded by "who else reads this?", so each
    # fix left the other consumers on a different source, and the twelfth symptom was a DECODE row
    # quoting 2.350 GB in its floor and 4.784 GB in its bandwidth -- 2.04x apart, three lines apart.
    #
    # A stage's read set now has ONE owner. _stage_roofs derives it (pinned -> measured -> estimate)
    # and puts it in the roof dict; every renderer divides by that, via _measured_bw_gbps. A renderer
    # reaching for the model-level `active_bytes` instead is the mistake this entry exists to catch --
    # legitimate only where the question really is model-level (the DRAM capacity panel asks what the
    # model OCCUPIES, not what a unit reads).
    "the per-stage read set": (
        "bytes",
        {
            "cc_optimize/summary.py": {
                # the owner, and the two helpers that read the value it produced
                "_stage_roofs",
                "_measured_bw_gbps",
                "_bytes_for",
                "_stage_measured_bytes",
                "_measured_stage_bytes",
                "_pinned_stage_bytes",
                "_roofline_tables",
                "_section_bytes_cached",
                "_stage_share",
                "_share_and_basis",
                "_roofline_lines",
                "_throughput_from_profile",
                "_model_facts",
                "_peak_for_stage",
            },
        },
    ),
    "the modeled floor": (
        "modeled_floor_ms",
        {
            # producers of the CURRENT-build number, and the single anchor reader
            "cc_optimize/perf_mcp.py": {
                "profile_model",
                "measure_candidate",
                "_select_perf_target",
                "_persist_throughput",
            },
            "cc_optimize/summary.py": {"_roofline_lines", "_throughput_from_profile"},
        },
    ),
}


def _sm():
    spec = importlib.util.spec_from_file_location("sm_ssot", _ROOT / "cc_optimize" / "summary.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sm_ssot"] = mod
    spec.loader.exec_module(mod)
    return mod


def _led():
    spec = importlib.util.spec_from_file_location("led_ssot", _ROOT / "cc_optimize" / "measurements.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["led_ssot"] = mod
    spec.loader.exec_module(mod)
    return mod


def _funcs_reading(path: Path, key: str) -> set:
    """Function names in `path` whose body mentions `key` as a string literal."""
    out = set()
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and sub.value == key:
                out.add(node.name)
                break
    return out


@pytest.mark.parametrize("fact", sorted(_OWNERS))
def test_only_the_owner_derives_the_fact(fact):
    key, allowed_by_file = _OWNERS[fact]
    offenders = {}
    for rel, allowed in allowed_by_file.items():
        found = _funcs_reading(_ROOT / rel, key)
        extra = found - allowed
        if extra:
            offenders[rel] = sorted(extra)
    assert not offenders, (
        "%r is re-derived outside its owner: %s\n"
        "Call the owning helper instead of reading %r directly -- a second site is how this fact "
        "started disagreeing with itself. If the new site is legitimately a producer, add it to "
        "_OWNERS with the reason." % (fact, offenders, key)
    )


def test_the_win_predicate_has_exactly_one_implementation():
    """A delegating wrapper is fine; a second body is not."""
    sm_src = (_ROOT / "cc_optimize" / "summary.py").read_text()
    assert "_ledger().is_win" in sm_src
    tree = ast.parse(sm_src)
    body = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_is_win")
    assert (
        len([n for n in ast.walk(body) if isinstance(n, ast.Compare)]) == 0
    ), "summary._is_win grew its own comparisons -- it must only delegate"


def test_the_kv_gate_uses_the_same_predicate_as_the_report():
    """The gate decides whether the run keeps ordering KV-cache work, so a looser definition here
    than in the report means the run acts on a win the report will not show."""
    src = (_ROOT / "cc_optimize" / "perf_mcp.py").read_text()
    assert "kv_won = any(_ledger().is_win(a) for a in kv_clean)" in src


def test_the_writer_cannot_stamp_a_win_the_readers_would_refuse(tmp_path, monkeypatch):
    """END TO END: writer and readers agree, so no log can contain a flag its number contradicts."""
    spec = importlib.util.spec_from_file_location("led_ssot", _ROOT / "cc_optimize" / "measurements.py")
    led = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(led)
    for ms, flag, expect in ((648.17, True, True), (0, True, False), (-1, True, False), (648.17, False, False)):
        rec = {"beat_baseline": led.is_win({"beat_baseline": flag, "measured_ms": ms}), "measured_ms": ms}
        assert led.is_win(rec) is expect, rec


def test_anchors_are_per_model_with_no_per_model_code(tmp_path, monkeypatch):
    """Two models, one rule: each gets its own anchor without anything model-specific being written."""
    spec = importlib.util.spec_from_file_location("sm_ssot", _ROOT / "cc_optimize" / "summary.py")
    sm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm)
    led = sm._ledger()
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    monkeypatch.setattr(
        led, "ledger_path", lambda model="", task="": tmp_path / ("%s_%s.jsonl" % (model or "m", task or "main"))
    )

    def pin(v, model):
        return led.anchor(led.KIND_FLOOR, v, depth="16", mode="roofline", model=model, task="main")

    assert pin(537.23, "llama_a") == 537.23
    assert pin(120.00, "whisper_b") == 120.00
    assert pin(331.86, "llama_a") == 537.23
    assert pin(99.00, "whisper_b") == 120.00
    # and the READ side keeps them apart too
    assert sm._floor_anchor(331.86, 16, "llama_a", "main") == 537.23
    assert sm._floor_anchor(99.00, 16, "whisper_b", "main") == 120.00

    # Docstrings cite real models as EVIDENCE of past defects, which is wanted; what must not exist
    # is a model name the code branches on. So inspect executable code only.
    tree = ast.parse((_ROOT / "cc_optimize" / "summary.py").read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            d = ast.get_docstring(node, clean=False)
            if d:
                docstrings.add(d)
    code_strings = [
        n.value.lower()
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value not in docstrings
    ]
    idents = [n.id.lower() for n in ast.walk(tree) if isinstance(n, ast.Name)]
    idents += [n.attr.lower() for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    haystack = code_strings + idents
    for name in ("llama", "whisper", "seamless", "nemotron", "voxtral", "qwen", "kokoro", "phi"):
        hits = [h for h in haystack if name in h]
        assert not hits, "summary.py branches on a specific model (%s): %s" % (name, hits[:5])


def test_the_ledger_anchor_outranks_the_snapshot_for_the_ceiling(tmp_path, monkeypatch):
    """PRECEDENCE, not just ownership. Both stores hold the bytes; the anchor must win. The snapshot is
    regenerated from perf_target_inputs.json, which lives in the model directory the optimize loop
    reverts between attempts -- it was rolled back twice in one run, each time restoring a different
    vintage, and the report printed each one as fact."""
    sm = _sm()
    led = _led()
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    led.anchor(led.KIND_ACTIVE_BYTES, 6094.651392, depth="token", mode="bytes_mb", source="t", model="m")
    snap = {
        "has_unit_ceiling": True,
        "theoretical_rate": 153.8,  # the stale vintage
        "band": [92.3, 123.0],
        "active_bytes": 3_330_000_000,  # ditto
        "peak_bw_gbps": 512.0,
        "tp_degree": 1,
        "perf_layers": "all",
        "unit": "token",
    }
    txt = "\n".join(sm._roofline_lines(snap, None, {"per_token_ms": 17.0}, "m", "main"))
    assert "84.0 tok/s/u" in _flat(txt), txt
    assert "153.8" not in txt and "92.3" not in txt, txt


def test_no_hop_re_derives_the_unit_from_the_model(tmp_path):
    """The unit is OBSERVED once and passed. A second derivation would be a second source of truth
    that can disagree.

    unit_from_config and unit_for_architectures are gone entirely -- a class name cannot say whether a
    model loops, and a unit in the wrong currency takes the band, the at-floor verdict and the
    headline rate with it. unit_for_tag survives for measurement CONDITIONS only (ISL/OSL/steps), so
    no module on this list may reach for it to answer "what does one unit of work cost"."""
    from pathlib import Path as _P

    root = _P(__file__).resolve().parents[1]
    for rel in ("cc_optimize/perf_mcp.py", "cc_optimize/summary.py", "agent/perf_target.py"):
        src = (root / rel).read_text()
        for name in ("unit_for_tag", "unit_from_config", "unit_for_architectures"):
            assert name not in src, "%s re-derives the unit via %s instead of reading it" % (rel, name)
    mb = (root / "agent" / "model_bytes.py").read_text()
    for gone in ("def unit_from_config", "def unit_for_architectures", "_UNIT_BY_ARCH_SUFFIX"):
        assert gone not in mb, "%s is back; the unit has a second source again" % gone
