# SPDX-License-Identifier: Apache-2.0
"""End-of-run optimization summary for the cc engine.

Reads the per-op kernel-attempts log + the baseline profile and renders a table of what was attempted
at each ladder level (grid / dtype / tt-lang / cpp / host) per op, the best device_ms reached, and the
overall old->new runtime with the percentage speedup. Pure stdlib; additive (touches no opt logic).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

_LEVEL_COLS = ("grid", "fidelity", "dtype", "shard", "host", "tt-lang", "cpp")
_ALL_COLS = _LEVEL_COLS + ("other",)  # "other" holds unclassifiable levers; rendered only when used
_HOST_KINDS = {"trace", "structural", "fusion", "fuse", "gather", "sparse", "cache", "kv-cache"}

_REPORT_NAME = "RUN_REPORT.md"


def upsert_report_section(model_root, key: str, block_md: str):
    try:
        path = Path(model_root) / _REPORT_NAME
        begin, end = f"<!-- BEGIN {key} -->", f"<!-- END {key} -->"
        block = f"{begin}\n{block_md.strip()}\n{end}"
        existing = path.read_text() if path.exists() else ""
        if begin in existing and end in existing:
            pre = existing.split(begin, 1)[0].rstrip()
            post = existing.split(end, 1)[1].lstrip()
            parts = [p for p in (pre, block, post) if p]
        else:
            parts = [existing.strip(), block] if existing.strip() else [block]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n\n".join(parts) + "\n")
        return path
    except Exception:  # noqa: BLE001
        return None


def optimize_block(model_root, attempts_len: int, text: str, when_note: str) -> str:
    from pathlib import Path as _P

    return (
        f"# Optimize (perf) — `{_P(model_root).name}`\n\n" f"_{when_note}_\n\n" "```\n" + (text or "").strip() + "\n```"
    )


def module_optimize_block(
    model_root,
    attempts_len: int,
    text: str,
    when_note: str,
    *,
    module: str,
    index: str = "",
    pcc_gate: str = "",
    outcome: str = "optimizing…",
) -> str:
    """Per-module optimize block for module-level runs: the standard optimize
    block wrapped in the module's ``## Module:`` header, so it renders INSIDE
    that module's own (pre-seeded, correctly-positioned) section and is labelled
    with the module name — instead of a single floating global block that stays
    pinned under whichever module was optimized first."""
    idx = f" — {index}" if index else ""
    head = f"## Module: `{module}`{idx}\n\n- pcc gate: `{pcc_gate}`\n- outcome: **{outcome}**\n\n"
    body = optimize_block(model_root, attempts_len, text, when_note)
    if body.startswith("# Optimize (perf)"):
        body = body.split("\n\n", 1)[1]
    return head + body


_LEVEL_ALIAS_CACHE = Path(tempfile.gettempdir()) / "perf_mcp_lever_alias_cache.json"

_LEVEL_SEMANTICS = (
    "grid = core-grid occupancy / spreading work across cores; "
    "fidelity = math fidelity (HiFi/LoFi); "
    "dtype = weight or activation precision (bf16/bf8_b/bf4_b); "
    "shard = memory sharding, L1 pinning, memory-config changes; "
    "host = host-side or dispatch-side work, tracing, fusion, caching; "
    "tt-lang = custom kernel authored in the tt-lang DSL; "
    "cpp = custom C++ Metalium kernel"
)


def _normalise_kind(kind: str) -> str:
    """Strip rung/knob prefixes and whitespace so `knob:grid` matches the `grid` column."""
    k = (kind or "").strip().lower()
    for pfx in ("knob:", "rung:", "lever:"):
        if k.startswith(pfx):
            k = k[len(pfx) :].strip()
    return k


def _level_of(kind: str) -> str:
    """Ladder column for an attempt, deterministic fast path.

    An unrecognised name returns "other" -- never "host", which previously swallowed
    every `knob:*` lever and any new naming silently. Semantic classification of
    unknown names lives in classify_level().
    """
    k = _normalise_kind(kind)
    if k in _LEVEL_COLS and k != "host":
        return k
    if k in _HOST_KINDS or k == "host":
        return "host"
    return "other"


def _alias_cache() -> dict:
    try:
        return json.loads(_LEVEL_ALIAS_CACHE.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _alias_cache_put(key: str, col: str) -> None:
    try:
        c = _alias_cache()
        c[key] = col
        _LEVEL_ALIAS_CACHE.write_text(json.dumps(c))
    except Exception:  # noqa: BLE001
        pass


def _classify_via_agent(kind: str, note: str, op_signature: str) -> str:
    if os.environ.get("PERF_MCP_NO_AGENT_CLASSIFY") == "1":
        return "other"
    claude = shutil.which("claude")
    if not claude:
        return "other"
    prompt = (
        "Classify ONE optimization attempt into exactly one ladder column.\n\n"
        f"columns: {', '.join(_LEVEL_COLS)}, other\n"
        f"semantics: {_LEVEL_SEMANTICS}\n\n"
        f"attempt label (a HINT only, may be wrong): {kind!r}\n"
        f"op: {op_signature!r}\n"
        f"what the attempt actually did: {(note or '(no note)')[:600]!r}\n\n"
        "Judge from what it DID, not the label. If nothing fits, answer other.\n"
        'Reply with ONLY: {"column":"<one of the above>"}'
    )
    try:
        r = subprocess.run(
            [claude, "-p", prompt, "--output-format", "text"], capture_output=True, text=True, timeout=120
        )
        out = (r.stdout or "").strip()
        i, j = out.find("{"), out.rfind("}")
        col = json.loads(out[i : j + 1]).get("column", "other")
        return col if (col in _LEVEL_COLS or col == "other") else "other"
    except Exception:  # noqa: BLE001
        return "other"


def classify_level(kind: str, note: str = "", op_signature: str = "") -> str:
    """Ladder column derived from what the attempt DID, not from its name.

    `kernel_kind` is a HINT only: reports have carried `tt-lang` on rows whose note
    described L1 weight-pinning (a shard change). Known names resolve deterministically;
    anything else is classified once by a Claude Code agent from the note, then cached by
    content hash so re-renders stay deterministic and cost nothing. Falls back to "other"
    (never "host") when no agent is available.
    """
    det = _level_of(kind)
    if det != "other":
        return det
    key = hashlib.sha1(f"{_normalise_kind(kind)}|{(note or '')[:400]}".encode()).hexdigest()[:16]
    cached = _alias_cache().get(key)
    if cached in _LEVEL_COLS or cached == "other":
        return cached
    col = _classify_via_agent(kind, note, op_signature)
    _alias_cache_put(key, col)
    return col


def _ttl_absent() -> bool:
    """True when the tt-lang (ttl) toolchain is not installed in this env. Rendering runs in the same
    env as the run, so this reflects the real availability the agent had."""
    return importlib.util.find_spec("ttl") is None


def _disp_level(label: str) -> str:
    """DISPLAY-only relabel: a tt-lang rung with no ttl toolchain is really a ttnn implementation the
    agent improvised, so show it as 'ttnn'. Internal column keys / kernel_kind stay 'tt-lang' — this
    changes nothing in the ladder or credit logic."""
    return "ttnn" if label == "tt-lang" and _ttl_absent() else label


def _op_label(sig: str, width: int = 34) -> str:
    """Display label for an op, KEEPING THE SHAPE that distinguishes it.

    This took `split(" ")[0]`, discarding the shape the signature carries -- so five different
    matmuls (32x4096x14336 ff1/ff3, 32x14336x4096 ff2, 32x4096x6144 QKV, 32x4096x4096 wo,
    128x4096x14336 prefill) rendered as seven identical `MatmulDeviceOperation` rows in the ladder
    matrix and the reader could not tell which one a ✓ belonged to. The redundant "DeviceOperation"
    suffix is dropped to make room, since every op in the table carries it.
    """
    s = (sig or "?").strip()
    if not s:
        return "?"
    head, _, rest = s.partition(" ")
    name = head.replace("DeviceOperation", "").replace("Operation", "") or head
    return ("%s %s" % (name, rest.replace(" ", ""))).strip()[:width]


def _read_json(path) -> object:
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return None


def _stage_table_lines(stages: list) -> list:
    """Render the per-stage (block-level) trace timing as bars — the SAME view the HITL pause screen
    shows, so both hitl and non-hitl RUN_REPORT.md surface where device time went per stage/block.
    Fed by the agent's stages passed to record_kernel_attempt. Empty list when no stages present."""
    st = [s for s in (stages or []) if isinstance(s, dict)]
    if not st:
        return []
    peak = max((s.get("ms") or 0) for s in st) or 1.0
    hot = max(st, key=lambda s: s.get("ms") or 0)
    out = []
    for s in st:
        ms = s.get("ms") or 0
        filled = int(round((ms / peak) * 22)) if peak else 0
        bar = "#" * filled + "." * (22 - filled)
        dom = f" · {s['dominant']}" if s.get("dominant") else ""
        mark = "  <- hottest" if s is hot else ""
        out.append(f"  {str(s.get('name', '?')):<12} {ms:>9.2f} ms  {bar}{dom}{mark}")
    return out


def _achievable_band_ms(floor_ms: float):
    """The 60-80%-of-ceiling band expressed in ms, or None.

    Derived from the SAME definition the LLM-decode branch uses: perf_target.target_from_floor_ms
    turns a floor into a rate ceiling and a (lo, hi) rate band, so a slower ms is a LOWER rate --
    the ms band is therefore [floor/hi_frac, floor/lo_frac]. Sourced from perf_target rather than
    re-deriving 0.6/0.8 here, so one module owns what "achievable" means.
    """
    try:
        from agent.perf_target import target_from_floor_ms
    except Exception:  # noqa: BLE001
        return None
    tgt = target_from_floor_ms(floor_ms)
    lo_rate, hi_rate = tgt.band
    if not (lo_rate and hi_rate):
        return None
    return (1000.0 / hi_rate, 1000.0 / lo_rate)


def _floor_status(floor_ms: float, measured_ms: float) -> str:
    """BELOW_BAND | IN_BAND | ABOVE_BAND for a floor-derived (non-decode) target.

    Delegates to perf_target.score so the "measured beat the ceiling -> target suspect, never bank
    it" judgement lives in ONE place; this table used to re-implement that check and then guess
    which side was stale.
    """
    try:
        from agent.perf_target import score, target_from_floor_ms
    except Exception:  # noqa: BLE001
        return "UNKNOWN"
    return str(score(target_from_floor_ms(floor_ms), measured_ms).get("status") or "UNKNOWN")


def _throughput_from_profile(baseline_profile: dict | None) -> dict | None:
    """Compute the roofline-target snapshot from the always-written baseline device profile, so the
    Roofline section renders deterministically even when the per-profile persist did not fire. Uses the
    pure-python roofline + perf_target modules via the agent package. Never raises."""
    if not isinstance(baseline_profile, dict):
        return None
    try:
        _pa = str(Path(__file__).resolve().parent.parent)
        if _pa not in sys.path:
            sys.path.insert(0, _pa)
        from agent import perf_target as _pt
        from agent import roofline as _rl

        rep = _rl.residual_report(baseline_profile, {})
        floor = rep.get("modeled_floor_ms")
        tgt = _pt.target_from_floor_ms(floor)
        return {
            "scope": "model",
            "is_llm_decode": False,
            "theoretical_tok_s": tgt.theoretical_tok_s,
            "band": [tgt.band[0], tgt.band[1]],
            "active_bytes": tgt.active_bytes,
            "peak_bw_gbps": 0.0,
            "tp_degree": tgt.tp_degree,
            "modeled_floor_ms": floor,
        }
    except Exception:  # noqa: BLE001
        return None


def _stages_from_profile(baseline_profile: dict | None) -> list:
    """Derive block-level stage rows from the profile op-class buckets, so the Block-level timing table
    renders deterministically even when no attempt carried per-stage timings."""
    prof = baseline_profile if isinstance(baseline_profile, dict) else {}
    rows = []
    for b in prof.get("buckets") or []:
        if not isinstance(b, dict):
            continue
        rows.append({"name": str(b.get("id", "?")), "ms": float(b.get("device_ms") or 0.0)})
    if not rows:
        return []
    rows.sort(key=lambda r: -r["ms"])
    rows[0]["dominant"] = True
    return rows


def _ledger():
    """The measurement ledger (cc_optimize/measurements.py), loaded by path."""
    global _LEDGER_MOD
    try:
        return _LEDGER_MOD
    except NameError:
        pass
    import importlib.util as _ilu

    _p = Path(__file__).with_name("measurements.py")
    _spec = _ilu.spec_from_file_location("tt_measurements", str(_p))
    _m = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    globals()["_LEDGER_MOD"] = _m
    return _m


def _ledger_pair(kind: str, model: str = "", task: str = ""):
    """(before, after) for one measurement kind, straight from the ledger.

    THE POINT: these carry their own depth and mode, so the report never has to infer what a number
    measured, and there is no chain to fall through when one is missing. `first` is the earliest
    reading ever taken for this (model, task), so it is the TRUE original even on the fifth rerun.
    """
    try:
        led = _ledger()
        return (
            led.first(kind, led.PHASE_BEFORE, model=model, task=task),
            led.last(kind, led.PHASE_AFTER, model=model, task=task),
        )
    except Exception:  # noqa: BLE001
        return None, None


def _ledger_line(kind: str, title: str, model: str = "", task: str = ""):
    """Render one before/after line from the ledger, or None when it has nothing to say."""
    try:
        led = _ledger()
        a, b = _ledger_pair(kind, model, task)
        if not a and not b:
            return None
        if a and not b:
            # A before with no after yet is the normal state for most of a run. Returning None here
            # printed "not measured" over a reading the ledger actually held, hiding the anchor until
            # the first after landed.
            _d = str(a.get("depth") or "unknown")
            return "%s (%s):  %.2f ms  ->  (after not measured yet)" % (
                title,
                "all layers" if _d == "all" else "%s layers" % _d,
                a.get("value_ms"),
            )
        if b and not a:
            _d = str(b.get("depth") or "unknown")
            return "%s (%s):  (before not measured)  ->  %.2f ms" % (
                title,
                "all layers" if _d == "all" else "%s layers" % _d,
                b.get("value_ms"),
            )
        av, bv = a.get("value_ms"), b.get("value_ms")
        depth = a.get("depth") or "unknown"
        dl = "all layers" if str(depth) == "all" else "%s layers" % depth
        ok, why = led.comparable(a, b)
        if not ok:
            return "%s (%s):  before %.2f ms [%s]  ->  after %.2f ms [%s]   — NOT COMPARABLE: %s" % (
                title,
                dl,
                av,
                a.get("mode") or "unknown mode",
                bv,
                b.get("mode") or "unknown mode",
                why,
            )
        pct = led.delta_pct(a, b)
        spd = (av / bv) if bv else 1.0
        return "%s (%s):  %.2f ms  ->  %.2f ms   (%+.1f%%, %.2fx)" % (title, dl, av, bv, pct, spd)
    except Exception:  # noqa: BLE001
        return None


def _depth_label(profile: dict | None = None) -> str:
    """How much of the model the tracy numbers cover. A ms figure means nothing without it: the whole
    2-layer-vs-16-layer confusion came from a headline that printed neither side's depth.

    Read the depth the PROFILE was stamped with, not this process's env. The depth is exported into
    the profiling SUBPROCESS, so the renderer's own TT_PERF_LAYERS is usually empty -- which made the
    label fall through to "all layers" for a run profiled at 16, printing a depth that was simply
    wrong. Env stays as the fallback for profiles written before stamping.
    """
    raw = ""
    if isinstance(profile, dict):
        raw = str(profile.get("perf_layers") or "").strip()
    if not raw:
        raw = (os.environ.get("TT_PERF_LAYERS") or "").strip()
    return "%s layers" % raw if raw.isdigit() and int(raw) > 0 else "all layers"


def _baseline_trace_ms(baseline_profile: dict | None):
    """Delegates to the ledger, which owns how a trace-pass reading is extracted."""
    try:
        return _ledger().trace_ms_from_profile(baseline_profile)
    except Exception:  # noqa: BLE001
        return None


def _floor_basis(profile: dict | None) -> str:
    """What the floor actually is, and how much of the profile it covers.

    "Σ per-op roofline floors" reads like an arbitrary sum. It is physics -- each op's floor is
    max(FLOPs/peak, bytes/bandwidth, count x dispatch) -- and for a weight-streaming model the memory
    term dominates, so the total is essentially bytes/bandwidth. But it sums each bucket's top_ops,
    not every op, and skips host_overhead as non-device work: on llama3_1_8b_p150 that is 86% of
    device time, so the number is a LOWER bound. A confirmation report should state the coverage
    rather than let the figure read as complete.
    """
    base = "Σ per-op max(FLOPs/peak, bytes/BW, dispatch)"
    try:
        buckets = (profile or {}).get("buckets") or []
        total = float((profile or {}).get("device_ms") or 0.0)
        covered = sum(
            float(o.get("device_ms") or 0.0)
            for b in buckets
            if b.get("id") != "host_overhead"
            for o in (b.get("top_ops") or [])
        )
        if total > 0 and covered > 0:
            return "%s; covers %.0f%% of device time" % (base, covered / total * 100.0)
    except Exception:  # noqa: BLE001
        pass
    return base


def _is_win(attempt) -> bool:
    """Delegates to the ledger, which owns what a win is. Not a second implementation."""
    try:
        return _ledger().is_win(attempt)
    except Exception:  # noqa: BLE001
        return False


def _win_set(attempts, baseline_ms=None) -> set:
    """Delegates to the ledger, which owns which attempts actually reduced the measured time."""
    try:
        return _ledger().winning_indices(attempts, baseline_ms)
    except Exception:  # noqa: BLE001
        return set()


def _floor_anchor(current_ms, depth, model: str = "", task: str = ""):
    """The pinned modeled floor for this (model, task, depth), or None if nothing is pinned yet.

    READ-ONLY. The floor is a property of the IMPLEMENTATION, not a goal: halving a weight's dtype
    halves the bytes it must move, so recomputing it each round makes the target retreat ahead of the
    measurement and it is never reached. The pin therefore has to happen once, where the floor is
    PRODUCED (perf_mcp._persist_throughput) -- not here, because rendering a report must not change
    what the next report says.
    """
    try:
        led = _ledger()
        d = str(depth if depth not in (None, "") else "unknown")
        return led.anchor_value(led.KIND_FLOOR, depth=d, model=model, task=task)
    except Exception:  # noqa: BLE001
        return None


def _roofline_lines(
    throughput: dict | None,
    forward_ms: float | None,
    profile: dict | None = None,
    model: str = "",
    task: str = "",
    per_token_ms: float | None = None,
) -> list:
    """The adaptive 'Roofline & utilization' table. MEASURED values (tok/s, mem BW, utilization,
    at-floor) are computed HERE from the ms actually being reported (`forward_ms`) against the STATIC
    target snapshot in `throughput` — so a stale measured can never leak in, and any missing/zero
    input renders 'n/a' rather than a fake 0.0 (the fix for the old '+0.0%' readout). LLM-decode
    pipelines get the tok/s/u form; everything else gets the roofline-floor (ms) form."""
    if not isinstance(throughput, dict):
        return []
    fm = forward_ms if (isinstance(forward_ms, (int, float)) and forward_ms > 0) else None
    theo = throughput.get("theoretical_tok_s")
    out = ["Roofline & utilization"]
    if throughput.get("is_llm_decode") and isinstance(theo, (int, float)) and theo > 0:
        band = throughput.get("band") or [None, None]
        active_bytes = throughput.get("active_bytes") or 0
        tp = max(1, int(throughput.get("tp_degree") or 1))
        per_dev_bytes = (active_bytes / tp) if active_bytes else 0
        # THE UNIT MUST MATCH THE CEILING. This ceiling is per TOKEN (peak_BW / bytes-per-token), and
        # `fm` here is the headline number -- a per-profile device_ms sum over the profiling window.
        # Dividing 1000 by that reported 1.9 tok/s/u against a 64 tok/s/u ceiling: a 3% utilisation
        # readout for a model running at 84%. The per-token reading is the profile's own, and when
        # there is none the line says so rather than printing a number of the wrong kind.
        _pt_ms = per_token_ms
        if not (isinstance(_pt_ms, (int, float)) and _pt_ms > 0):
            try:
                _pt_ms = _ledger().trace_ms_from_profile(profile)
            except Exception:  # noqa: BLE001
                _pt_ms = None
        if isinstance(_pt_ms, (int, float)) and _pt_ms > 0:
            fm = float(_pt_ms)
        measured = (1000.0 / fm) if fm else None
        util = (measured / theo) if measured else None
        bw_gbps = ((per_dev_bytes / (fm / 1000.0)) / 1e9) if (per_dev_bytes and fm) else None
        out.append(f"  theoretical ceiling : {theo:.1f} tok/s/u")
        if band[0] is not None:
            out.append(f"  achievable (60-80%) : {band[0]:.1f} - {band[1]:.1f} tok/s/u")
        out.append(
            f"  measured            : {measured:.1f} tok/s/u   (1000 / {fm:.2f} ms)"
            if measured
            else "  measured            : n/a (no valid forward ms)"
        )
        if bw_gbps is not None:
            out.append(f"  measured mem BW     : {bw_gbps:.0f} GB/s   ({per_dev_bytes / 1e9:.2f} GB / {fm:.2f} ms)")
        if util is None:
            out.append("  utilization         : n/a")
        elif util > 1.0:
            # measured beat the theoretical ceiling -> active_bytes / the target is stale/suspect
            # (the target and the measured ms are from different states); do NOT print a >100% util.
            out.append(f"  utilization         : measured EXCEEDS ceiling — target stale/suspect (re-profile)")
        else:
            out.append(f"  utilization         : {util * 100:.0f}%   (measured / ceiling)")
    else:
        _current = throughput.get("modeled_floor_ms")
        floor = _floor_anchor(_current, throughput.get("perf_layers"), model, task) or _current
        have_floor = isinstance(floor, (int, float)) and floor > 0
        out.append(
            f"  modeled floor       : {floor:.2f} ms   ({_floor_basis(profile)})"
            if have_floor
            else "  modeled floor       : n/a"
        )
        # The floor is NOT a goal: its compute term assumes the FULL grid even for an op that ran on
        # a few cores, and L1/sharded ops fall back to DRAM bandwidth for want of a calibrated L1
        # peak. No real kernel reaches it, so a bare "% of floor" invites chasing an unreachable
        # number. Show the same 60-80% ACHIEVABLE band the LLM branch shows -- derived from THIS
        # floor via perf_target -- so both branches share one definition of "done".
        _band = _achievable_band_ms(floor) if have_floor else None
        if _band:
            out.append(f"  achievable (60-80%) : {_band[0]:.2f} - {_band[1]:.2f} ms")
        out.append(
            f"  measured            : {fm:.2f} ms" if fm else "  measured            : n/a (no valid forward ms)"
        )
        if have_floor and fm:
            status = _floor_status(floor, fm)
            if status == "ABOVE_BAND":
                # Measured beat the floor. Two very different things produce that, and calling both
                # "impossible" puts a false claim in a confirmation document:
                #
                #   the work SHRANK -- the reported floor is pinned to the BASELINE, and optimization
                #       removes bytes (bf8_b -> bf4_b halves a weight read), so the current build's own
                #       bound is lower and beating the baseline's bound is exactly the goal. On
                #       llama3_1_8b_p150 the run reached 534.44 ms against a 537.23 ms baseline floor
                #       while its own floor stood at 331.86 ms: a real win the report called impossible
                #       and told the reader never to bank.
                #
                #   the numbers DISAGREE -- measured beats even the current build's floor, which no
                #       kernel can do, so one side is stale (a 2-layer measurement against a 16-layer
                #       floor produced this).
                #
                # The current floor is the only thing that separates them. It is read here and NOT
                # printed: the report states the baseline, one floor line.
                _beats_own = isinstance(_current, (int, float)) and _current > 0 and fm < _current
                if _beats_own:
                    out.append(
                        f"  status              : ABOVE_BAND — stale/suspect: beats this build's own floor "
                        f"{_current:.2f} ms (re-profile; never bank this as a win)"
                    )
                else:
                    out.append("  status              : PAST BASELINE FLOOR — keep optimizing")
            else:
                _hint = "reached the achievable band — done" if status == "IN_BAND" else "keep optimizing"
                out.append(
                    f"  at-floor            : {floor / fm * 100:.0f}%   ({fm - floor:.2f} ms reachable headroom)"
                )
                out.append(f"  status              : {status} — {_hint}")
        # Do not assert WHY unless it is known. This line claimed "not an LLM decode pipeline" for
        # Llama-3.1-8B, which is false -- the pipeline runs a real traced KV-cache decode. The ms
        # branch is taken whenever the tok/s target is unavailable, and for this model that is because
        # active_bytes is 0: perf_target_inputs.json was never produced, so the weight-bytes-per-token
        # physics has no numerator. A report that goes out for confirmation must not explain a missing
        # input by inventing a property of the model.
        _ab = throughput.get("active_bytes") if isinstance(throughput, dict) else None
        if not _ab:
            out.append("  (tok/s/u — n/a: no weight-bytes input for this pipeline)")
        else:
            out.append("  (tok/s/u — n/a: not an LLM decode pipeline)")
    out.append("")
    return out


def _baseline_bucket_lines(baseline_profile: dict | None, report_csv: str = "") -> list:
    """Render the baseline op-class breakdown (device time per op class, ranked) so an operator can
    read WHAT to target directly from RUN_REPORT.md instead of the terminal/CSV. Sourced from the
    baseline profile's `buckets`; falls back to the baseline_profile.json beside report_csv. Empty
    list when no bucket data is available, so the section silently skips rather than blocking."""
    prof = baseline_profile
    if not (isinstance(prof, dict) and prof.get("buckets")) and report_csv:
        prof = _read_json(Path(report_csv).parent / "baseline_profile.json") or {}
    buckets = prof.get("buckets") if isinstance(prof, dict) else None
    if not buckets:
        return []
    # STATE WHICH MEASUREMENT THIS IS, do not name it. The variable is called baseline_profile, but
    # perf_mcp REWRITES that file on every profile, so the word BASELINE was a claim about provenance
    # the number did not have: on llama3_1_8b_p150 these rows summed to 615.69 ms of device time -- the
    # PREVIOUS reading -- while the roofline above reported 534.44 ms and the true baseline was
    # 2464.18 ms. Three points in the run, one of them labelled the other. Printing the total the rows
    # sum to lets the reader tie the table to a measurement instead of trusting a label.
    _tot = sum((b.get("device_ms") or 0.0) for b in buckets if isinstance(b, dict))
    _depth = _depth_label(prof) if isinstance(prof, dict) else ""
    _hdr_note = "totalling %.2f ms" % _tot if _tot > 0 else "total unknown"
    if _depth:
        _hdr_note += " over %s" % _depth
    out = ["Op breakdown — device time by op class (profile %s · what to target, ranked):" % _hdr_note]
    hdr = f"{'op class':<15} {'device_ms':>10} {'%':>6} {'count':>7} {'bound':>6}  dominant op (shape)"
    out.append(hdr)
    out.append("-" * min(len(hdr) + 30, 118))
    for b in sorted(buckets, key=lambda x: -(x.get("device_ms") or 0.0)):
        if not isinstance(b, dict):
            continue
        top = (b.get("top_ops") or [{}])[0] if b.get("top_ops") else {}
        dom = str(top.get("op_code", "") or top.get("shape", "")).strip()
        ms = b.get("device_ms") or 0.0
        # Recomputed against the total of the rows SHOWN, so the column sums to 100%. The stored pct
        # used a denominator that excluded host_overhead, so host_overhead's own share was overstated
        # (3.5% of 615.69 rather than 3.4% of 637.24) and the column did not add up.
        pct = (ms / _tot * 100.0) if _tot > 0 else (b.get("pct") or 0.0)
        cnt = b.get("count") or 0
        bound = (b.get("tags") or {}).get("bound") or "—"
        out.append(f"{str(b.get('id', '?')):<15} {ms:>10.2f} {pct:>5.1f}% {cnt:>7} {bound:>6}  {dom[:52]}")
    out.append("")
    return out


def render_summary(
    kernel_log_path: str | Path,
    baseline_ms: float | None = None,
    *,
    model: str = "",
    task: str = "main",
    metric: str = "device_ms",
    committed_wins: int | None = None,
    opt_branch: str = "",
    perf_test: str = "",
    report_csv: str = "",
    residual: dict | None = None,
    baseline_profile: dict | None = None,
    finalized: bool = True,
    final_override_ms: float | None = None,
    throughput: dict | None = None,
) -> str:
    """Return a markdown summary. Degrades gracefully when data is partial."""
    attempts = _read_json(kernel_log_path) or []
    if not isinstance(attempts, list):
        attempts = []

    # THE baseline and THE win-set, computed once, before anything renders. A win must have reduced
    # the model's measured time, which is a property of the SEQUENCE, not of one row -- so it is
    # decided here and every section below reads this set instead of re-judging rows.
    _base_row = _ledger_pair(_ledger().KIND_EAGER, model, task)[0]
    hdr_base = float(_base_row["value_ms"]) if _base_row else None
    _wins = _win_set(attempts, hdr_base)

    by_op: dict[str, dict] = {}
    for _i, a in enumerate(attempts):
        if not isinstance(a, dict):
            continue
        sig = a.get("op_signature", "?")
        lvl = classify_level(a.get("kernel_kind", ""), a.get("note", ""), sig)
        ms = a.get("measured_ms")
        won = _i in _wins
        op = by_op.setdefault(sig, {c: None for c in _ALL_COLS})
        cur = op.get(lvl)
        # 'win' beats 'try'; track best (lowest) measured ms per cell
        status = "win" if won else ("wedge" if a.get("wedged") else "try")
        if cur is None:
            op[lvl] = (status, ms)
        elif status == "win" and cur[0] != "win":
            op[lvl] = (status, ms)
        elif cur[0] == "wedge" and status == "try":
            op[lvl] = (status, ms)
        elif cur and ms is not None and cur[1] is not None and ms < cur[1] and cur[0] != "win":
            op[lvl] = (cur[0], ms)

    win_ms = [attempts[i].get("measured_ms") for i in _wins]
    final_ms = final_override_ms if final_override_ms is not None else (min(win_ms) if win_ms else baseline_ms)
    # ANCHOR for the headline, most trustworthy first. Falling straight back to `baseline_ms` is
    # wrong: run.py passes the CURRENT committed ms in that slot, so refusing a stale original made a
    # 3.45x run print "714.94 -> 714.94 (+0.0%)". This run's own baseline_profile.json is written once
    # at the start and is the only value guaranteed to predate every lever.
    # The baseline is a measured fact, never a derived one. This used to substitute the SLOWEST
    # measurement ever recorded whenever the real baseline was not better than the final -- i.e.
    # precisely when the run achieved NOTHING -- so 100 -> 105 with a 180 ms failed experiment
    # printed `baseline 180.00 -> final 105.00 (+41.7%)`, and every `gain vs base` cell was then
    # computed against that invented number. A run that did not improve must say so.

    lines = []
    title = f"Optimization summary — {model or 'model'} · {task} ({metric})"
    lines.append(title)
    lines.append("=" * len(title))
    if not finalized:
        lines.append(
            "optimizing… — baseline->final speedup is finalized when the module converges (per-attempt detail below is live)"
        )
    else:
        _eager = _ledger_line(_ledger().KIND_EAGER, "eager per-op device time", model, task)
        lines.append(_eager or "eager per-op device time: not measured (no ledger reading for this run)")
    _tp = _ledger_line(_ledger().KIND_TRACE_PASS, "tracy trace pass", model, task)
    if _tp:
        lines.append(_tp)
    _trace_scope = f"module ({task})" if os.environ.get("TT_PERF_MODULE_LEVEL") == "1" else "full-pipeline e2e"
    _fp = _ledger_line(_ledger().KIND_FULLPIPE, "trace+1CQ %s" % _trace_scope, model, task)
    if _fp:
        lines.append(_fp)
    lines.append("")

    if not isinstance(throughput, dict):
        throughput = _throughput_from_profile(baseline_profile)
    # The decode ceiling is per TOKEN, so hand the renderer the per-token reading EXPLICITLY rather
    # than letting it divide the headline per-profile device_ms: 1000/534 ms reads 1.9 tok/s/u against
    # a 64 tok/s/u ceiling, i.e. 3% utilisation for a model running at 84%.
    _tok_ms = None
    try:
        _tok_ms = _ledger().trace_ms_from_profile(baseline_profile)
        if _tok_ms is None:
            _row = _ledger_pair(_ledger().KIND_TRACE_PASS, model, task)[1]
            _tok_ms = float(_row["value_ms"]) if _row else None
    except Exception:  # noqa: BLE001
        _tok_ms = None
    # For a per-token ceiling the per-profile sum is not a fallback, it is a WRONG ANSWER, so it is
    # never offered: with no per-token reading the line reads n/a instead of "3% utilisation".
    _is_decode = bool(throughput.get("is_llm_decode")) if isinstance(throughput, dict) else False
    # An EXPLICIT final_override_ms is the caller stating the measured value, and in a decode run that
    # is already the per-token figure (_reliable_forward_ms), so it is honoured. What is withheld is
    # the DERIVED final_ms -- a per-profile sum -- which is not a fallback for a per-token ceiling.
    _fm_for_roofline = final_override_ms if final_override_ms is not None else (None if _is_decode else final_ms)
    lines.extend(
        _roofline_lines(
            throughput,
            _fm_for_roofline,
            baseline_profile,
            model,
            task,
            per_token_ms=_tok_ms,
        )
    )
    lines.extend(_baseline_bucket_lines(baseline_profile, report_csv))

    _st = next((a for a in reversed(attempts) if isinstance(a, dict) and a.get("stages")), None)
    _stages = _st["stages"] if _st else _stages_from_profile(baseline_profile)
    if _stages:
        _lbl = (
            f"latest lever on {_op_label(_st.get('op_signature', '?'))}"
            if _st
            else "op-class breakdown (same profile as the table above)"
        )
        lines.append(f"Block-level timing (per-stage trace) — {_lbl}:")
        lines.extend(_stage_table_lines(_stages))
        lines.append("")

    if by_op:
        _cols = list(_LEVEL_COLS) + (["other"] if any(o.get("other") for o in by_op.values()) else [])
        hdr = f"{'op':<34} " + "  ".join(f"{_disp_level(c):<8}" for c in _cols) + f"  {'best ms':>9}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for sig in sorted(by_op):
            op = by_op[sig]
            cells = []
            best = None
            for c in _cols:
                cell = op.get(c)
                if cell is None:
                    cells.append(f"{'—':<8}")
                else:
                    st, ms = cell
                    mark = "✓win" if st == "win" else ("·wedge" if st == "wedge" else "·try")
                    cells.append(f"{mark:<8}")
                    if ms is not None and (best is None or ms < best):
                        best = ms
            best_s = f"{best:.2f}" if best is not None else "—"
            lines.append(f"{_op_label(sig):<34} " + "  ".join(cells) + f"  {best_s:>9}")
    else:
        lines.append("(no kernel attempts recorded — nothing was tried, or the run stopped before any lever)")

    lines.append("")
    if committed_wins is not None:
        suffix = f" (branch {opt_branch})" if opt_branch else ""
        lines.append(f"committed wins: {committed_wins}{suffix}")

    # --- Per-attempt detail: gain of EVERY optimization tried (#5a) ---
    if attempts:
        lines.append("")
        lines.append("Per-attempt detail (every optimization tried — win OR fail — with gain vs baseline and WHY):")
        ah = f"{'op':<34} {'lever':>12} {'ms':>9} {'gain vs base':>13}  {'result':<10} why tried / why it won or failed"
        lines.append(ah)
        lines.append("-" * min(len(ah), 120))
        for _i, a in enumerate(attempts):
            if not isinstance(a, dict):
                continue
            sig = _op_label(a.get("op_signature", "?"))
            lever = _disp_level(a.get("kernel_kind") or "?")
            ms = a.get("measured_ms")
            ms_s = f"{ms:.2f}" if isinstance(ms, (int, float)) else "—"
            if hdr_base and isinstance(ms, (int, float)):
                gain_s = f"{hdr_base - ms:+.2f} ms"
            else:
                gain_s = "—"
            res = "✓ win" if _i in _wins else ("· wedged" if a.get("wedged") else "· no gain")
            note = " ".join((a.get("note") or "").split())[:200] or "(no reason recorded)"
            lines.append(f"{sig:<34} {lever:>12} {ms_s:>9} {gain_s:>13}  {res:<10} {note}")

    # --- Code changes: the actual source diff for EVERY attempt tried (win or fail) ---
    if any(isinstance(a, dict) and (a.get("diff") or "").strip() for a in attempts):
        lines.append("")
        lines.append("Code changes — every attempt (win or fail):")
        lines.append("=" * 43)
        for i, a in enumerate(attempts, 1):
            if not isinstance(a, dict):
                continue
            d = (a.get("diff") or "").strip()
            if not d:
                continue
            sig = _op_label(a.get("op_signature", "?"))
            lever = _disp_level(a.get("kernel_kind") or "?")
            res = "win" if (i - 1) in _wins else ("wedged" if a.get("wedged") else "no gain")
            ms = a.get("measured_ms")
            gain = f"  {hdr_base - ms:+.2f} ms" if (hdr_base and isinstance(ms, (int, float))) else ""
            lines.append("")
            lines.append(f"[#{i}] {sig} · {lever} · {res}{gain}")
            for dl in d.splitlines():
                lines.append("    " + dl)

    # --- Limitations / suggested manual next steps (#5c) ---
    _won_ops = {attempts[i].get("op_signature") for i in _wins}
    _no_gain = sorted({o for o in by_op} - {o for o in _won_ops if o})
    lines.append("")
    lines.append("Limitations / suggested manual next steps:")
    if _no_gain:
        shown = ", ".join(_op_label(o, 26) for o in _no_gain[:8]) + (" …" if len(_no_gain) > 8 else "")
        lines.append(f"- {len(_no_gain)} op(s) tried but no lever beat baseline: {shown}")
        lines.append("  -> inspect the per-op device report and consider a hand-written kernel or a structural change.")
    _measured = [a for a in attempts if isinstance(a, dict) and a.get("measured_ms") is not None]
    _any_win = bool(_wins)
    if not _measured and attempts:
        _unmeasured = len(attempts)
        lines.append(
            f"- INCONCLUSIVE — {_unmeasured} attempt(s) were made but NONE produced a valid measurement, "
            "so no statement about speedup or the ttnn floor is possible. See the per-attempt reasons above."
        )
    elif baseline_ms and final_ms and final_ms >= baseline_ms and _measured and not _any_win:
        lines.append(
            "- No net speedup recorded — the model may already be at its ttnn floor, or the dominant op needs a custom kernel."
        )
    if residual:
        _rv = residual.get("verdict") or residual.get("summary") or residual.get("reason")
        if _rv:
            lines.append(f"- Roofline residual: {str(_rv)[:200]}")
    if not _no_gain and not residual and not (baseline_ms and final_ms and final_ms >= baseline_ms):
        lines.append("- (none flagged automatically — see the per-op device report for remaining headroom.)")

    # --- Reproduce these numbers (#6) ---
    lines.append("")
    lines.append("Reproduce:")
    lines.append(
        f"  trace+1CQ perf:  python -m pytest {perf_test} -svv"
        if perf_test
        else "  trace+1CQ perf:  (node-id not provided)"
    )
    # Derive the demo (real input/output) + full-model e2e PCC test from the perf-test path
    # (perf tests live under models/demos/<model>/tests/...); best-effort, pointer only.
    _demo_root = ""
    if perf_test:
        _pt = perf_test.split("::")[0]
        _mi = _pt.find("/tests/")
        if _mi > 0:
            _demo_root = _pt[:_mi]
    if _demo_root:
        import os as _os

        _demo_dir = _os.path.join(_demo_root, "demo")
        _e2e_dir = _os.path.join(_demo_root, "tests", "e2e")
        try:
            _demos = sorted(f for f in _os.listdir(_demo_dir) if f.startswith("demo_") and f.endswith(".py"))
        except Exception:
            _demos = []
        try:
            _pccs = sorted(
                f for f in _os.listdir(_e2e_dir) if f.startswith("test_") and f.endswith(".py") and "perf" not in f
            )
        except Exception:
            _pccs = []
        if _demos:
            lines.append(f"  demo (real input→output):  python {_demo_dir}/{_demos[0]}")
        if _pccs:
            lines.append(f"  full-model e2e PCC:  python -m pytest {_e2e_dir}/{_pccs[0]} -svv")
    if report_csv:
        lines.append(f"  per-op device report (tt-metal format): {report_csv}")

    lines.append("")
    lines.append(
        f"levels: grid -> fidelity -> dtype -> shard -> host -> {_disp_level('tt-lang')} -> cpp   |   ✓win = beat baseline, ·try = measured no-gain, ·wedge = wedged/crashed when tried, — = not attempted"
    )
    return "\n".join(lines)
