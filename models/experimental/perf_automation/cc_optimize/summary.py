# SPDX-License-Identifier: Apache-2.0
"""End-of-run optimization summary for the cc engine.

Reads the per-op kernel-attempts log + the baseline profile and renders a table of what was attempted
at each ladder level (grid / dtype / tt-lang / cpp / host) per op, the best device_ms reached, and the
overall old->new runtime with the percentage speedup. Pure stdlib; additive (touches no opt logic).
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

_LEVEL_COLS = ("grid", "fidelity", "dtype", "shard", "tt-lang", "cpp", "host")
_HOST_KINDS = {"trace", "2cq", "structural", "fusion", "fuse", "gather", "sparse", "cache", "kv-cache"}

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


def _level_of(kind: str) -> str:
    k = (kind or "").lower()
    if k in ("grid", "dtype", "fidelity", "shard", "tt-lang", "cpp"):
        return k
    if k in _HOST_KINDS:
        return "host"
    return "host"


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
    name = (sig or "?").strip().split(" ")[0] or "?"
    return name[:width]


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
    out = ["Op breakdown — device time by op class (latest profile · what to target, ranked):"]
    hdr = f"{'op class':<15} {'device_ms':>10} {'%':>6} {'count':>7} {'bound':>6}  dominant op (shape)"
    out.append(hdr)
    out.append("-" * min(len(hdr) + 30, 118))
    for b in sorted(buckets, key=lambda x: -(x.get("device_ms") or 0.0)):
        if not isinstance(b, dict):
            continue
        top = (b.get("top_ops") or [{}])[0] if b.get("top_ops") else {}
        dom = str(top.get("op_code", "") or top.get("shape", "")).strip()
        ms = b.get("device_ms") or 0.0
        pct = b.get("pct") or 0.0
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
    before_ms: float | None = None,
    after_ms: float | None = None,
    baseline_profile: dict | None = None,
    finalized: bool = True,
    original_baseline_ms: float | None = None,
    final_override_ms: float | None = None,
) -> str:
    """Return a markdown summary. Degrades gracefully when data is partial."""
    attempts = _read_json(kernel_log_path) or []
    if not isinstance(attempts, list):
        attempts = []

    by_op: dict[str, dict] = {}
    for a in attempts:
        if not isinstance(a, dict):
            continue
        sig = a.get("op_signature", "?")
        lvl = _level_of(a.get("kernel_kind", ""))
        ms = a.get("measured_ms")
        won = bool(a.get("beat_baseline"))
        op = by_op.setdefault(sig, {c: None for c in _LEVEL_COLS})
        cur = op[lvl]
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

    win_ms = [
        a.get("measured_ms")
        for a in attempts
        if isinstance(a, dict) and a.get("beat_baseline") and a.get("measured_ms") is not None
    ]
    final_ms = final_override_ms if final_override_ms is not None else (min(win_ms) if win_ms else baseline_ms)
    hdr_base = original_baseline_ms if original_baseline_ms is not None else baseline_ms

    lines = []
    title = f"Optimization summary — {model or 'model'} · {task} ({metric})"
    lines.append(title)
    lines.append("=" * len(title))
    if not finalized:
        lines.append(
            "optimizing… — baseline->final speedup is finalized when the module converges (per-attempt detail below is live)"
        )
    elif hdr_base and final_ms and hdr_base > 0:
        pct = (hdr_base - final_ms) / hdr_base * 100.0
        spd = hdr_base / final_ms if final_ms > 0 else 1.0
        lines.append(f"baseline {hdr_base:.2f} ms  ->  final {final_ms:.2f} ms   ({pct:+.1f}%, {spd:.2f}x)")
    elif hdr_base:
        lines.append(f"baseline {hdr_base:.2f} ms  ->  (no measured win recorded)")
    else:
        lines.append("baseline/final ms unavailable (no baseline profile found)")
    _trace_scope = f"module ({task})" if os.environ.get("TT_PERF_MODULE_LEVEL") == "1" else "full-pipeline e2e"
    if before_ms and after_ms:
        _d = (before_ms - after_ms) / before_ms * 100.0 if before_ms else 0.0
        lines.append(
            f"trace+2CQ {_trace_scope}:  before {before_ms:.2f} ms  ->  after {after_ms:.2f} ms"
            f"   ({_d:+.1f}% {'faster' if _d >= 0 else 'SLOWER'})"
        )
    elif before_ms:
        lines.append(f"trace+2CQ {_trace_scope}:  before {before_ms:.2f} ms  ->  (after not measured)")
    lines.append("")

    lines.extend(_baseline_bucket_lines(baseline_profile, report_csv))

    _st = next((a for a in reversed(attempts) if isinstance(a, dict) and a.get("stages")), None)
    if _st:
        lines.append(
            f"Block-level timing (per-stage trace) — latest lever on {_op_label(_st.get('op_signature', '?'))}:"
        )
        lines.extend(_stage_table_lines(_st["stages"]))
        lines.append("")

    if by_op:
        hdr = f"{'op':<34} " + "  ".join(f"{_disp_level(c):<8}" for c in _LEVEL_COLS) + f"  {'best ms':>9}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for sig in sorted(by_op):
            op = by_op[sig]
            cells = []
            best = None
            for c in _LEVEL_COLS:
                cell = op[c]
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
        for a in attempts:
            if not isinstance(a, dict):
                continue
            sig = _op_label(a.get("op_signature", "?"))
            lever = _disp_level(a.get("kernel_kind") or "?")
            ms = a.get("measured_ms")
            ms_s = f"{ms:.2f}" if isinstance(ms, (int, float)) else "—"
            if baseline_ms and isinstance(ms, (int, float)):
                gain_s = f"{baseline_ms - ms:+.2f} ms"
            else:
                gain_s = "—"
            res = "✓ win" if a.get("beat_baseline") else ("· wedged" if a.get("wedged") else "· no gain")
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
            res = "win" if a.get("beat_baseline") else ("wedged" if a.get("wedged") else "no gain")
            ms = a.get("measured_ms")
            gain = f"  {baseline_ms - ms:+.2f} ms" if (baseline_ms and isinstance(ms, (int, float))) else ""
            lines.append("")
            lines.append(f"[#{i}] {sig} · {lever} · {res}{gain}")
            for dl in d.splitlines():
                lines.append("    " + dl)

    # --- Limitations / suggested manual next steps (#5c) ---
    _won_ops = {a.get("op_signature") for a in attempts if isinstance(a, dict) and a.get("beat_baseline")}
    _no_gain = sorted({o for o in by_op} - {o for o in _won_ops if o})
    lines.append("")
    lines.append("Limitations / suggested manual next steps:")
    if _no_gain:
        shown = ", ".join(_op_label(o, 26) for o in _no_gain[:8]) + (" …" if len(_no_gain) > 8 else "")
        lines.append(f"- {len(_no_gain)} op(s) tried but no lever beat baseline: {shown}")
        lines.append("  -> inspect the per-op device report and consider a hand-written kernel or a structural change.")
    if baseline_ms and final_ms and final_ms >= baseline_ms:
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
        f"  trace+2CQ perf:  python -m pytest {perf_test} -svv"
        if perf_test
        else "  trace+2CQ perf:  (node-id not provided)"
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
        f"levels: grid -> fidelity -> dtype -> shard -> {_disp_level('tt-lang')} -> cpp -> host   |   ✓win = beat baseline, ·try = measured no-gain, ·wedge = wedged/crashed when tried, — = not attempted"
    )
    return "\n".join(lines)
