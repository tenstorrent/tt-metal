# SPDX-License-Identifier: Apache-2.0
"""End-of-run optimization summary for the cc engine.

Reads the per-op kernel-attempts log + the baseline profile and renders a table of what was attempted
at each ladder level (grid / dtype / tt-lang / cpp / host) per op, the best device_ms reached, and the
overall old->new runtime with the percentage speedup. Pure stdlib; additive (touches no opt logic).
"""
from __future__ import annotations

import json
from pathlib import Path

_LEVEL_COLS = ("grid", "dtype", "tt-lang", "cpp", "host")
_HOST_KINDS = {"trace", "2cq", "structural", "fusion", "fuse", "gather", "sparse", "cache", "kv-cache"}


def _level_of(kind: str) -> str:
    k = (kind or "").lower()
    if k in ("grid", "dtype", "tt-lang", "cpp"):
        return k
    if k in _HOST_KINDS:
        return "host"
    return "host"


def _op_label(sig: str, width: int = 34) -> str:
    name = (sig or "?").strip().split(" ")[0] or "?"
    return name[:width]


def _read_json(path) -> object:
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return None


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
        status = "win" if won else "try"
        if cur is None or (status == "win" and cur[0] != "win"):
            op[lvl] = (status, ms)
        elif cur and ms is not None and (cur[1] is None or ms < cur[1]):
            op[lvl] = (cur[0] if not won else "win", ms)

    win_ms = [
        a.get("measured_ms")
        for a in attempts
        if isinstance(a, dict) and a.get("beat_baseline") and a.get("measured_ms") is not None
    ]
    final_ms = min(win_ms) if win_ms else baseline_ms

    lines = []
    title = f"Optimization summary — {model or 'model'} · {task} ({metric})"
    lines.append(title)
    lines.append("=" * len(title))
    if baseline_ms and final_ms and baseline_ms > 0:
        pct = (baseline_ms - final_ms) / baseline_ms * 100.0
        spd = baseline_ms / final_ms if final_ms > 0 else 1.0
        lines.append(f"baseline {baseline_ms:.2f} ms  ->  final {final_ms:.2f} ms   ({pct:+.1f}%, {spd:.2f}x)")
    elif baseline_ms:
        lines.append(f"baseline {baseline_ms:.2f} ms  ->  (no measured win recorded)")
    else:
        lines.append("baseline/final ms unavailable (no baseline profile found)")
    if before_ms and after_ms:
        _d = (before_ms - after_ms) / before_ms * 100.0 if before_ms else 0.0
        lines.append(
            f"trace+2CQ full-pipeline e2e:  before {before_ms:.2f} ms  ->  after {after_ms:.2f} ms"
            f"   ({_d:+.1f}% {'faster' if _d >= 0 else 'SLOWER'})"
        )
    elif before_ms:
        lines.append(f"trace+2CQ full-pipeline e2e:  before {before_ms:.2f} ms  ->  (after not measured)")
    lines.append("")

    if by_op:
        hdr = f"{'op':<34} " + " ".join(f"{c:>7}" for c in _LEVEL_COLS) + f" {'best ms':>9}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for sig in sorted(by_op):
            op = by_op[sig]
            cells = []
            best = None
            for c in _LEVEL_COLS:
                cell = op[c]
                if cell is None:
                    cells.append(f"{'—':>7}")
                else:
                    st, ms = cell
                    mark = "✓win" if st == "win" else "·try"
                    cells.append(f"{mark:>7}")
                    if ms is not None and (best is None or ms < best):
                        best = ms
            best_s = f"{best:.2f}" if best is not None else "—"
            lines.append(f"{_op_label(sig):<34} " + " ".join(cells) + f" {best_s:>9}")
    else:
        lines.append("(no kernel attempts recorded — nothing was tried, or the run stopped before any lever)")

    lines.append("")
    if committed_wins is not None:
        suffix = f" (branch {opt_branch})" if opt_branch else ""
        lines.append(f"committed wins: {committed_wins}{suffix}")

    # --- Per-attempt detail: gain of EVERY optimization tried (#5a) ---
    if attempts:
        lines.append("")
        lines.append("Per-attempt detail (every optimization tried, with gain vs baseline):")
        ah = f"{'op':<34} {'lever':>10} {'ms':>9} {'gain vs base':>13}  result"
        lines.append(ah)
        lines.append("-" * len(ah))
        for a in attempts:
            if not isinstance(a, dict):
                continue
            sig = _op_label(a.get("op_signature", "?"))
            lever = _level_of(a.get("kernel_kind", "")) or (a.get("kernel_kind") or "?")
            ms = a.get("measured_ms")
            ms_s = f"{ms:.2f}" if isinstance(ms, (int, float)) else "—"
            if baseline_ms and isinstance(ms, (int, float)):
                gain_s = f"{baseline_ms - ms:+.2f} ms"
            else:
                gain_s = "—"
            res = "✓ win" if a.get("beat_baseline") else "· no gain"
            lines.append(f"{sig:<34} {lever:>10} {ms_s:>9} {gain_s:>13}  {res}")

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
        lines.append("- No net speedup recorded — the model may already be at its ttnn floor, or the dominant op needs a custom kernel.")
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
        f"  trace+2CQ perf:  python -m pytest {perf_test} -svv" if perf_test else "  trace+2CQ perf:  (node-id not provided)"
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
        "levels: grid -> dtype -> tt-lang -> cpp -> host   |   ✓win = beat baseline, ·try = measured no-gain, — = not attempted"
    )
    return "\n".join(lines)
