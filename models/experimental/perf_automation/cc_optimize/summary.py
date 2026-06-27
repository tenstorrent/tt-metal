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
    lines.append(
        "levels: grid -> dtype -> tt-lang -> cpp -> host   |   ✓win = beat baseline, ·try = measured no-gain, — = not attempted"
    )
    return "\n".join(lines)
