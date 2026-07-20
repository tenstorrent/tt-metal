"""Aggregate the op-attribution sidecar into ranked hot SOURCE LINES.

Turns the raw per-op records (op, src file:line, shape) that op_attribution_plugin
emits into "which source lines emit the most matmul work" — the automated answer to
"where does the hot op live". A matmul's cost scales with its tensor size, so we rank
each source line by  (#matmul/linear calls there) × (tensor element count)  as a
flops proxy. That surfaces e.g. `decoder_layer.py:382` (384 big 1024->4096 matmuls,
once per decode step) above `feed_forward_network.py` (12 encoder calls) — directing
the agent to where the dominant matmuls ACTUALLY execute, no human deep-dive needed.
"""

from __future__ import annotations

import json
import os
from math import prod

_COMPUTE = {"matmul", "linear"}


def aggregate(sidecar_path: str | os.PathLike, top_k: int = 8) -> list[dict]:
    """Read the attribution sidecar -> ranked hot source lines. Empty list if absent."""
    if not sidecar_path or not os.path.exists(sidecar_path):
        return []
    by_src: dict[str, dict] = {}
    for line in open(sidecar_path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        src = r.get("src")
        if not src:
            continue
        shape = r.get("shape") or []
        size = prod(int(d) for d in shape) if shape else 0
        e = by_src.setdefault(
            src, {"src": src, "func": r.get("func"), "ops": {}, "calls": 0, "work": 0, "shapes": set()}
        )
        op = r.get("op", "?")
        e["ops"][op] = e["ops"].get(op, 0) + 1
        e["calls"] += 1
        if op in _COMPUTE:
            e["work"] += size
        if shape:
            e["shapes"].add(tuple(shape))
    out = []
    for e in by_src.values():
        out.append(
            {
                "src": e["src"],
                "func": e["func"],
                "ops": e["ops"],  # {op_name: count}
                "calls": e["calls"],
                "work": e["work"],  # Σ matmul tensor elements (flops proxy)
                "shapes": sorted(e["shapes"])[:3],
            }
        )
    # Rank by matmul work (the flops proxy); ties by call count.
    out.sort(key=lambda x: (x["work"], x["calls"]), reverse=True)
    return out[:top_k]


def format_hot_sources(hot: list[dict]) -> str:
    """Human/agent-readable block for the route brief + structural-agent prompt."""
    if not hot:
        return "  (op→source attribution unavailable)"
    lines = []
    for h in hot:
        opsum = ", ".join(f"{k}×{v}" for k, v in sorted(h["ops"].items(), key=lambda kv: -kv[1]))
        shp = h["shapes"][0] if h["shapes"] else "?"
        lines.append(f"  - {h['src']} ({h.get('func','?')}): {opsum}; shape {shp}; work≈{h['work']:,}")
    return "\n".join(lines)
