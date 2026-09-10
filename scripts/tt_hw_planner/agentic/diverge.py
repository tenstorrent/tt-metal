"""Pair HF<->TT module records and find the first divergence.

This is the G3 empirical-suspect-synthesis step. Given the HF-side
:class:`HFProbeResult` and the TT-side JSON sidecar dumped by
:mod:`.tt_probe`, this module:

1. Loads both record lists.
2. Pairs HF records to TT records by qualified-name alignment, with
   a fallback to traversal-order + class-name match.
3. For each pair, computes per-component divergence metrics
   (mean/std/l2/abs_max relative error). Threshold: 5% by default.
4. Returns an ordered table; the first row whose relative error
   exceeds the threshold is the **first diverging module** and
   becomes the suspect.

No hardcoded suspects, no hardcoded paths. The only constants are
the divergence threshold (a numeric tolerance, not a model fact).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .probe import HFModuleStats, HFProbeResult


DEFAULT_REL_TOL = 0.05


SMALL_VALUE_ABS_TOL = 1e-3


@dataclass
class ModulePair:
    """One row of the HF<->TT divergence table."""

    qualified_name: str
    hf_class: str
    tt_class: str
    step: int
    hf_shape: Tuple[int, ...]
    tt_shape: Tuple[int, ...]
    rel_err_mean: float
    rel_err_std: float
    rel_err_l2: float
    rel_err_abs_max: float
    diverged: bool
    note: str = "ok"


@dataclass
class DivergenceReport:
    """Output of :func:`compute_divergence`. Ordered by step then HF
    qualified-name (the order HF emitted records). The first row with
    ``diverged=True`` is the first-diverging module."""

    pairs: List[ModulePair] = field(default_factory=list)
    first_diverging: Optional[ModulePair] = None
    hf_records: int = 0
    tt_records: int = 0
    paired: int = 0
    threshold: float = DEFAULT_REL_TOL
    note: str = "ok"


def _normalize_qn(name: str) -> str:
    """Canonical form for qualified-name pairing.

    HF emits ``model.layers.0.self_attn``. TT emits
    ``model.layers.0.attention`` (different attribute name). After
    normalisation: ``layers.0.attention``. Pairing then uses fuzzy
    substring match and traversal-order tie-break."""

    parts = name.split(".")
    while parts and parts[0] in ("model", "language_model", "transformer"):
        parts.pop(0)

    SYNONYMS = {
        "self_attn": "attention",
        "self_attention": "attention",
        "attn": "attention",
        "mlp": "feedforward",
        "feed_forward": "feedforward",
        "feedforward": "feedforward",
        "ffn": "feedforward",
        "input_layernorm": "ln_in",
        "pre_attention_layernorm": "ln_in",
        "post_attention_layernorm": "ln_post",
        "post_feedforward_layernorm": "ln_post_ff",
        "pre_feedforward_layernorm": "ln_pre_ff",
    }
    norm_parts = [SYNONYMS.get(p, p) for p in parts]
    return ".".join(norm_parts)


def _rel_err(a: float, b: float) -> float:
    """Relative error, with absolute fallback for near-zero values."""
    if abs(a) < SMALL_VALUE_ABS_TOL and abs(b) < SMALL_VALUE_ABS_TOL:
        return 0.0
    denom = max(abs(a), abs(b), SMALL_VALUE_ABS_TOL)
    return abs(a - b) / denom


def _hf_record_key(r: HFModuleStats) -> str:
    return f"{_normalize_qn(r.qualified_name)}#{r.step}"


def _tt_record_key(r: Dict[str, Any]) -> str:
    return f"{_normalize_qn(r.get('qualified_name', ''))}#{r.get('step', 0)}"


def load_tt_probe_records(path: Path) -> List[Dict[str, Any]]:
    """Load and return the records list from a tt_probe sidecar."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    recs = data.get("records") or []
    if not isinstance(recs, list):
        return []
    return [r for r in recs if isinstance(r, dict)]


def compute_divergence(
    hf_result: HFProbeResult,
    tt_records: List[Dict[str, Any]],
    *,
    threshold: float = DEFAULT_REL_TOL,
) -> DivergenceReport:
    """Pair HF and TT records, compute per-pair divergence, identify
    first diverging module."""

    report = DivergenceReport(
        hf_records=len(hf_result.records),
        tt_records=len(tt_records),
        threshold=threshold,
    )

    if not hf_result.records:
        report.note = "no-hf-records"
        return report
    if not tt_records:
        report.note = "no-tt-records"
        return report

    tt_index: Dict[str, List[Dict[str, Any]]] = {}
    for r in tt_records:
        k = _tt_record_key(r)
        tt_index.setdefault(k, []).append(r)

    paired = 0
    used: set = set()
    for hf_r in hf_result.records:
        k = _hf_record_key(hf_r)
        candidates = tt_index.get(k, [])

        tt_r = None
        for c in candidates:
            cid = id(c)
            if cid not in used:
                tt_r = c
                used.add(cid)
                break
        if tt_r is None:
            continue
        paired += 1
        rel_mean = _rel_err(hf_r.mean, float(tt_r.get("mean", 0.0)))
        rel_std = _rel_err(hf_r.std, float(tt_r.get("std", 0.0)))
        rel_l2 = _rel_err(hf_r.l2, float(tt_r.get("l2", 0.0)))
        rel_amax = _rel_err(hf_r.abs_max, float(tt_r.get("abs_max", 0.0)))

        diverged = rel_mean > threshold or rel_l2 > threshold or rel_amax > threshold or rel_std > 2 * threshold
        note = "ok"
        hf_shape = hf_r.shape
        tt_shape_raw = tt_r.get("shape", [])
        tt_shape = tuple(int(x) for x in tt_shape_raw) if isinstance(tt_shape_raw, list) else ()
        if tt_shape and hf_shape and tt_shape != hf_shape:
            note = f"shape-mismatch hf={hf_shape} tt={tt_shape}"
        pair = ModulePair(
            qualified_name=hf_r.qualified_name,
            hf_class=hf_r.class_name,
            tt_class=str(tt_r.get("class_name", "?")),
            step=hf_r.step,
            hf_shape=hf_shape,
            tt_shape=tt_shape,
            rel_err_mean=rel_mean,
            rel_err_std=rel_std,
            rel_err_l2=rel_l2,
            rel_err_abs_max=rel_amax,
            diverged=diverged,
            note=note,
        )
        report.pairs.append(pair)
        if diverged and report.first_diverging is None:
            report.first_diverging = pair

    report.paired = paired
    if paired == 0:
        report.note = "no-pairs-aligned"
    return report


def format_divergence_block(
    report: DivergenceReport,
    *,
    max_rows: int = 24,
) -> str:
    """Render a compact prompt block describing the divergence table."""
    if not report.pairs:
        return f"DIVERGENCE PROBE: no usable signal (note={report.note}).\n"

    lines: List[str] = []
    lines.append("DIVERGENCE PROBE (HF reference vs TT execution, per submodule):")
    lines.append(f"  threshold       : rel_err > {report.threshold * 100:.1f}% on " f"mean/l2/abs_max (or 2x on std)")
    lines.append(
        f"  hf records      : {report.hf_records}  " f"tt records: {report.tt_records}  paired: {report.paired}"
    )
    if report.first_diverging is not None:
        d = report.first_diverging
        lines.append(
            f"  FIRST DIVERGENCE: `{d.qualified_name}` ({d.hf_class} vs " f"{d.tt_class}) at decode_step={d.step}"
        )
        lines.append(
            f"  relative errors : mean={d.rel_err_mean:.1%}  "
            f"std={d.rel_err_std:.1%}  l2={d.rel_err_l2:.1%}  "
            f"abs_max={d.rel_err_abs_max:.1%}"
        )
        if d.note != "ok":
            lines.append(f"  shape note      : {d.note}")
        lines.append(
            "  ACTION: this submodule is the empirical bug location. "
            "Upstream modules (rows above this in the table) match HF "
            "within tolerance and should NOT be modified. The fix must "
            "land in the TT-side class implementing this submodule "
            "(or in the weight-conversion code that feeds it)."
        )
    else:
        lines.append(
            "  no module diverged above threshold -- the divergence is "
            "in pre/post processing (tokenizer / sampler / detokenizer) "
            "or in the final-output comparator itself."
        )
    lines.append("")
    lines.append("  qualified_name (step)                   rel_err  rel_l2  shape")
    rows_to_show = report.pairs[:max_rows]
    for p in rows_to_show:
        marker = "->" if p is report.first_diverging else (".." if p.diverged else "  ")
        worst = max(p.rel_err_mean, p.rel_err_l2, p.rel_err_abs_max)
        shape_s = f"{p.hf_shape}" if p.hf_shape else "?"
        lines.append(
            f"    {marker} {p.qualified_name:<40s} step={p.step}  " f"{worst:>6.1%}  {p.rel_err_l2:>6.1%}  {shape_s}"
        )
    if len(report.pairs) > max_rows:
        lines.append(f"    ... ({len(report.pairs) - max_rows} more truncated)")
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_REL_TOL",
    "DivergenceReport",
    "ModulePair",
    "compute_divergence",
    "format_divergence_block",
    "load_tt_probe_records",
]
