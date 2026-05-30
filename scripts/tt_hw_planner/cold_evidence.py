"""Evidence-based COLD/HOT classification.

The previous classifier flagged a component COLD whenever the workload
probe said "never fired" during a single capture pass. That's necessary
but not sufficient — never-fired is just one of several signals that a
component shouldn't be on TT device.

The real question is: **does putting this component on device add
performance value, or is CPU the right placement?**

We collect four signals:

  Signal 1 — Frequency
    How often does the component fire in the natural workload?
      0      → strong COLD evidence (truly unreachable, e.g. video
               components in image-mode workload)
      0..1   → conditional path; might fire on some inputs
      >=1    → on hot path

  Signal 2 — CPU latency contribution
    What fraction of total inference time does this component
    consume on CPU?
      < 0.5%  → COLD-worthy regardless of frequency (won't move
                the needle even if graduated)
      0.5-5%  → marginal value
      > 5%    → HOT — graduation is worthwhile

  Signal 3 — Compute density
    ops_count / (input_bytes + output_bytes)
      Low     → bandwidth-bound; device won't help (host↔device
                transfer cost dominates)
      High    → compute-bound; device helps significantly

  Signal 4 — Empirical CPU vs device benchmark (post-graduation only)
    For graduated components, compare wall-clock CPU vs device
    (including transfer cost). Gold-standard evidence.
      CPU_time <= Device_time + 2*Transfer_time → COLD-worthy
      CPU_time > Device_time + 2*Transfer_time  → HOT (correctly graduated)

The categorization layer reads this enriched evidence and makes a
verdict that is justified by EVIDENCE rather than falling back to
"COLD as safe default."
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Thresholds for the evidence rules. Conservative defaults — operators
# can override per-run via env vars or CLI flags if a particular model
# needs different cutoffs.
COLD_FREQUENCY_THRESHOLD = 0.0  # frequency <= this → COLD-eligible by Signal 1
COLD_LATENCY_PCT_THRESHOLD = 0.5  # latency_pct < this → COLD-eligible by Signal 2
COLD_DENSITY_THRESHOLD = 1e-7  # density < this → bandwidth-bound, COLD-eligible by Signal 3
# Signal 1b — op-affinity rule: only NEGATIVE affinity is COLD-evidence
# (component's ops dominated by bandwidth-bound primitives like permute,
# reshape, gather; CPU wins because transfer cost > compute savings).
# A score of 0 is NEUTRAL (could mean "no opplan data" OR "only neutral
# ops like gelu/relu") and does NOT push toward COLD on its own — other
# signals decide.


@dataclass
class ComponentEvidence:
    """All evidence collected about a single component, persisted to
    the enriched hot_cold.json.

    ``workload_mode`` tags WHICH workload was running when this evidence
    was collected (e.g. ``image`` vs ``video`` for SAM2). The
    categorizer + union-of-evidence helpers use the tag to merge
    multiple per-mode records into a single union verdict: HOT iff HOT
    in ANY mode."""

    kind: str = "UNKNOWN"  # "HOT" | "COLD" | "UNKNOWN"
    frequency: Optional[float] = None  # fires per natural-workload pass
    cpu_latency_ms: Optional[float] = None
    cpu_latency_pct: Optional[float] = None
    ops_count: Optional[int] = None
    io_bytes: Optional[int] = None
    compute_density: Optional[float] = None
    affinity_score: Optional[int] = None  # Σ ttnn-op affinities from opplan
    evidence: List[str] = field(default_factory=list)  # reasons for kind
    workload_mode: str = "default"  # "image" | "video" | "text" | "audio" | "default"
    captured_ts: float = field(default_factory=time.time)


def union_evidence_across_modes(
    per_mode_records: Dict[str, "ComponentEvidence"],
) -> "ComponentEvidence":
    """Combine per-workload-mode evidence records into a single union
    verdict.

    Rule: a component is HOT iff it's HOT in ANY mode. COLD iff COLD
    in EVERY measured mode. UNKNOWN iff no mode has signal.

    Aggregation per signal: take MAX across modes (the strongest
    positive signal wins). The ``evidence`` list is union-deduplicated
    so the operator sees reasons from all modes.

    Use case: profile-cold is run per-workload-mode (e.g. once for
    image, once for video). Each run writes its own record under the
    same hot_cold.json entry. The categorizer reads all modes and
    unions them so multi-mode models are correctly classified.
    """
    if not per_mode_records:
        return ComponentEvidence()

    union = ComponentEvidence(workload_mode="union")
    union.kind = "UNKNOWN"
    union.evidence = []
    seen_reasons: set = set()

    has_hot = False
    all_cold = True
    has_any_signal = False

    for mode, rec in per_mode_records.items():
        if rec.kind == "HOT":
            has_hot = True
            all_cold = False
            has_any_signal = True
        elif rec.kind == "COLD":
            has_any_signal = True
            # all_cold stays True unless we see HOT or UNKNOWN
        else:
            all_cold = False  # UNKNOWN doesn't confirm COLD

        # MAX aggregation across modes
        for attr in ("frequency", "cpu_latency_ms", "cpu_latency_pct", "compute_density", "affinity_score"):
            cur = getattr(union, attr)
            new = getattr(rec, attr)
            if new is None:
                continue
            if cur is None or new > cur:
                setattr(union, attr, new)

        for attr in ("ops_count", "io_bytes"):
            cur = getattr(union, attr)
            new = getattr(rec, attr)
            if new is None:
                continue
            if cur is None or new > cur:
                setattr(union, attr, new)

        for reason in rec.evidence:
            tagged = f"[{mode}] {reason}"
            if tagged not in seen_reasons:
                union.evidence.append(tagged)
                seen_reasons.add(tagged)

    if has_hot:
        union.kind = "HOT"
    elif has_any_signal and all_cold:
        union.kind = "COLD"
    else:
        union.kind = "UNKNOWN"

    return union


def _drive_model_natural(model: Any, pixel_values: Any) -> None:
    """Invoke ``model`` through the natural-workload entry point, using
    the same multi-layer driver framework that capture-inputs uses.

    Tries (in order):
      1. Learned driver registered via @register_capture_driver for
         this model class (e.g. sam2videomodel.py — handles session
         construction for SAM2 video).
      2. Generic drivers (try_capture_drivers fan-out: get_image_embeddings,
         direct forward, etc.).

    Silently returns on failure — the caller is interested in which
    hooks fired, not in whether the model succeeded end-to-end.
    """
    try:
        # Make sure any persisted learned drivers are registered.
        from .auto_capture_driver_onboard import load_learned_drivers

        load_learned_drivers()
    except Exception:
        pass
    try:
        from .capture_drivers import try_capture_drivers

        try_capture_drivers(model, pixel_values)
    except Exception:
        pass


def measure_frequency(
    model: Any,
    pixel_values_fn: Callable[[int], Any],
    component_paths: Dict[str, str],
    n_passes: int = 3,
) -> Dict[str, float]:
    """Signal 1 — Natural-workload firing frequency.

    Runs the model via the driver framework ``n_passes`` times with
    varied inputs, counts how often each component fires (forward-hook
    triggered). Returns ``{component_name: fires_per_pass}``.

    Uses the same driver fan-out as capture-inputs (incl. learned
    drivers) so models with non-trivial entry-point conventions
    (e.g. session-based video models) are driven correctly.

    Multi-pass with varied inputs catches conditional paths — a
    component that only fires on certain inputs gets a fractional
    frequency in (0, 1) rather than a single-pass "yes/no".
    """
    import torch

    fire_counts: Dict[str, int] = defaultdict(int)
    path_to_name = {path: name for name, path in component_paths.items()}

    def make_hook(path: str):
        def hook(_mod, _inputs, _output):
            if path in path_to_name:
                fire_counts[path_to_name[path]] += 1

        return hook

    handles = []
    try:
        for path, mod in model.named_modules():
            if path in path_to_name:
                try:
                    handles.append(mod.register_forward_hook(make_hook(path)))
                except Exception:
                    continue
        for i in range(n_passes):
            inputs = pixel_values_fn(i)
            with torch.no_grad():
                _drive_model_natural(model, inputs)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    return {name: fire_counts.get(name, 0) / n_passes for name in component_paths}


def measure_cpu_latency(
    model: Any,
    pixel_values_fn: Callable[[int], Any],
    component_paths: Dict[str, str],
    n_iters: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Signal 2 — CPU latency contribution per component.

    Runs ``model.forward`` ``n_iters`` times, timing each component's
    forward-pass via pre/post hooks. Returns
    ``{component_name: {"mean_ms": float, "total_ms": float, "pct": float}}``
    where pct is the component's share of total inference time.

    Uses ``time.perf_counter_ns`` for sub-microsecond accuracy. Median-
    of-mean: per-component we record every call's duration, then take
    the mean across calls. Pct = mean_ms / total_inference_mean_ms.
    """
    import torch

    timings: Dict[str, List[float]] = defaultdict(list)
    starts: Dict[str, int] = {}
    path_to_name = {path: name for name, path in component_paths.items()}

    def make_pre_hook(name: str):
        def pre(_mod, _inputs):
            starts[name] = time.perf_counter_ns()

        return pre

    def make_post_hook(name: str):
        def post(_mod, _inputs, _output):
            t0 = starts.pop(name, None)
            if t0 is not None:
                timings[name].append((time.perf_counter_ns() - t0) / 1e6)  # ms

        return post

    handles = []
    try:
        for path, mod in model.named_modules():
            name = path_to_name.get(path)
            if name is None:
                continue
            try:
                handles.append(mod.register_forward_pre_hook(make_pre_hook(name)))
                handles.append(mod.register_forward_hook(make_post_hook(name)))
            except Exception:
                continue

        full_times = []
        model.eval()
        for i in range(n_iters):
            inputs = pixel_values_fn(i)
            t0 = time.perf_counter_ns()
            with torch.no_grad():
                _drive_model_natural(model, inputs)
            full_times.append((time.perf_counter_ns() - t0) / 1e6)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    total_mean_ms = (sum(full_times) / len(full_times)) if full_times else 0.0
    out: Dict[str, Dict[str, float]] = {}
    for name in component_paths:
        durations = timings.get(name, [])
        if not durations:
            out[name] = {"mean_ms": 0.0, "total_ms": 0.0, "pct": 0.0}
            continue
        mean_ms = sum(durations) / len(durations)
        out[name] = {
            "mean_ms": mean_ms,
            "total_ms": sum(durations),
            "pct": (mean_ms / total_mean_ms * 100.0) if total_mean_ms > 0 else 0.0,
        }
    return out


def measure_affinity_score(
    demo_dir: Path,
    component_paths: Dict[str, str],
) -> Dict[str, int]:
    """Signal 1b — ttnn-op affinity score per component.

    Reads each component's opplan manifest and scores it against the
    static ttnn-op affinity catalog. Positive score = component's ops
    are mostly device-favorable (matmul, conv2d, attention); negative =
    bandwidth-bound (permute, reshape, gather).

    Handles three opplan schema variants:
      * ``counts: {op_name: count}``   (legacy summarized form)
      * ``palette: [op_name, ...]``    (legacy palette form)
      * ``pre_bound: [{ttnn_target: op_name, ...}, ...]`` (op-synth form)

    Returns ``{component_name: score}``. Components without an opplan
    manifest get score 0 (neutral safe-default)."""
    from .bringup_loop import _safe_id
    from .op_affinity import component_affinity_score

    out: Dict[str, int] = {}
    for name in component_paths:
        safe = _safe_id(name)
        manifest = demo_dir / "_stubs" / f"{safe}.opplan.json"
        score = 0
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text())
            except Exception:
                data = {}
            # Schema priority: pre_bound (richest — per-instance op
            # records); counts; palette. Score whichever exists.
            if isinstance(data.get("pre_bound"), list):
                ops = [entry.get("ttnn_target") for entry in data["pre_bound"] if isinstance(entry, dict)]
                ops = [op for op in ops if isinstance(op, str) and op]
                score = component_affinity_score(ops)
            elif isinstance(data.get("counts"), dict):
                score = component_affinity_score(data["counts"])
            elif isinstance(data.get("palette"), list):
                score = component_affinity_score(data["palette"])
        out[name] = score
    return out


def measure_compute_density(
    demo_dir: Path,
    component_paths: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """Signal 3 — ops_count / (input_bytes + output_bytes).

    Reads:
      - ops_count from <demo_dir>/_stubs/<safe>.opplan.json (sum of
        REUSE+ADAPT+NEW op counts; the same `_component_op_count` shape).
      - tensor shapes from <demo_dir>/_captured/<safe>/manifest.json
        (the args/kwargs structured representation).

    Returns ``{component_name: {"ops_count": N, "io_bytes": M,
    "density": ops/bytes}}``. Components without opplan or capture
    artifacts get density=0.0.
    """
    from .bringup_loop import _safe_id

    out: Dict[str, Dict[str, float]] = {}
    for name in component_paths:
        safe = _safe_id(name)
        ops = _read_ops_count(demo_dir, safe)
        io_bytes = _read_io_bytes(demo_dir, safe)
        density = (ops / io_bytes) if io_bytes > 0 else 0.0
        out[name] = {"ops_count": ops, "io_bytes": io_bytes, "density": density}
    return out


def _read_ops_count(demo_dir: Path, safe: str) -> int:
    """Sum the opplan manifest's counts dict (REUSE+ADAPT+NEW).
    Falls back to len(palette) if counts is missing."""
    manifest = demo_dir / "_stubs" / f"{safe}.opplan.json"
    if not manifest.is_file():
        return 0
    try:
        data = json.loads(manifest.read_text())
    except Exception:
        return 0
    counts = data.get("counts") or {}
    if isinstance(counts, dict) and counts:
        return sum(int(v) for v in counts.values() if isinstance(v, (int, float)))
    palette = data.get("palette") or []
    return len(palette) if isinstance(palette, list) else 0


def _read_io_bytes(demo_dir: Path, safe: str) -> int:
    """Sum byte sizes of captured args + kwargs + output tensors.
    Reads from manifest.json's structured shape representation
    (avoids loading the actual .pt tensors)."""
    cap_dir = demo_dir / "_captured" / safe
    manifest_path = cap_dir / "manifest.json"
    if not manifest_path.is_file():
        return 0
    try:
        mani = json.loads(manifest_path.read_text())
    except Exception:
        return 0
    total = 0
    for section_key in ("args", "kwargs", "output"):
        total += _walk_tensor_shapes(mani.get(section_key))
    return total


_DTYPE_BYTES = {
    "torch.float32": 4,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float64": 8,
    "torch.int8": 1,
    "torch.int16": 2,
    "torch.int32": 4,
    "torch.int64": 8,
    "torch.bool": 1,
}


def _walk_tensor_shapes(node: Any) -> int:
    """Recursively walk the manifest's nested tensor representation,
    summing byte sizes."""
    if node is None:
        return 0
    if not isinstance(node, dict):
        return 0
    kind = node.get("kind")
    if kind == "tensor":
        shape = node.get("shape") or []
        dtype = node.get("dtype", "torch.float32")
        elem_bytes = _DTYPE_BYTES.get(dtype, 4)
        elements = 1
        for d in shape:
            try:
                elements *= int(d)
            except (TypeError, ValueError):
                return 0
        return elements * elem_bytes
    if kind in ("tuple", "list"):
        return sum(_walk_tensor_shapes(item) for item in (node.get("items") or []))
    if kind == "dict":
        return sum(_walk_tensor_shapes(v) for v in (node.get("items") or {}).values())
    return 0


def classify_evidence(
    frequency: Optional[float],
    cpu_latency_pct: Optional[float],
    compute_density: Optional[float],
    affinity_score: Optional[int] = None,
) -> Tuple[str, List[str]]:
    """Combine the available signals into a HOT/COLD verdict + reasons.

    Returns ``(kind, evidence)`` where kind is "HOT" | "COLD" | "UNKNOWN"
    and evidence is a list of human-readable reasons.

    Rule (any single qualifying signal triggers COLD):
      Signal 1   — frequency <= COLD_FREQUENCY_THRESHOLD (never fires)
      Signal 2   — latency_pct < COLD_LATENCY_PCT_THRESHOLD (negligible)
      Signal 3   — density < COLD_DENSITY_THRESHOLD (bandwidth-bound)
      Signal 1b  — affinity_score <= COLD_AFFINITY_THRESHOLD (component
                   lacks device-favorable ops; device adds no value)

    HOT iff none of the COLD signals fire AND we have at least one
    positive HOT signal (freq > 0 OR latency >= threshold OR
    affinity > 0).

    UNKNOWN iff we have no signals at all (probe didn't run).
    """
    evidence: List[str] = []
    is_cold = False
    has_any_signal = False

    if frequency is not None:
        has_any_signal = True
        if frequency <= COLD_FREQUENCY_THRESHOLD:
            is_cold = True
            evidence.append(f"frequency={frequency:.2f} (never-fired in workload)")

    if cpu_latency_pct is not None:
        has_any_signal = True
        if cpu_latency_pct < COLD_LATENCY_PCT_THRESHOLD:
            is_cold = True
            evidence.append(f"cpu_latency_pct={cpu_latency_pct:.2f}% (< {COLD_LATENCY_PCT_THRESHOLD}% threshold)")

    if compute_density is not None and compute_density > 0:
        has_any_signal = True
        if compute_density < COLD_DENSITY_THRESHOLD:
            is_cold = True
            evidence.append(f"compute_density={compute_density:.2e} (bandwidth-bound)")

    if affinity_score is not None:
        has_any_signal = True
        if affinity_score < 0:
            is_cold = True
            evidence.append(f"affinity_score={affinity_score} (lacks device-favorable ops)")

    if not has_any_signal:
        return "UNKNOWN", ["no probe data — fall back to safe default"]

    if is_cold:
        return "COLD", evidence

    if frequency is not None and frequency > 0:
        evidence.append(f"frequency={frequency:.2f} (on hot path)")
    if cpu_latency_pct is not None and cpu_latency_pct >= COLD_LATENCY_PCT_THRESHOLD:
        evidence.append(f"cpu_latency_pct={cpu_latency_pct:.2f}% (>= threshold)")
    if affinity_score is not None and affinity_score > 0:
        evidence.append(f"affinity_score={affinity_score} (has device-favorable ops)")
    return "HOT", evidence


def build_evidence_report(
    model: Any,
    demo_dir: Path,
    pixel_values_fn: Callable[[int], Any],
    component_paths: Dict[str, str],
    *,
    n_passes_freq: int = 3,
    n_iters_latency: int = 5,
) -> Dict[str, ComponentEvidence]:
    """End-to-end: collect all signals, classify each component,
    return per-component evidence records ready to persist to
    hot_cold.json.

    ``component_paths`` is ``{name: submodule_path}`` (from
    bringup_status.json + capture manifests). Only these components are
    probed.
    """
    freq = measure_frequency(model, pixel_values_fn, component_paths, n_passes=n_passes_freq)
    lat = measure_cpu_latency(model, pixel_values_fn, component_paths, n_iters=n_iters_latency)
    density = measure_compute_density(demo_dir, component_paths)
    affinity = measure_affinity_score(demo_dir, component_paths)

    report: Dict[str, ComponentEvidence] = {}
    for name in component_paths:
        f = freq.get(name)
        lat_entry = lat.get(name) or {}
        lat_ms = lat_entry.get("mean_ms")
        lat_pct = lat_entry.get("pct")
        dens_entry = density.get(name) or {}
        ops = dens_entry.get("ops_count")
        io_b = dens_entry.get("io_bytes")
        dens = dens_entry.get("density")
        aff = affinity.get(name)

        kind, ev = classify_evidence(f, lat_pct, dens, aff)
        report[name] = ComponentEvidence(
            kind=kind,
            frequency=f,
            cpu_latency_ms=lat_ms,
            cpu_latency_pct=lat_pct,
            ops_count=ops,
            io_bytes=io_b,
            compute_density=dens,
            affinity_score=aff,
            evidence=ev,
        )
    return report


def evidence_report_to_hot_cold_dict(report: Dict[str, ComponentEvidence]) -> Dict[str, dict]:
    """Convert evidence records to the schema persisted via
    overlay_manager.persist_hot_cold(). The categorizer reads from
    this format."""
    out: Dict[str, dict] = {}
    for name, ev in report.items():
        out[name] = asdict(ev)
    return out


def merge_evidence_into_persisted(
    existing: Dict[str, dict],
    new_report: Dict[str, ComponentEvidence],
    workload_mode: str,
) -> Dict[str, dict]:
    """Merge a new per-workload-mode evidence run into the persisted
    hot_cold.json.

    Schema:
      Single-mode (legacy): {comp: {evidence-fields...}}
      Multi-mode (new):     {comp: {"modes": {mode: {evidence-fields...}},
                                    "kind": <union>, "evidence": [<union>]}}

    On merge:
      - If the existing entry is single-mode (no "modes" key), promote
        it to multi-mode with its content under modes["<existing_mode>"]
        (defaults to "default" if no tag).
      - Add/overwrite the new run under modes["<workload_mode>"].
      - Recompute the union-of-evidence verdict at the top level.

    The categorizer always reads the top-level "kind" — which is now
    the union across modes. Per-mode raw data is preserved for
    drill-down.
    """
    merged: Dict[str, dict] = {}
    all_names = set(existing.keys()) | set(new_report.keys())

    for name in all_names:
        old_entry = existing.get(name)
        modes: Dict[str, ComponentEvidence] = {}
        # Track per-mode extra fields (bench, bench_verdict, ...) that
        # don't fit in ComponentEvidence — these need to ROUND-TRIP
        # through the merge so a profile-cold re-run doesn't wipe a
        # prior bench-component result for the same mode.
        mode_extras: Dict[str, dict] = {}

        if old_entry is not None:
            if isinstance(old_entry, dict) and isinstance(old_entry.get("modes"), dict):
                # Already multi-mode; load each.
                for mode_name, mode_data in old_entry["modes"].items():
                    if not isinstance(mode_data, dict):
                        continue
                    modes[mode_name] = _evidence_from_dict(mode_data)
                    # Capture non-evidence fields (bench, bench_verdict, etc.)
                    mode_extras[mode_name] = {k: v for k, v in mode_data.items() if k not in _COMPONENT_EVIDENCE_FIELDS}
            elif isinstance(old_entry, dict):
                # Legacy single-mode entry; promote.
                old_mode = old_entry.get("workload_mode") or "default"
                modes[old_mode] = _evidence_from_dict(old_entry)
                mode_extras[old_mode] = {k: v for k, v in old_entry.items() if k not in _COMPONENT_EVIDENCE_FIELDS}

        if name in new_report:
            modes[workload_mode] = new_report[name]

        if not modes:
            continue

        union = union_evidence_across_modes(modes)
        # Rebuild modes dict, merging extras (bench results etc.) back in.
        modes_persisted: dict = {}
        for m, rec in modes.items():
            entry = asdict(rec)
            extras = mode_extras.get(m, {})
            for k, v in extras.items():
                # Don't let extras overwrite freshly-measured values
                if k not in entry:
                    entry[k] = v
            modes_persisted[m] = entry

        merged[name] = {
            "kind": union.kind,
            "evidence": union.evidence,
            "modes": modes_persisted,
            "frequency": union.frequency,
            "cpu_latency_ms": union.cpu_latency_ms,
            "cpu_latency_pct": union.cpu_latency_pct,
            "compute_density": union.compute_density,
            "ops_count": union.ops_count,
            "io_bytes": union.io_bytes,
            "affinity_score": union.affinity_score,
            "captured_ts": time.time(),
        }
    return merged


_COMPONENT_EVIDENCE_FIELDS = frozenset(
    {
        "kind",
        "frequency",
        "cpu_latency_ms",
        "cpu_latency_pct",
        "ops_count",
        "io_bytes",
        "compute_density",
        "affinity_score",
        "evidence",
        "workload_mode",
        "captured_ts",
    }
)


def _evidence_from_dict(d: dict) -> ComponentEvidence:
    """Construct a ComponentEvidence from a JSON-loaded dict, accepting
    extra/missing keys gracefully."""
    return ComponentEvidence(
        kind=str(d.get("kind", "UNKNOWN")),
        frequency=d.get("frequency"),
        cpu_latency_ms=d.get("cpu_latency_ms"),
        cpu_latency_pct=d.get("cpu_latency_pct"),
        ops_count=d.get("ops_count"),
        io_bytes=d.get("io_bytes"),
        compute_density=d.get("compute_density"),
        affinity_score=d.get("affinity_score"),
        evidence=list(d.get("evidence") or []),
        workload_mode=str(d.get("workload_mode", "default")),
        captured_ts=float(d.get("captured_ts") or time.time()),
    )
