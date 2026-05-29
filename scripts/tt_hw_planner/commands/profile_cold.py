"""`tt_hw_planner profile-cold <model>` — evidence-based COLD/HOT
classification.

Walks the model's components, measures three signals on the natural
workload:

  1. Frequency      — fires per pass over N forward passes
  2. CPU latency    — per-component contribution to total inference time
  3. Compute density — ops / I/O bytes (bandwidth-bound check)

Persists the enriched evidence record to hot_cold.json. The
categorizer then makes evidence-based COLD verdicts instead of
defaulting to COLD when signals are missing.

This complements `classify-hot-cold` (which only records the kind
label, no measurements). Existing tools that read just the kind via
`load_hot_cold` continue to work — the loader transparently extracts
the kind from the enriched schema.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


def cmd_profile_cold(args) -> int:
    model_id = args.model_id
    n_passes = int(getattr(args, "n_passes", 3) or 3)
    n_iters = int(getattr(args, "n_iters", 5) or 5)

    from ..bringup_loop import _safe_id, find_demo_dir
    from ..cold_evidence import build_evidence_report, evidence_report_to_hot_cold_dict
    from ..overlay_manager import persist_hot_cold

    demo_dir = find_demo_dir(model_id)
    if demo_dir is None:
        print(f"error: no scaffolded demo dir for `{model_id}`; run `up` first.", file=sys.stderr)
        return 2

    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        print(f"error: no bringup_status.json at {status_path}", file=sys.stderr)
        return 2

    try:
        status = json.loads(status_path.read_text())
    except Exception as exc:
        print(f"error: failed to read bringup_status.json: {exc}", file=sys.stderr)
        return 2

    # Build {component_name: submodule_path} for every NEW component that
    # has a captured manifest (submodule_path is recorded there).
    component_paths: Dict[str, str] = {}
    for c in status.get("components", []) or []:
        if c.get("status") != "NEW":
            continue
        name = c.get("name") or ""
        if not name:
            continue
        submodule_path = (c.get("submodule_path") or "").strip()
        if not submodule_path:
            # Fall back to capture manifest
            safe = _safe_id(name)
            mp = demo_dir / "_captured" / safe / "manifest.json"
            if mp.is_file():
                try:
                    submodule_path = (json.loads(mp.read_text()).get("submodule_path") or "").strip()
                except Exception:
                    pass
        if submodule_path:
            component_paths[name] = submodule_path

    if not component_paths:
        print(f"error: no components have submodule_path recorded; nothing to probe.", file=sys.stderr)
        return 2

    print(f"profile-cold: loading HF model `{model_id}`…")
    try:
        import transformers

        model = transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
    except Exception as exc:
        print(f"error: HF load failed: {exc}", file=sys.stderr)
        return 2

    image_size = _infer_image_size(model)
    print(
        f"profile-cold: probing {len(component_paths)} component(s) with "
        f"image_size={image_size}, n_passes={n_passes}, n_iters={n_iters}"
    )

    import torch

    def pixel_values_fn(i: int) -> Any:
        # Varied seeds so multi-pass actually exercises different inputs.
        gen = torch.Generator().manual_seed(i)
        return torch.randn(1, 3, image_size, image_size, generator=gen)

    report = build_evidence_report(
        model=model,
        demo_dir=demo_dir,
        pixel_values_fn=pixel_values_fn,
        component_paths=component_paths,
        n_passes_freq=n_passes,
        n_iters_latency=n_iters,
    )

    classification = evidence_report_to_hot_cold_dict(report)
    persist_hot_cold(model_id, classification)

    # Print per-component summary.
    print()
    print(f"  {'component':<35} {'kind':<10} {'freq':>6}  {'cpu_ms':>8} {'pct':>7}  {'density':>10}  {'evidence'}")
    print(f"  {'-'*35} {'-'*10} {'-'*6}  {'-'*8} {'-'*7}  {'-'*10}  {'-'*40}")
    for name in sorted(report.keys()):
        ev = report[name]
        freq = "-" if ev.frequency is None else f"{ev.frequency:.2f}"
        lat_ms = "-" if ev.cpu_latency_ms is None else f"{ev.cpu_latency_ms:.2f}"
        pct = "-" if ev.cpu_latency_pct is None else f"{ev.cpu_latency_pct:.2f}%"
        dens = "-" if (ev.compute_density is None or ev.compute_density == 0) else f"{ev.compute_density:.2e}"
        why = "; ".join(ev.evidence)[:60]
        print(f"  {name:<35} {ev.kind:<10} {freq:>6}  {lat_ms:>8} {pct:>7}  {dens:>10}  {why}")

    hot_count = sum(1 for e in report.values() if e.kind == "HOT")
    cold_count = sum(1 for e in report.values() if e.kind == "COLD")
    unknown_count = sum(1 for e in report.values() if e.kind == "UNKNOWN")
    print()
    print(f"  HOT: {hot_count}, COLD: {cold_count}, UNKNOWN: {unknown_count}")
    print(f"  persisted to: {Path('scripts/tt_hw_planner/overlays') / model_id.replace('/', '_') / 'hot_cold.json'}")
    return 0


def _infer_image_size(model) -> int:
    """Pull image_size from model.config (HF convention). Falls back to
    1024 if not present (SAM2's default)."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        # Try common attribute names
        for attr in ("image_size", "input_size"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int):
                return val
        # Nested vision config (CLIP-style)
        vision_cfg = getattr(cfg, "vision_config", None)
        if vision_cfg is not None:
            val = getattr(vision_cfg, "image_size", None)
            if isinstance(val, int):
                return val
    return 1024
