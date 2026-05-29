"""CLI command: ``classify-hot-cold <model_id>``.

Runs the workload's forward pass with hooks attached to each NEW
component, classifies each as HOT (fired during forward) / COLD
(never fired) / UNRESOLVED (couldn't find the submodule). Persists
the result to ``overlays/<model>/hot_cold.json`` so the auto-iterate
loop can honor it on subsequent runs.

The point of this command is to align Phase 2 with reality: components
that are never invoked by the workload don't need to be on TT device
and don't need standalone PCC -- CPU fallback is the correct state.

Currently uses a default image-style sample input (``torch.randn(1, 3, H, W)``)
since most of the NEW components we care about are vision encoders. For
text / audio / other workloads, the profiler API accepts custom sample
inputs but this CLI wrapper doesn't expose that yet (add ``--workload``
flag if needed)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List


def cmd_classify_hot_cold(args: argparse.Namespace) -> int:
    """Walk through HF model load -> hook attach -> forward pass ->
    classify -> persist."""
    from ..bringup_loop import find_demo_dir
    from ..hot_cold_profiler import (
        make_sample_input,
        profile_hot_cold,
        summarize_hot_cold,
    )
    from ..overlay_manager import persist_hot_cold

    model_id = args.model_id
    demo_dir = find_demo_dir(model_id)
    if demo_dir is None:
        print(f"demo dir not found for {model_id} — has scaffold been run?", file=sys.stderr)
        return 2

    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        print(f"bringup_status.json missing at {status_path}", file=sys.stderr)
        return 2
    try:
        status = json.loads(status_path.read_text())
    except Exception as exc:
        print(f"bringup_status.json unreadable: {exc}", file=sys.stderr)
        return 2

    components = status.get("components", [])
    new_components = [c for c in components if c.get("status") == "NEW"]
    if not new_components:
        print(f"no NEW components in {model_id} — nothing to classify")
        return 0

    print(f"classify-hot-cold for {model_id}")
    print(f"  demo_dir         : {demo_dir}")
    print(f"  NEW components   : {len(new_components)}")

    print(f"  loading HF model from transformers (this may take a moment)...")
    try:
        import transformers

        model = transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
    except Exception as exc:
        print(f"failed to load HF model: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    image_size = int(getattr(args, "image_size", 0) or 0)
    if image_size <= 0:
        cfg = getattr(model, "config", None)
        image_size = int(getattr(cfg, "image_size", 0) or 1024)
    print(f"  sample input     : (1, 3, {image_size}, {image_size}) [image workload]")
    sample = make_sample_input(height=image_size, width=image_size)

    print(f"  running forward pass with hooks attached...")
    classification = profile_hot_cold(
        model=model,
        components=components,
        demo_dir=demo_dir,
        sample_input=sample,
    )

    # BUG-3 FIX: if every NEW component classified as COLD, the model's
    # forward signature likely isn't pixel_values-compatible. Offer to
    # auto-onboard an LLM-drafted invoker (gated on flag/env var).
    all_cold = bool(classification) and all(v == "COLD" for v in classification.values())
    auto_onboard = bool(getattr(args, "auto_onboard", False)) or (
        os.environ.get("TT_PLANNER_AUTO_ONBOARD_HOT_COLD_INVOKER", "") == "1"
    )
    if all_cold and auto_onboard:
        print()
        print(
            "  ALL COMPONENTS COLD — likely model's forward isn't "
            "pixel_values-compatible. Drafting custom invoker via LLM..."
        )
        from ..auto_hot_cold_invoker_onboard import auto_onboard_hot_cold_invoker

        ok, path, msg = auto_onboard_hot_cold_invoker(model=model, model_id=model_id)
        print(f"  [auto-onboard-invoker] {msg}")
        if ok:
            print(f"  Re-running profiler with the new invoker registered...")
            classification = profile_hot_cold(
                model=model,
                components=components,
                demo_dir=demo_dir,
                sample_input=sample,
            )

    buckets = summarize_hot_cold(classification)
    hot = buckets.get("HOT", [])
    cold = buckets.get("COLD", [])
    unresolved = buckets.get("UNRESOLVED", [])

    print()
    print(f"classification:")
    print(f"  HOT        ({len(hot):2}): {', '.join(hot) if hot else '(none)'}")
    print(f"  COLD       ({len(cold):2}): {', '.join(cold) if cold else '(none)'}")
    print(f"  UNRESOLVED ({len(unresolved):2}): {', '.join(unresolved) if unresolved else '(none)'}")

    if not getattr(args, "dry_run", False):
        persist_hot_cold(model_id, classification)
        print()
        print(f"persisted to overlays/<model>/hot_cold.json")
        if cold:
            print(
                f"the auto-iterate loop will now exclude {len(cold)} COLD "
                f"component(s) from work (CPU fallback is correct for them)"
            )
    else:
        print()
        print(f"DRY RUN: classification not persisted (re-run without --dry-run to apply)")

    return 0
