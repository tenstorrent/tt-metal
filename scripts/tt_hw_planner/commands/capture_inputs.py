from __future__ import annotations
from ..discovery import safe_relative_to_root

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def cmd_capture_inputs(args) -> int:
    """Capture real per-component IO by running the HF model once with
    forward hooks; saves the tensors to `<demo>/_captured/<safe>/...` and
    patches the generated PCC tests to load them instead of synthetic
    randoms. This eliminates the `synthetic inputs incompatible` SKIPs
    for prompt-conditioned heads (SAM2's mask_decoder being the
    motivating example).
    """
    from ..cli import REPO_ROOT, _load_bringup_status

    try:
        demo_dir, status = _load_bringup_status(args.model_id)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    from ..capture_inputs import capture_real_inputs, upgrade_all_tests_in_demo

    if args.component:
        comps = [args.component]
    else:
        comps = [c.get("name") for c in status.get("components", []) if c.get("status") == "NEW" and c.get("name")]

    if not comps:
        print("No NEW components to capture (model already fully native or " "scaffold not yet run).")
        return 0

    print(f"\nCapture-and-replay inputs for {args.model_id}")
    print(f"  demo dir:   {safe_relative_to_root(demo_dir)}")
    print(f"  components: {len(comps)}  ({', '.join(comps)})")
    print()

    results = capture_real_inputs(
        model_id=args.model_id,
        demo_dir=demo_dir,
        components=comps,
        image_size_override=args.image_size,
        verbose=True,
    )

    print()
    print("Capture results:")
    captured_count = 0
    for comp, info in results.items():
        status_str = info.get("status", "?")
        if status_str == "captured":
            captured_count += 1
        sub_path = info.get("submodule_path", "-")
        print(f"  {comp:30s} {status_str:25s} via `{sub_path}`")
    print()
    print(f"  captured {captured_count}/{len(comps)} components")

    if not args.no_upgrade_tests:
        print()
        print("Patching PCC tests to prefer captured inputs:")
        outcomes = upgrade_all_tests_in_demo(demo_dir)
        upgraded = sum(1 for _n, m in outcomes if m)
        for name, modified in outcomes:
            badge = "UPGRADED" if modified else "skip"
            print(f"  {name:50s} {badge}")
        print(f"  {upgraded} file(s) upgraded.")

    return 0
