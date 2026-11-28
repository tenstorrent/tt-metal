from __future__ import annotations


def cmd_overlay_revert(args) -> int:
    from ..overlay_manager import revert_for

    n, files = revert_for(args.model_id)
    if n == 0:
        print(f"  no overlays reverted for {args.model_id} (none applied?)")
        return 1
    print(f"  reverted {n} overlay(s) for {args.model_id}:")
    for f in files:
        print(f"    - {f}")
    return 0
