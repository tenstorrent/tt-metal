from __future__ import annotations


def cmd_overlay_apply(args) -> int:
    from ..overlay_manager import apply_for

    n, files = apply_for(args.model_id)
    if n == 0:
        print(f"  no overlays applied for {args.model_id}")
        return 1
    print(f"  applied {n} overlay(s) for {args.model_id}:")
    for f in files:
        print(f"    + {f}")
    return 0
