from __future__ import annotations


def cmd_overlay_drop(args) -> int:
    from ..overlay_manager import drop

    ok = drop(args.model_id, args.rel_path)
    if not ok:
        print(f"  no overlay for {args.rel_path} under {args.model_id}")
        return 1
    print(f"  dropped overlay: {args.model_id} / {args.rel_path}")
    return 0
