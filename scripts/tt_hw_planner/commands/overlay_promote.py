from __future__ import annotations


def cmd_overlay_promote(args) -> int:
    from ..overlay_manager import promote

    ok, msg = promote(args.model_id, args.rel_path)
    print(f"  {msg}")
    if not ok:
        return 1
    print(f"  next step: review `git diff {args.rel_path}` and commit through normal PR review")
    return 0
