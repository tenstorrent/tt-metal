from __future__ import annotations


def cmd_overlay_extract(args) -> int:
    from ..overlay_manager import extract_from_working_tree

    failures = 0
    for rel_path in args.rel_paths:
        ok, msg = extract_from_working_tree(
            args.model_id,
            rel_path,
            hunks_matching=getattr(args, "hunks_matching", None),
            intended_for_production=getattr(args, "intended_for_production", False),
        )
        prefix = "  ok " if ok else "  fail"
        print(f"{prefix} {rel_path}: {msg}")
        if not ok:
            failures += 1
    return 1 if failures else 0
