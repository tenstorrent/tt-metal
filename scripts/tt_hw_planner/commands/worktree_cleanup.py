from __future__ import annotations


def cmd_worktree_cleanup(args) -> int:
    from ..worktree import cleanup_orphans

    n = cleanup_orphans(prompt=not args.yes)
    print(f"  removed {n} orphan worktree(s)")
    return 0
