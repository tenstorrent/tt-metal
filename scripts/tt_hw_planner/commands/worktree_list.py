from __future__ import annotations

import time


def cmd_worktree_list(args) -> int:
    from ..worktree import list_active, list_orphans

    active = list_active()
    orphans = {id(s) for s in list_orphans()}

    if not active:
        print("  no tt_hw_planner worktrees present")
        return 0

    print(f"  {'PATH':<70s}  {'MODEL':<48s}  PID    AGE(h)  STATUS")
    print("  " + "-" * 140)
    for s in active:
        age_h = (time.time() - s.created_ts) / 3600.0
        status = "ORPHAN" if id(s) in orphans else "active"
        print(f"  {str(s.path):<70s}  " f"{s.model_id:<48s}  " f"{s.creator_pid:<6d}  " f"{age_h:>6.1f}  " f"{status}")
    return 0
