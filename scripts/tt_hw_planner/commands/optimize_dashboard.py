# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``optimize-dashboard`` — serve the live optimize view for a run, without running optimize.

Attaches to the newest run (or ``--run`` / a target's newest run) and serves the dashboard until
Ctrl+C. The same collector feeds ``optimize --dashboard``, so what a live run shows and what this
shows afterwards are identical.
"""

from __future__ import annotations

from ..optimize_dashboard import (
    collect_state,
    find_run_dir,
    repo_root_for_run,
    run_slug,
    serve,
    state_dir_candidates,
)
from .optimize import _repo_root, _resolve_target


def cmd_optimize_dashboard(args) -> int:
    repo_root = _repo_root()
    slug = None
    target = getattr(args, "target", None)
    if target:
        demo = _resolve_target(target, repo_root)
        if demo is None:
            print(f"  [dashboard] could not resolve '{target}' to a demo dir; showing the latest run instead.")
        else:
            slug = demo.name
    run_dir = find_run_dir(repo_root, slug=slug, run_ref=getattr(args, "run", None))
    if run_dir is None:
        what = f"for '{slug}' " if slug else ""
        print(f"  [dashboard] no optimize run found {what}under {repo_root}/models/experimental/perf_automation/runs.")
        print("  Start one with `optimize <target>`, or pass --run <run-id|path>.")
        return 2
    slug = slug or run_slug(run_dir)
    state_root = repo_root_for_run(run_dir, repo_root)
    print(f"  [dashboard] run: {run_dir.name}")
    return serve(
        args.host,
        args.port,
        lambda: collect_state(run_dir, state_dir_candidates(state_root, slug), slug),
    )
