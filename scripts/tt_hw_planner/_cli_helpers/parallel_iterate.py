from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class AgentJob:
    component: str
    prompt: str
    cwd: Path
    provider: str
    agent_bin: str
    model: str
    timeout_s: int
    complexity_bonus: int = 0
    iter_tag: Optional[str] = None
    deliverable_dirs: Optional[List[Path]] = None
    expected_deliverable_files: Optional[List[Path]] = None
    require_edit_progress: bool = False


@dataclass
class AgentResult:
    component: str
    rc: int
    cwd: Path


def _remap_under(paths, main_repo: Path, wt: Path):
    if not paths:
        return paths
    out = []
    for p in paths:
        try:
            rel = Path(p).relative_to(main_repo)
            out.append(wt / rel)
        except ValueError:
            out.append(p)
    return out


def run_parallel_agents_adaptive(
    jobs: List[AgentJob],
    *,
    ceiling: int,
    model_id: str,
    main_repo: Path,
) -> List[AgentResult]:
    """Run parallel agents memory-aware: each agent runs in its OWN git
    worktree and the live concurrency scales with host RAM. Returns a
    List[AgentResult] aligned to `jobs`.
    """
    if not jobs:
        return []
    import shutil
    import threading

    from .adaptive_scheduler import AdaptiveScheduler, ScalePolicy
    from .agent import _invoke_agent
    from .agent_worktree_pool import AgentWorktreePool

    main_repo = Path(main_repo)
    pool = AgentWorktreePool(main_repo=main_repo, model_id=model_id)
    harvest_lock = threading.Lock()

    def _run_one(job: AgentJob, slot: int) -> AgentResult:
        wt = pool.acquire(slot)
        try:
            rc = _invoke_agent(
                job.prompt,
                provider=job.provider,
                agent_bin=job.agent_bin,
                cwd=wt,
                model=job.model,
                timeout_s=job.timeout_s,
                complexity_bonus=job.complexity_bonus,
                iter_tag=job.iter_tag,
                deliverable_dirs=_remap_under(job.deliverable_dirs, main_repo, wt),
                expected_deliverable_files=_remap_under(job.expected_deliverable_files, main_repo, wt),
                require_edit_progress=job.require_edit_progress,
            )
            with harvest_lock:
                pool.harvest(slot)
                src_handoff = pool.handoff_dir(slot)
                if src_handoff and src_handoff.is_dir():
                    dst_handoff = main_repo / "_handoff"
                    dst_handoff.mkdir(parents=True, exist_ok=True)
                    for f in src_handoff.iterdir():
                        if f.is_file():
                            shutil.copy2(f, dst_handoff / f.name)
            return AgentResult(component=job.component, rc=rc, cwd=wt)
        finally:
            pool.release(slot)

    ceiling = max(1, int(ceiling))
    print(
        f"  [adaptive] up to {ceiling} agent(s) in isolated worktrees; "
        f"host-RAM monitor will scale concurrency; targets: "
        f"{', '.join(j.component for j in jobs)}"
    )
    policy = ScalePolicy(floor=1, ceiling=ceiling)
    sched = AdaptiveScheduler(jobs=list(jobs), run_one=_run_one, policy=policy)
    try:
        raw = sched.run()
    finally:
        pool.cleanup()

    results: List[AgentResult] = []
    for j, r in zip(jobs, raw):
        results.append(r if isinstance(r, AgentResult) else AgentResult(component=j.component, rc=2, cwd=main_repo))

    success_count = sum(1 for r in results if r.rc == 0)
    limits = [e.new_limit for e in sched.events]
    print(
        f"  [adaptive] {len(jobs)} agent(s) done: {success_count} OK / "
        f"{len(jobs) - success_count} non-zero; limit path: {limits or '[stable]'}"
    )
    return results


def pick_n_distinct_targets(
    candidates: List[str],
    n: int,
    *,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    if n <= 0:
        return []
    excl = set(exclude or [])
    out: List[str] = []
    for c in candidates:
        if c in excl:
            continue
        if c in out:
            continue
        out.append(c)
        if len(out) >= n:
            break
    return out
