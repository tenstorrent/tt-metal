from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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


def run_parallel_agents(
    jobs: List[AgentJob],
    *,
    max_workers: int = 4,
) -> List[AgentResult]:
    if not jobs:
        return []
    if max_workers < 1:
        max_workers = 1
    effective_workers = min(max_workers, len(jobs))

    from .agent import _invoke_agent

    def _run_one(job: AgentJob) -> AgentResult:
        rc = _invoke_agent(
            job.prompt,
            provider=job.provider,
            agent_bin=job.agent_bin,
            cwd=job.cwd,
            model=job.model,
            timeout_s=job.timeout_s,
            complexity_bonus=job.complexity_bonus,
            iter_tag=job.iter_tag,
            deliverable_dirs=job.deliverable_dirs,
            expected_deliverable_files=job.expected_deliverable_files,
            require_edit_progress=job.require_edit_progress,
        )
        return AgentResult(component=job.component, rc=rc, cwd=job.cwd)

    print(
        f"  [parallel] spawning {len(jobs)} agent(s) "
        f"({effective_workers} concurrent); targets: "
        f"{', '.join(j.component for j in jobs)}"
    )

    results: List[AgentResult] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        future_to_idx = {pool.submit(_run_one, j): i for i, j in enumerate(jobs)}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                results[idx] = AgentResult(
                    component=jobs[idx].component,
                    rc=2,
                    cwd=jobs[idx].cwd,
                )
                print(f"  [parallel] agent for {jobs[idx].component!r} raised: " f"{type(exc).__name__}: {exc}")

    success_count = sum(1 for r in results if r.rc == 0)
    print(f"  [parallel] {len(jobs)} agent(s) done: " f"{success_count} OK / {len(jobs) - success_count} non-zero")
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


def build_target_focused_prompt(
    *,
    base_prompt: str,
    target_component: str,
    attempts_so_far: int = 0,
    prior_failure_class: str = "",
    prior_failure_block: str = "",
) -> str:
    sep_line = "=" * 78
    header_lines = [
        sep_line,
        f"YOUR TARGET COMPONENT FOR THIS RUN: {target_component}",
        sep_line,
        "",
        f"Edit ONLY `_synth_responses/{target_component}.py` (and read its "
        f"sibling `_stubs/{target_component}.py` for the current stub source).",
        "Do NOT touch other components' files. Other ungraduated components",
        "are being attempted by parallel agents in this same worktree.",
        "",
    ]
    if attempts_so_far > 0:
        header_lines.append(
            f"THIS IS ATTEMPT #{attempts_so_far + 1} ON `{target_component}`. "
            f"Previous attempt failure class: {prior_failure_class or 'NONE'}."
        )
        header_lines.append("")
    if prior_failure_block:
        header_lines.append(
            f"PRIOR FAILURE FOR `{target_component}` (use this to " f"avoid repeating the same mistake):"
        )
        header_lines.append(prior_failure_block)
        header_lines.append("")
        header_lines.append(sep_line)
        header_lines.append("")
    header_lines.append(
        "The base prompt below contains shared context (hardware, exemplars, "
        "shape probes, attempt history across multiple components). Filter "
        f"it for `{target_component}` and ignore sections naming other components."
    )
    header_lines.append(sep_line)
    header_lines.append("")
    return "\n".join(header_lines) + base_prompt
