# Workflow Report: Multi-Instance Claude Code for Matmul Helper Redesign

## Problem

Design and implement composable matmul compute helpers for Tenstorrent's kernel_lib. The production kernel has ~10 `#ifdef` paths across 827 lines. Prior single-instance attempts produced monolithic helpers that couldn't cover these paths.

## Workflow

### Setup

1. Described the problem, context, and constraints to Claude in a single session
2. Claude generated an orchestration doc (~700 lines) defining phases, instance assignments, output files, and constraints
3. Reviewed and iterated on the plan before execution
4. The orchestration doc served as the shared context for all instances — every instance read it as its starting prompt

### Phase 1 — Parallel Analysis (4 instances)

| Instance | Task | Output |
|----------|------|--------|
| 1 | Feature matrix across all matmul kernels | `docs/analysis_feature_matrix.md` |
| 2 | Production kernel `#ifdef` control flow map | `docs/analysis_production_kernel.md` |
| 3 | Existing helper and composition pattern analysis | `docs/analysis_composition_patterns.md` |
| 4 | Migration gap classification for unmigrated kernels | `docs/analysis_migration_gaps.md` |

### Phase 2 — Design Synthesis (1 instance)

Read all phase 1 outputs. Produced API design with function signatures, type system, coverage matrix, migration sequence, and code sketches for the hardest targets.

### Phase 3 — Parallel Implementation (3 instances)

| Instance | Task |
|----------|------|
| 1 | Core helper implementation (matmul_block refactor + bias_add helper) |
| 2 | C++ integration tests (12 feature combination tests) |
| 3 | Production kernel migration + full regression |

### Phase 4 — Extend Coverage (2 instances, unplanned)

Added after phase 3 to close gaps identified during implementation — matmul_tile for the tile-by-tile pattern and PreKBlockFn to eliminate inline code duplication. This phase was not in the original orchestration doc, which led to less structured coordination.

| Instance | Task |
|----------|------|
| 1 | matmul_tile helper + bmm.cpp migration |
| 2 | PreKBlockFn callback + in0_transpose migration + Blackhole fixes |

### Consolidation

A final session reviewed peer feedback, removed matmul_tile (abysmal perf), answered reviewer questions, and produced a clean PR branch with analysis artifacts stripped out.

### Coordination Mechanics

- Each instance wrote results to a dedicated file (e.g., `docs/phase3_instance1_results.md`)
- Instances ran in separate Claude Code sessions on the same branch
- Outputs were manually combined via git after each phase
- Later phases read earlier phases' output files as input
- The orchestration doc was updated between phases with decisions and lessons learned

## What Worked

- **Orchestration doc as shared context**: The single most valuable artifact. Every instance had the same understanding of constraints, prior failures, and goals. No drift between instances. It also captured "lessons from prior attempts" which prevented repeating known mistakes.
- **Separating analysis from design from implementation**: Phase 1 instances were told "document facts, do NOT propose design changes." This prevented premature design decisions before the full picture was understood, and gave the design phase a solid foundation.
- **Clean parallelism in research phases**: Phase 1's four analysis tasks had zero dependencies. This is the ideal case for multi-instance work — each instance explores independently, results are combined afterward.
- **"Expendable prior work" framing**: Telling instances the existing helper was reference-only, not sacred, led to a better redesign rather than incremental patching.

## Lessons Learned

### Gate design on peer review before implementing

Phase 2's design doc went straight to implementation without peer review. Reviewers later questioned the matmul_tile helper (abysmal perf — shouldn't exist) and the bias_add helper (overlap with existing eltwise helpers). Both could have been caught before spending phases 3-4 implementing. **Recommendation**: Insert a human review checkpoint between design and implementation phases.

### Specify test suites explicitly in the orchestration doc

Claude ran local tests after each phase but chose a representative subset for speed (C++ integration tests + Python matmul unit tests). CI later caught failures that local testing missed. Claude either failed to identify all applicable tests or opted for speed over completeness — either way, the passing local tests created false confidence. **Recommendation**: The orchestration doc should list the exact test commands and explicitly state whether to prioritize completeness or speed.

### Plan for unplanned phases

Phase 4 was added ad-hoc when phase 3 revealed gaps (matmul_tile need, inline code duplication). Because it wasn't in the original plan, the instance assignments were less structured and one instance's entire output (matmul_tile) was later discarded. **Recommendation**: Either build slack into the original plan for discovered work, or treat unplanned phases as a signal to pause and re-plan rather than bolt on.

### Parallelize only truly independent work

Phase 1 (4 analysis instances) was ideal parallelism — no dependencies, no coordination needed. Phase 3 (3 implementation instances) was not — instance 3 needed instances 1+2 done first, serializing the phase and requiring manual coordination. **Recommendation**: If instances have ordering dependencies, run them sequentially or combine them into fewer, larger instances. Reserve parallelism for genuinely independent tasks.

### Expect high artifact-to-PR ratio

11 analysis/design/results files (~5,300 lines) were generated and discarded from the final PR. This is not waste — the analysis was necessary to arrive at the right design. But it means the visible output (a clean PR) understates the actual work, which matters for planning and setting expectations.

### Rebase early

The orchestration ran over several days on a branch 231 commits behind main. By the time CI ran, a "device 2.0 API" migration had landed on main that conflicted with the same kernel files we modified. An earlier rebase or CI check would have caught this before the full workflow completed.
