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

## Coordination Mechanics

- Each instance wrote results to a dedicated file (e.g., `docs/phase3_instance1_results.md`)
- Instances ran in separate Claude Code sessions on the same branch
- Outputs were manually combined via git after each phase
- Later phases read earlier phases' output files as input
- The orchestration doc was updated between phases with decisions and lessons learned

## What Worked

- **Analysis parallelism (phase 1)**: Four independent research tasks with no dependencies. Clean parallelism, high-quality outputs that phase 2 could synthesize.
- **Orchestration doc as shared context**: Every instance had the same understanding of constraints, prior failures, and goals. No drift between instances.
- **Explicit "lessons from prior attempts" section**: Prevented repeating known mistakes (param structs, unused enums, programming example migrations).
- **Separating analysis from design from implementation**: Phase 1 instances were told "document facts, do NOT propose design changes." This prevented premature design decisions before the full picture was understood.

## What Didn't Work

- **Phase 3 ordering dependencies**: Instance 3 (migration) needed instances 1+2 (helpers + tests) done first. This required manual coordination and serialized part of the "parallel" phase.
- **matmul_tile built then removed**: Phase 4 instance 1 built a complete helper that was cut after peer review. The orchestration doc could have flagged this as needing external validation before implementation.
- **High artifact-to-PR ratio**: 11 analysis/design/results files (~5,300 lines) were generated and ultimately discarded. The analysis was necessary to arrive at the right design, but someone replicating this should expect this ratio.
- **Incomplete local test coverage**: Claude ran local tests after each phase (C++ integration tests + Python matmul unit tests), but chose a representative subset rather than exhaustive testing. CI later caught failures that local testing missed — meaning Claude either failed to identify all applicable tests or opted for speed over completeness. The local tests passed, creating false confidence.

## Claude's Assessment

**Strengths of this workflow**: The phased approach forced disciplined thinking — analysis before design, design before implementation. The orchestration doc was the key artifact: it gave each instance enough context to make judgment calls independently rather than following narrow instructions. The "expendable prior work" framing was important — telling instances the existing matmul_block helper was reference-only, not sacred, led to a better redesign.

**What I'd change**:

- **Gate implementation on external review.** Phase 2's design doc should have been reviewed by peers before phase 3 started. The matmul_tile removal and bias_add questions from reviewers could have been caught earlier, saving a full phase of wasted work.
- **Reduce phases.** Phases 1 and 2 could merge — one instance can analyze and design. The analysis files were useful as intermediate artifacts but a single instance with the right prompt could have produced the design directly. This would cut the calendar time roughly in half.
- **Better local test selection.** Claude chose a representative test subset for speed, but missed tests that CI caught. Either the orchestration doc should specify which test suites to run, or Claude should be explicitly told to prioritize completeness over speed for validation phases. A full test run may take longer but avoids false confidence from a passing subset.
- **Fewer, larger instances over more, smaller ones.** The 4-instance phase 1 was clean parallelism, but phase 3's 3 instances had dependencies that made coordination overhead outweigh the parallelism benefit. Two instances (helpers+tests combined, migration separate) would have been simpler.
