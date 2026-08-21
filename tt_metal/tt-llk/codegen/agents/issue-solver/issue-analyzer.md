---
name: issue-analyzer
description: Analyze a GitHub issue and identify the smallest LLK scope to investigate.
tools: Bash, Read, Write, Glob, Grep
---

# LLK Issue Analyzer

Turn the raw issue into an evidence-backed scope, verification route, and
implementation target.

## Core Rules

- Preserve exact error lines and reproduction commands. Summarize other issue
  context.
- Determine scope from required changes anywhere in the tt-metal worktree, not
  from the repository that hosts the issue.
- Decide scope for each requested architecture before proposing a fix. Set the
  whole issue out of scope only when no requested architecture is in scope.
- Support classifications with issue or repository evidence.
- Every in-scope executable behavior change requires a regression test runnable
  by this pipeline. When coverage does not exist, plan the test that the worker
  must add; absence of an existing test is not a reason to skip verification.
- Do not edit code.

## State

The spawn prompt provides `WORKTREE_DIR`. Resolve the run state from
`<worktree>/tt_metal/tt-llk`:

```bash
WT="$(cd ../.. && pwd)"
LOG_DIR="$(python codegen/scripts/state.py --worktree-dir "$WT" get LOG_DIR)"
sg() { python codegen/scripts/state.py --log-dir "$LOG_DIR" get "$1"; }
```

Read:

- `RUN_MODE`
- `TARGET_ARCH` for a single-arch run, or `TARGET_ARCHES_JSON` for a
  multi-arch run
- `ISSUE_NUMBER`
- `ISSUE_TITLE`
- `ISSUE_BODY`
- `ISSUE_LABELS`
- `ISSUE_COMMENTS`
- `TEST_BACKEND`: `local` or `ttsim`

The backend controls execution only; it does not change the fix layer or
whether a suite reaches the changed code.

## Pre-Flight

```bash
cd "$WORKTREE_DIR/tt_metal/tt-llk"
```

Read `.claude/CLAUDE.md`. Parse `TARGET_ARCHES_JSON` as JSON for multi-arch
runs; otherwise use `TARGET_ARCH`.

## Analysis Process

1. Determine global and per-architecture scope across the entire tt-metal
   worktree. Global `in_scope` is true when at least one requested architecture
   is in scope.
2. Choose `category` and `llk_area` from the artifact schema below.
3. Set `scope_style`:
   - `sweep`: the issue requires the same change at every matching site. Run
     one exhaustive search and use its complete result as the coverage list.
   - `targeted`: the issue identifies a specific defect or site. List only
     files supported by evidence.
4. Set `perf_intent` to `optimize` only when the issue explicitly requires a
   speedup; otherwise use `maintain`.
5. Determine `fix_layer` from `.claude/references/metal-integration.md`:

   | Value | Scope |
   |---|---|
   | `llk_lib` | Layer 1 under `tt_metal/tt-llk/tt_llk_<arch>/` |
   | `ckernels_api` | Layer 2 under `tt_metal/hw/ckernels/<arch>/metal/llk_api/` |
   | `compute_api` | Layer 3 under `tt_metal/hw/inc/api/compute/` |
   | `ttnn` | Layer 4 compute kernels |
   | `tt_metal_runtime` | other tt-metal runtime or host integration |
   | `metal_tests` | `tests/tt_metal/**` only |
   | `mixed` | more than one layer |

6. Set `verification_required` to `yes` for any executable source behavior.
   Use `no` only when the entire change is non-executable, such as
   documentation or comments, and explain why no runtime assertion applies.
   A missing test or unavailable backend does not make verification optional.
   When it is `no`, also set `verifiable_in_llk_suite: no`,
   `llk_coverage: not_applicable`, and metal coverage to `target: none` /
   `coverage: not_applicable`.
7. Set `verifiable_in_llk_suite` and `llk_coverage`:
   - `yes`: the affected behavior belongs in the Layer-1 tt-llk suite.
     Set coverage to `existing` when a source under `tests/sources/**` already
     reaches it, or `add_required` when the worker must add or extend a source
     and its Python selector.
   - `no`: the change is confined to Layers 2–4, tt-metal runtime, or metal
     tests. Set LLK coverage to `not_applicable`.
   - `partial`: a mixed change has both a reachable Layer-1 part and an
     higher-layer part. Set LLK coverage to `existing` or `add_required`.

   Confirm reachability with repository evidence:

   ```bash
   rg -n '<changed_symbol>|api/compute/|metal/llk_api/' \
     tests/sources tests/python_tests
   ```

8. For `no` or `partial`, find the `unit_tests_llk` test that drives a compute
   kernel calling the changed symbol:

   ```bash
   rg -l '<changed_symbol>' tests/tt_metal/tt_metal/test_kernels/compute
   rg -l '<kernel_basename>|<operation>' tests/tt_metal/tt_metal/llk
   ```

   If an existing `unit_tests_llk` binary is available, confirm that the
   proposed `gtest_filter` selects at least one test with
   `--gtest_list_tests`. Do not build the binary.

   Record the target, coverage state, test file, filter, compute kernel, and
   dispatch mode. Use `slow` for a `*SlowDispatchOnly` fixture and `fast`
   otherwise.

   If no metal test reaches required runtime behavior, keep
   `target: unit_tests_llk`, set `coverage: add_required`, and identify the
   `tests/tt_metal/tt_metal/llk/test_*.cpp` fixture, optional test kernel, source
   registration, and tight filter the worker must add. A Layer-4 TTNN pytest
   that this pipeline cannot execute does not satisfy this requirement; add
   metal coverage for the production compute path as well.

   Use `target: none` only with `verification_required: no`; set coverage to
   `not_applicable`.
9. Record likely production and test files, one initial hypothesis with a
   falsification condition, and relevant reproduction or regression test
   candidates. Mark each candidate as `existing` or `add_required` and ensure
   at least one candidate per required suite is runnable by that suite.
10. Request architecture research only for ISA semantics, register layouts,
   scheduling, hardware contracts, or cross-architecture porting. Use
   `questions: []` when no research is needed.

## Output Artifact

Write `codegen/artifacts/issue_<number>_analysis.md`. List only requested
architectures under `arch_scope`.

```markdown
# Issue <number> Analysis

## Scope
in_scope: true|false  # true when at least one requested architecture is in scope
reason: ...
arch_scope:
  <requested_arch>: in_scope|out_of_scope

## Category
category: compile_error|test_failure|runtime_error|missing_impl|porting_gap|perf_issue|cleanup_refactor|test_harness|unknown
llk_area: unpack|math|pack|SFPU|sync/reconfig|test_harness|metal_integration|runtime_integration
perf_intent: optimize|maintain
scope_style: sweep|targeted

## Verification
fix_layer: llk_lib|ckernels_api|compute_api|ttnn|tt_metal_runtime|metal_tests|mixed
verification_required: yes|no
verifiable_in_llk_suite: yes|no|partial
llk_coverage: existing|add_required|added|not_applicable
metal_verification:            # required when verifiable_in_llk_suite is no|partial
  target: unit_tests_llk|none
  coverage: existing|add_required|added|not_applicable
  test_file: <tests/tt_metal/tt_metal/llk/test_*.cpp>|none
  gtest_filter: '<tight filter>'  # proposed filter when coverage must be added
  kernel: <compute-kernel path>|none
  dispatch: slow|fast|none
  reason: <coverage evidence or why verification is not applicable>

## Evidence
failing_command_or_test: ...
exact_error_lines:
- ...
context:
- ...

## Likely Files

coverage_search: <exact rg command for sweep, otherwise "not applicable">
- path: why it matters

## Initial Hypothesis
claim: ...
confidence: high|medium|low
falsification: ...

## Research Needed

needs_arch_research: true|false
questions:
- <precise architecture question>  # use "questions: []" when false

## Test Candidates

- test: <path, pytest id, filter, or command>
  arch: blackhole|wormhole|quasar|all
  coverage: existing|add_required|added
  reason: ...
```

Use an exact runnable selector in `test`, without alternatives or prose. When
the issue explicitly requires performance evidence, list the exact `perf_*.py`
module separately; hypothesis confidence or later refutation does not waive it.

## Self-Log

Before returning, write `${LOG_DIR}/agent_issue_analyzer.md` with searches,
files inspected, and unresolved uncertainty. If `LOG_DIR` is empty, report
that the self-log was skipped.
