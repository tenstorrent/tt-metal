---
name: ttnn-kernel-writer-tdd
description: Execute the full stage-gated TDD loop for TTNN kernels in one continuous session.
argument-hint: "<operation_path>"
---

# TTNN Kernel Writer TDD (Codex)

Use this role to own Phase 4 end-to-end: implement, test, fix, and advance across all registered stages.

## Inputs

- `{op_path}/op_design.md`
- `{op_path}/.tdd_state.json`
- Kernel stubs and descriptor files

## Loop Contract

For each stage in order:
1. Inspect stage status with `tdd_orchestrator.py status`.
2. Implement only current-stage requirements, preferring existing helpers over raw/manual kernel logic.
3. Run `tdd_orchestrator.py test --op-path <path>`.
4. If pass: `advance`, commit, continue.
5. If fail: run `parse-failure`, apply focused fix, retry within budget.

If retry budget is exhausted, run rollback and produce `tdd_failure_report.md`.

## Output Contract

- Stage-by-stage pass/fail ledger
- Upstream fixes performed (if any)
- Final status: all stages passed OR human review required

## Guardrails

- Never skip stages.
- Never implement future stages early.
- Keep helper-first behavior across retries; only bypass helpers for explicit, documented helper gaps.
- Keep fixes minimal and traceable to the current failure.

## Legacy Mapping

This skill is the Codex-native runtime replacement for:
- `tt_metal/third_party/tt_ops_code_gen/agents/ttnn-kernel-writer-tdd.md`


## User Controls

Before execution, read `references/user-config.md` and apply user preferences for scope, depth, and output style.
