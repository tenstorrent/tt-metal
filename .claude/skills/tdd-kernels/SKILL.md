---
name: tdd-kernels
description: Stage-gated TDD pipeline for TTNN kernel implementation. Use after operation-architect and generic-op-builder have produced a design document and stubs. Invoked by /create-op at Phase 4, or standalone. Args = operation path.
---

# TDD Kernel Pipeline

Stage-gated test-driven development for TTNN kernels. Each stage is tested independently before advancing.

## Prerequisites

Before using this pipeline, the operation directory MUST contain:
1. An operation design document (`op_design.md`) — from the architect
2. Stub kernel files from the generic-op-builder (reader, compute, writer)
3. A working `__init__.py` that exports the operation function
4. A program descriptor that configures CBs and kernel args
5. A `.tdd_state.json` with pre-registered stages (from the architect)

## CLI Reference

All commands go through:
```
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py <command> [args]
```

| Command | Purpose |
|---------|---------|
| `init <spec_path> --op-path <path>` | Initialize pipeline, create state file |
| `add-stage '<json>' --op-path <path>` | Register a stage (must be done before any testing) |
| `status --op-path <path>` | Show pipeline state: stages, pass/fail, current stage |
| `test --op-path <path>` | Run current stage's test (uses dev-test.sh with watcher + hang detection) |
| `advance --op-path <path>` | Mark current stage as passed, move to next |
| `parse-failure --op-path <path>` | Parse test failure into structured JSON |
| `rollback --op-path <path>` | Restore kernels to last passing commit (on max retries) |

---

## Stage Discovery

**Stages are pre-registered by the operation architect.** The architect has already:
1. Applied H1/H2 heuristics to determine stage ordering and granularity
2. Registered all stages via `tdd_orchestrator.py add-stage`
3. Documented stages in `op_design.md` Part 2 (TDD Stage Plan)

Verify stages exist:
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py status --op-path <path>
```

If `.tdd_state.json` is missing or has no stages, the architect phase did not complete — go back and run it.

Test files are located at `tests/ttnn/unit_tests/operations/{op_name}/test_stage_*.py`.

---

## Stage Loop Protocol

For each registered stage, in order:

### 1. Check Status
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py status --op-path <path>
```

### 2. Implement — Invoke Kernel Writer

Launch `ttnn-kernel-writer` with a **stage-scoped** prompt.

**Agent reuse rule**: Track the kernel-writer's `total_tokens` from the task notification after each invocation.
- **< 160k tokens**: Resume the same agent (pass `resume: {agentId}`). Faster — agent already has design doc, kernel files, and prior work in context.
- **>= 160k tokens**: Spawn a fresh agent. Context is getting heavy and risks compaction.

This applies across both retries and new stages. The agent only knows what the current prompt tells it — scope is controlled by the prompt, not by spawning fresh.

**Prompt template** (used for both fresh and resumed):

```
Implement TDD stage '{stage_name}' for {op_name}.
Design: {op_path}/op_design.md

Stage description: {stage_description}
Kernel files to modify: {kernel_files}
Previous failure: {failure_json or 'None — first attempt'}

IMPORTANT: Only implement what this stage requires. Do not implement future stages.
Previous stages already pass — do not modify their behavior.
```

For Stage 1 (data pipeline): Specify which kernels need full implementation vs. minimum viable, and what the compute passthrough should do (e.g., "tilize then immediately untilize" for RM I/O operations).

### 3. Test
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py test --op-path <path>
```

This runs the stage test with `dev-test.sh` (watcher enabled, hang detection at 5s timeout, device reset on failure).

### 4. Branch on Result

**PASS (exit 0)**:
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py advance --op-path <path>
git add {op_path}/kernels/ && git commit -m "[kw-tdd] {op_name}: stage {stage_name} passed"
```
Continue to next stage.

**FAIL (exit 1 or 2)**:
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py parse-failure --op-path <path>
```
Read the structured JSON. It includes:
- `classification`: failure type (see retry policy below)
- `summary`: one-line description
- `suggested_action`: what to try next
- `cost`: `FREE` or `HARD`
- `remaining_attempts`: hard retries left
- `budget_exhausted`: whether to rollback

**Retry Policy — Classification Escalation**:

| Classification | Cost | Rationale |
|---|---|---|
| `compilation_error` | FREE | Typos, missing includes — trivially fixable |
| `shape_mismatch` | FREE | Wrong CB count, bad allocation — trivially fixable |
| `runtime_error` | HARD (1) | Needs algorithmic understanding |
| `numerical_mismatch` | HARD (1) | Needs parameter/logic debugging |
| `watcher_assert` | HARD (1) | Hardware-level issue |
| `hang_cb_deadlock` | HARD (1) | CB synchronization bug |
| `hang_noc_error` | HARD (1) | DMA address issue |
| `hang_unknown` | HARD (1) | Unclear root cause |

- **Hard budget**: 6 attempts per stage (only HARD failures consume this)
- **Free safety cap**: 10 retries (prevents infinite loops on repeated easy errors)
- A stage is exhausted when `budget_exhausted == true`

**If NOT exhausted**: go to step 2, include failure JSON in the kernel-writer prompt:
```
RETRY — Previous attempt failed:
  Classification: {classification} (cost: {cost})
  Summary: {summary}
  Action: {suggested_action}

Fix the issue described above. Do not change parts that were working.
```

**If exhausted**:
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py rollback --op-path <path>
```
STOP. Report: `HUMAN REVIEW REQUIRED — see {op_path}/tdd_failure_report.md`

---

## Anti-Patterns

| Anti-Pattern | Why It's Wrong | What To Do Instead |
|--------------|----------------|-------------------|
| Skip `advance` after a pass | Gate marker not cleared, next stage breaks | Always advance before moving on |
| Register stages after pipeline started | Index confusion in state file | Register ALL stages before first `test` |
| Test before kernel-writer returns | Wastes an attempt on unchanged stubs | Wait for kernel-writer to complete |
| Implement future stages early | Untested code that may break subtly | Each stage implements ONLY its scope |
| Commit without gate marker | Pre-commit hook will reject | Re-run `test` to regenerate marker |
| Change multiple kernels without considering impact | May break earlier stages | Only modify kernels listed for the current stage |
