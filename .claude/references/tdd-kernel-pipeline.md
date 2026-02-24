# TDD Kernel Pipeline — Orchestrator Protocol

## PURPOSE

This reference enables the orchestrator agent to run a stage-gated TDD pipeline for kernel implementation. Each stage is tested independently before advancing.

## PREREQUISITES

Before using this pipeline, you MUST have:
1. An operation design document (`op_design.md`) in the operation directory
2. Stub kernel files created by the generic-op-builder
3. A working `__init__.py` that exports the operation function
4. A `.tdd_state.json` with pre-registered stages (from the architect)

## CLI TOOL

All commands go through:
```
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py <command> [args]
```

## INITIALIZATION PROTOCOL

Run these steps exactly, in order:

### Step 1: Initialize
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py init {op_path}/op_design.md --op-path {op_path}
```

### Step 2: Read the kernel design document
Read `{op_path}/op_design.md` to determine TDD stages. Stages follow this ordering heuristic:
- Stage 1: Data pipeline (reader + writer + passthrough compute)
- Stage 2: Bookend phases together (e.g., tilize + untilize as identity roundtrip)
- Stage 3+: Compute phases in pipeline order

### Step 3: Register all stages
For each stage, run `add-stage`. Register ALL stages before implementing any.

```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py add-stage '{
  "name": "{stage_name}",
  "description": "{what this stage adds}",
  "reference_body": "{python expression — receives input_tensor, return expected output}",
  "tolerance": {"rtol": {rtol}, "atol": {atol}},
  "shapes": ["{shape1}", "{shape2}", ...]
}' --op-path {op_path}
```

## STAGE LOOP PROTOCOL

For each stage, in order:

### 1. CHECK STATUS
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py status --op-path {op_path}
```

### 2. IMPLEMENT
Invoke kernel-writer with a stage-scoped prompt:

```
Implement TDD stage '{stage_name}' for {op_name}.
Design: {op_path}/op_design.md
Stage description: {stage_description}
Kernel files to modify: {kernel_files}
Previous failure: {failure_json or 'None — first attempt'}
IMPORTANT: Only implement what this stage requires. Do not implement future stages.
```

### 3. TEST
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py test --op-path {op_path}
```

### 4. BRANCH ON RESULT

**If PASS (exit 0):**
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py advance --op-path {op_path}
git add {op_path}/kernels/ && git commit -m "[kw-tdd] stage {stage_name} passed"
```
Continue to next stage.

**If FAIL (exit 1 or 2):**
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py parse-failure --op-path {op_path}
```
Read the structured JSON output.

  - If `remaining_attempts > 0`:
    Go to step 2. Include the failure JSON in the kernel-writer prompt as `Previous failure: {json}`.

  - If `remaining_attempts == 0`:
    ```bash
    python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py rollback --op-path {op_path}
    ```
    STOP. Report: `HUMAN REVIEW REQUIRED — see {op_path}/tdd_failure_report.md`

## STAGE REGISTRATION RULES

### Required JSON fields
- `name`: snake_case string — becomes the test filename (`test_stage_{name}.py`)
- `description`: Human-readable — goes into test docstring
- `reference_body`: Python expression that receives `input_tensor` and returns expected output
- `tolerance`: `{"rtol": float, "atol": float}` — comparison tolerance
- `shapes`: List of shape strings like `"(1, 1, 32, 64)"`

### How to write `reference_body`
The body becomes the return statement of `pytorch_reference(input_tensor)`:

```python
# For passthrough:
"return input_tensor.clone()"

# For mean subtraction:
"return input_tensor - input_tensor.mean(dim=-1, keepdim=True)"

# For variance:
"mean = input_tensor.mean(dim=-1, keepdim=True)\n    return ((input_tensor - mean) ** 2).mean(dim=-1, keepdim=True)"
```

### How to set tolerances
- **Passthrough/identity stages**: `{"rtol": 0.01, "atol": 0.01}`
- **Simple compute (add, sub, mul)**: `{"rtol": 0.01, "atol": 0.05}`
- **Reduce operations**: `{"rtol": 0.02, "atol": 0.1}`
- **Multi-step compute (normalize)**: `{"rtol": 0.05, "atol": 0.2}`

### Optional fields
- `extra_imports`: Additional import lines (e.g., `"import math"`)
- `extra_args`: Appended to op call (e.g., `", gamma, beta"`)
- `extra_setup`: Python code for extra tensor setup before the op call
- `extra_ttnn_setup`: Python code for extra TTNN setup after input creation
- `output_shape_expr`: Python expression for expected output shape if different from input (e.g., `"list(shape[:-1]) + [1]"`)
- `kernel_files`: List of kernel files this stage modifies (informational, passed to kernel-writer)

## FAILURE HANDLING

### Reading parse-failure output
The JSON contains:
- `classification`: One of `numerical_mismatch`, `compilation_error`, `shape_mismatch`, `runtime_error`, `watcher_assert`, `hang_cb_deadlock`, `hang_noc_error`, `hang_unknown`
- `summary`: One-line description of what went wrong
- `suggested_action`: What to try on the next attempt
- `remaining_attempts`: How many retries are left

### Feeding failure context to kernel-writer
Add to the kernel-writer prompt:
```
Previous failure (attempt {N}):
  Classification: {classification}
  Summary: {summary}
  Suggested action: {suggested_action}
  Details: {details}
```

### When rollback triggers
Rollback happens when `remaining_attempts == 0`. It:
1. Restores kernel files to the last passing commit
2. Marks the stage as `failed_permanent`
3. Writes `tdd_failure_report.md` with the full failure history

### What to tell the user on `failed_permanent`
```
HUMAN REVIEW REQUIRED — Stage '{name}' failed after {N} attempts.
See: {op_path}/tdd_failure_report.md

The failure report contains all {N} attempts with classifications and suggested fixes.
Kernel files have been rolled back to the last passing commit ({commit}).
```

## ANTI-PATTERNS

### DON'T skip `advance` after a pass
The gate marker is removed by `advance`. If you skip it, the next `test` will work but you won't be able to commit (pre-commit hook checks the marker for the NEXT stage).

### DON'T register stages after the pipeline has started
All stages must be registered via `add-stage` before the first `test`. Adding stages mid-pipeline will cause index confusion.

### DON'T test a stage before the kernel-writer has returned
The kernel files must be written before testing. Running `test` on unchanged stubs wastes an attempt.

### DON'T commit without the gate marker
The pre-commit hook checks for `.tdd_gate_passed`. If you deleted it manually, re-run `test` to regenerate it.

### DON'T implement future stages
Each kernel-writer invocation must ONLY implement what the current stage requires. Implementing ahead means untested code that may break in subtle ways.

## INTEGRATION WITH KERNEL-WRITER

### Prompt template for per-stage invocation
```
Implement TDD stage '{stage_name}' for {op_name}.
Design: {op_path}/op_design.md

Stage description: {stage_description}
Kernel files to modify: {kernel_files}
Previous failure: {failure_json or 'None — first attempt'}

IMPORTANT: Only implement what this stage requires. Do not implement future stages.
Previous stages already pass — do not modify their behavior.
```

### How the kernel-writer should scope changes
- **Stage 1 (data pipeline)**: Implement reader, writer, and passthrough compute (copy_tile)
- **Bookend stages**: Add tilize in reader/compute, untilize in compute/writer — both at once for identity verification
- **Compute stages**: Modify only the compute kernel to add the new phase. Reader/writer should already be correct from earlier stages.

### Retry with failure context
On retry, append failure info to the prompt:
```
RETRY — Previous attempt failed:
  Classification: {classification}
  Summary: {summary}
  Action: {suggested_action}

Fix the issue described above. Do not change parts that were working.
```
