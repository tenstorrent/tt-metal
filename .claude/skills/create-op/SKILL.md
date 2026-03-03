---
name: create-op
description: Full pipeline for creating TTNN operations using generic_op workflow with TDD kernel implementation. Orchestrates discovery, analysis, design, build, TDD kernels, and reporting. Args = operation requirements.
---

# TTNN Operation Creation Pipeline

End-to-end workflow for creating new TTNN operations using the Python-based generic_op infrastructure with stage-gated TDD kernel implementation.

## Overview

```
Phase 0: Discovery ─► Phase 1: Analysis ─► Phase 2: Design
                      (parallel analyzers)    (architect)

─► Phase 3: Build ─► Phase 4: TDD Kernels ─► Phase 5: Report
   (generic-op-builder)  (stage-gated loop)
```

---

## Spec Detection

Before parsing input, check if a spec already exists:

1. Extract the operation name from the user's message (snake_case identifier)
2. Look for `ttnn/ttnn/operations/{op_name}/op_spec.md`
3. **If found**:
   - Read it and use its contents as the operation requirements
   - Extract all fields (math, tensors, parameters, test criteria, hardware preferences)
   - Skip interactive questions for information already in the spec
   - Phase 0 still runs for reference discovery (unless references are specified in the spec)
   - Pass spec contents to the architect in Phase 2 as structured requirements
   - Log: "Using existing op_spec.md from ttnn/ttnn/operations/{op_name}/op_spec.md"
4. **If not found**:
   - Proceed with normal input parsing below

---

## Input Parsing

Extract from the user's message (or from op_spec.md if detected above):
1. **Operation name**: snake_case identifier (e.g., `row_centralize`)
2. **Math definition**: The formula or algorithm to implement
3. **Input tensor**: Layout (RM/tile), memory layout (interleaved/sharded), dtype, rank
4. **Output tensor**: Shape relationship to input, layout, dtype
5. **Parameters**: Additional op parameters (e.g., epsilon) with types and defaults
6. **Reference operations**: If the user specifies any (paths or names)

If any critical information is missing and the mode is interactive, ask. If automated, make reasonable assumptions and document them.

---

## Automation Mode

Two modes, determined by the user's message:

| Mode | Trigger | Behavior |
|------|---------|----------|
| **Interactive** (default) | No special keyword | Pause at checkpoints for user review |
| **Automated** | User says "FULLY AUTOMATED" or "no confirmations" | Skip all review checkpoints, proceed with best judgment |

Checkpoints affected by mode:
- Phase 0: Reference confirmation
- Phase 2: Design approval
- Phase 3: Builder output review

---

## Logging Setup

If the user requests breadcrumbs, logging, or tracing:

```bash
touch .claude/active_logging
```

A hook automatically injects breadcrumb instructions into every subagent's context. No need to mention logging in agent prompts. Logs appear at `{op_path}/agent_logs/`.

---

## Canonical Paths

Operation code lives at:
```
ttnn/ttnn/operations/{operation_name}/
```

Tests live at:
```
tests/ttnn/unit_tests/operations/{operation_name}/
```

### Operation directory:
```
ttnn/ttnn/operations/{op_name}/
├── __init__.py                             # Re-export main function
├── {op_name}.py                            # Entry point with validation
├── {op_name}_program_descriptor.py         # CB config, work distribution, kernel setup
├── kernels/
│   ├── {op_name}_reader.cpp
│   ├── {op_name}_compute.cpp
│   └── {op_name}_writer.cpp
├── op_design.md                            # Operation design doc (Phase 2)
├── .tdd_state.json                         # TDD pipeline state
├── REPORT.md                               # Build report (Phase 5)
└── agent_logs/                             # Breadcrumbs (if logging enabled)
```

### Test directory:
```
tests/ttnn/unit_tests/operations/{op_name}/
├── test_{op_name}.py                       # Integration test (Phase 3)
└── test_stage_*.py                         # TDD stage tests (registered by architect)
```

---

## Phase 0: Discovery

**Goal**: Identify reference operations for analysis.

### Compute Detection

ALL compute operations (FPU/SFPU) require tilized data. Check:

| Input Layout | Has Compute? | Output Layout | References Needed |
|-------------|-------------|---------------|-------------------|
| Row-major | Yes | Row-major | tilize (input) + compute ref + untilize (output) |
| Row-major | Yes | Tile | tilize (input) + compute ref |
| Tile | Yes | Row-major | compute ref + untilize (output) |
| Tile | Yes | Tile | compute ref only |
| Row-major | No (data movement only) | Row-major | data movement ref |

The pattern for RM + compute + RM: `RM input → read sticks → tilize → compute (tiles) → untilize → write sticks → RM output`

### Reference Selection

1. Parse user requirements for format keywords:
   - "row-major input" + any compute → need tilize reference
   - Any compute + "row-major output" → need untilize reference
   - "sharded" → need sharded-input reference
2. Select appropriate variants: prefer simpler (single_core) for initial implementation
3. For the compute reference: find an existing operation with similar math (reduction, eltwise, etc.)

### Mode Determination

- Single reference → Derivative mode
- Multiple references with different roles → Hybrid mode

### Interactive Checkpoint

Present discovered references as a table:

```
| Role | Operation | Path | Reason |
|------|-----------|------|--------|
| input_stage | tilize | .../tilize_multi_core_interleaved_program_factory.cpp | RM input |
| output_stage | untilize | .../untilize_multi_core_program_factory.cpp | RM output |
| compute_core | {op} | .../{op}_program_factory.cpp | similar compute pattern |

Planning Mode: Hybrid
```

In automated mode, proceed directly.

---

## Phase 1: Analysis

**Goal**: Deep architectural analysis of reference operations.

Launch `ttnn-operation-analyzer` on EACH reference operation. Run all analyzers **in parallel** (multiple Task calls in a single message).

```
Task: ttnn-operation-analyzer
  Input: program factory path
  Output: {ref_dir}/{ref_name}_analysis.md
```

Wait for ALL analyzers to complete before proceeding.

---

## Phase 2: Design

**Goal**: Produce a complete operation design — architecture + kernel implementation strategy + TDD stages.

Launch `ttnn-operation-architect` with:
- All analyzer output paths
- Operation requirements (name, math definition, tensor requirements, parameters)
- Mode (derivative/hybrid) and role assignments
- Target path: `{op_path}/op_design.md`

```
Task: ttnn-operation-architect
  Input: analyzer outputs + requirements
  Output: {op_path}/op_design.md + .tdd_state.json (with registered stages)
```

The architect:
1. **Pass 1 (Architecture)**: Defines CB layout, work distribution, data flow, tensor requirements
2. **Pass 2 (Implementation)**: Maps phases to helpers, validates architecture against helper requirements, determines TDD stages
3. Registers ALL TDD stages via `tdd_orchestrator.py add-stage`

### Interactive Checkpoint

Present the design summary: API, CB layout, helper decisions, TDD stages. User approves or requests changes.

In automated mode, proceed directly.

---

## Phase 3: Build

**Goal**: Create Python infrastructure with stub kernels.

Launch `ttnn-generic-op-builder` with:
- `{op_path}/op_design.md` (reads Part 1 for architecture)
- `.tdd_state.json` (reads to discover registered stages)

```
Task: ttnn-generic-op-builder
  Input: {op_path}/op_design.md + .tdd_state.json
  Output: Python files + stub kernels + integration test
```

The builder:
1. Reads `.tdd_state.json` to discover registered stages
2. Creates Python orchestration, program descriptor, stub kernels
3. Writes integration test to `tests/ttnn/unit_tests/operations/{op_name}/`
4. Verifies stage test files exist (generated by orchestrator during architect's registration)

### Interactive Checkpoint

Present builder outputs:
- Generic op structure (CBs, runtime args, kernel paths)
- Test validation results

In automated mode, proceed directly.

---

## Phase 4: TDD Kernels

**Goal**: Implement kernels incrementally with test verification.

Launch a single `ttnn-kernel-writer-tdd` agent. This agent owns the full TDD loop — it iterates through all stages internally, retaining context across stages.

```
Task: ttnn-kernel-writer-tdd
  Prompt: |
    Implement all TDD stages for {op_name}.
    Operation path: {op_path}

    Follow the TDD loop exactly: implement stage → test → fix → advance → commit → next stage.
    Do NOT skip stages. Do NOT implement ahead.
```

Do NOT:
- Loop over stages yourself
- Spawn multiple kernel-writer agents
- Call tdd_orchestrator.py yourself

The agent handles all of this. It returns a structured report with per-stage results, upstream fixes, and design deviations.

**If the agent reports `HUMAN REVIEW REQUIRED`**: A stage exhausted its retry budget. Relay the failure report path to the user.

---

## Phase 5: Reporting

**Goal**: Produce a structured markdown report.

Generate `{op_path}/REPORT.md` containing:

1. **Summary**: Operation name, what it does, overall result
2. **Pipeline execution**: Table of phases with agents, durations, outputs
3. **Agent summaries**: Key findings/decisions from each agent
4. **TDD pipeline results**: Table of stages with pass/fail, attempt counts, failure classifications
5. **Files produced**: Directory listing
6. **Git history**: Relevant commits
7. **Decisions and deviations**: Assumptions made, deviations from design, pain points

Commit the report.

---

## Critical Rules

These rules apply across all phases. Violating them causes subtle bugs.

### Scaler Packing Format
`generate_reduce_scaler()` and `generate_bcast_scalar_bfloat16()` expect `(bf16 << 16 | bf16)` format — two bf16 values packed into a uint32. NOT IEEE 754 float32. The program descriptor handles this conversion.

### Compute Kernel Syntax
Use `void kernel_main() {}`. NOT the deprecated `namespace NAMESPACE { void MAIN {} }`.

### CB Index Convention
- 0-7: inputs
- 8-15: special (scalers, constants)
- 16-23: outputs
- 24-31: intermediates

### Testing
- Use `tt-test.sh --dev` for TDD stages (real kernels can hang — needs watcher + timeout)
- Use plain `pytest` for builder phases (stub kernels can't hang)
- If pytest hangs: `pkill -9 -f pytest || true` then `tt-smi -r`

### allocate_tensor_on_device
Use positional args, not keyword args.

### NoWaitNoPop Policy
When a helper uses `NoWaitNoPop`, it does NOT pop the input CB. The caller MUST manually `cb_pop_front()` after, or use a subsequent helper with `NoWaitPopAtEnd` that handles it.

---

## Subagent Reference

| Agent | Type | Purpose |
|-------|------|---------|
| `ttnn-operation-analyzer` | Opus | Deep analysis of existing operations |
| `ttnn-operation-architect` | Opus | Design new operation (architecture + kernel implementation) |
| `ttnn-generic-op-builder` | Sonnet | Python infrastructure + stub kernels |
| `ttnn-kernel-writer-tdd` | Opus | Implement all TDD stages in one session (owns full loop) |
| `ttnn-riscv-debugger` | Sonnet | Debug kernel issues (hangs, wrong output) |

### Dependency Graph
```
analyzer(s) ──► architect ──► op_design.md + .tdd_state.json
                                      │
                                      ▼
                    generic_op_builder ──► stubs + Python + tests
                                      │
                                      ▼
                    kernel_writer_tdd (single agent, all stages)
```

### When to use the debugger

Invoke `ttnn-riscv-debugger` when:
- The kernel-writer-tdd agent reports `HUMAN REVIEW REQUIRED` after exhausting retry budget
- The failure suggests a CB synchronization or semaphore bug that the kernel writer couldn't resolve

Provide: symptom description, test command, paths to kernels and analysis files.
