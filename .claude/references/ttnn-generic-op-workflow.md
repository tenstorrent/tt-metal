# TTNN Generic Operation Creation Workflow

This reference contains the mandatory routing and workflow for creating new TTNN operations using the Python-based generic_op infrastructure. It is intended for the orchestrator agent only.

## Overview

This workflow uses the **generic_op** infrastructure, producing Python orchestration code, ProgramDescriptor APIs, and stub kernel files. This is ideal for:
- Rapid prototyping without CMake/nanobind overhead
- Custom operations that don't require C++ TTNN registration
- Operations using `ttnn.generic_op()` and ProgramDescriptor

## Canonical Operation Path (SOURCE OF TRUTH)

All generic_op operations MUST be created at:

```
ttnn/ttnn/operations/{operation_name}/
```

This is the **single source of truth** for operation location. All agents in this pipeline read this path from here.

This places operations within the `ttnn` package, enabling direct imports:
```python
from ttnn.operations.<op_name> import <op_name>
```

**Operation directory**:
```
ttnn/ttnn/operations/{operation_name}/
├── __init__.py                             # Re-export main function
├── {operation_name}.py                     # Entry point with output allocation
├── {operation_name}_program_descriptor.py  # CB config, work distribution, kernel setup
├── kernels/
│   ├── {operation_name}_reader.cpp         # Data movement: DRAM → L1
│   ├── {operation_name}_compute.cpp        # FPU/SFPU operations
│   └── {operation_name}_writer.cpp         # Data movement: L1 → DRAM
├── agent_logs/                             # Execution logs (if logging enabled)
│   ├── {agent_name}_breadcrumbs.jsonl
│   └── {agent_name}_execution_log.md
├── op_design.md                            # Operation design doc (from architect)
└── .tdd_state.json                         # TDD pipeline state
```

**Test directory**:
```
tests/ttnn/unit_tests/operations/{operation_name}/
├── test_{operation_name}.py                # Integration test
└── test_stage_*.py                         # TDD stage tests
```

**Example**: For an operation named `row_centralize`:
```
ttnn/ttnn/operations/row_centralize/
├── __init__.py
├── row_centralize.py
├── row_centralize_program_descriptor.py
├── kernels/
│   ├── row_centralize_reader.cpp
│   ├── row_centralize_compute.cpp
│   └── row_centralize_writer.cpp
├── op_design.md
└── .tdd_state.json

tests/ttnn/unit_tests/operations/row_centralize/
├── test_row_centralize.py
└── test_stage_*.py
```

**Running tests**:
```bash
pytest tests/ttnn/unit_tests/operations/row_centralize/test_row_centralize.py -v
```

## Pipeline Structure

```
analyzer → architect → generic_op_builder → kernel_writer
           (+ TDD stages)  (reads .tdd_state)   (per stage)
```

**Key sequencing**: The `architect` runs first (determines TDD stages and registers them), then `generic_op_builder` runs (reads `.tdd_state.json` to discover stages). The `kernel_writer` is invoked per TDD stage after both are complete.

---

## Mandatory Routing

When user requests a new TTNN operation via generic_op, STOP and answer these questions:

### Step 1: Are reference operations specified?
- YES with paths to reference_operation_analysis.md → Skip to Phase 1 (Analyzer)
- YES but vague ("like softmax") → Search for that operation's program_factory.cpp
- NO → Continue to discovery

### Step 2: Discovery Checklist (if references not specified)

**⚠️ CRITICAL: COMPUTE REQUIRES TILES**
All compute operations (FPU/SFPU) require tilized data. Even if BOTH input AND output are row-major:
- Row-major input → MUST tilize before compute
- Compute operates on 32×32 tiles ONLY
- Row-major output → MUST untilize after compute

Pattern: `RM input → read sticks → tilize → compute (tiles) → untilize → write sticks → RM output`

□ **First: Determine if operation has compute**:
  - ANY math operation (reduction, eltwise, matrix ops) → REQUIRES tilized data
  - Row-major input + compute → MUST include tilize reference (Hybrid Mode)
  - Compute + row-major output → MUST include untilize reference (Hybrid Mode)
  - Row-major input + compute + row-major output → Hybrid Mode with 3 references:
    1. tilize (input_stage)
    2. compute operation (compute_core)
    3. untilize (output_stage)

□ Parse for format keywords:
  - "row-major input" + ANY compute → need tilize reference
  - ANY compute + "row-major output" → need untilize reference
  - "sharded" → need sharded-input reference (layernorm, etc.)

□ Select appropriate variant:
  - Match memory layout: interleaved → *_interleaved_*, sharded → *_sharded_*
  - Prefer simpler variant (single_core) for templates

□ Query DeepWiki for unknowns:
  - "Which TTNN operations perform [X]?"
  - "Which operations convert ROW_MAJOR to TILE_LAYOUT?"

### Step 3: Mode Determination
- Single reference → Derivative mode
- Multiple references with different roles → Hybrid mode

### Step 4: Reference Confirmation (USER CHECKPOINT)

Before running analyzers, present discovered references:

"I identified these references:
| Role | Operation | Path | Reason |
|------|-----------|------|--------|
| input_stage | tilize | .../tilize_multi_core_interleaved_program_factory.cpp | row-major + tilize keywords |
| output_stage | untilize | .../untilize_multi_core_program_factory.cpp | untilize + row-major keywords |

Planning Mode: Hybrid

Proceed with analysis, or suggest different references?"

- User confirms → proceed to Phase 1
- User suggests alternatives → update references and re-confirm

### Step 5: Enable Logging (If Requested)

If the user requests breadcrumbs or logging, check that the signal file exists:

```bash
ls .claude/active_logging 2>/dev/null && echo "Logging ON" || echo "Logging OFF"
```

If logging is requested but the file doesn't exist, create it:
```bash
touch .claude/active_logging
```

A `SubagentStart` hook automatically injects breadcrumb instructions into every agent's context. No need to mention logging in agent prompts. See `.claude/references/logging-mechanism.md` for details.

### Step 6: Execute Workflow

1. **Phase 1**: Run `ttnn-operation-analyzer` on EACH confirmed reference
2. **Phase 2**: Run `ttnn-operation-architect` with all analyzer outputs
   - Produces: `op_design.md` (architecture + kernel implementation + helper mapping)
   - Registers TDD stages in `.tdd_state.json` via `tdd_orchestrator.py add-stage`
3. **USER REVIEW** (MANDATORY): Present the generated `op_design.md` to the user
   - User approves → proceed to Phase 3
   - User requests changes → refine, re-present for approval
   - Do NOT proceed without explicit user approval
4. **Phase 3** — Run `ttnn-generic-op-builder` with `op_design.md`:
   - Reads `.tdd_state.json` to discover registered stages
   - Produces: Python orchestration, ProgramDescriptor, stub kernels
   - Writes tests to `tests/ttnn/unit_tests/operations/{op_name}/`
5. **USER REVIEW**: Present builder outputs
6. **Phase 4**: Run `ttnn-kernel-writer` per TDD stage (stages are pre-registered)

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  User provides: New operation requirements                                   │
│  (may or may not specify reference operations)                              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
          ┌────────────────────────┴────────────────────────┐
          │                                                 │
          ▼ (refs not specified)                            ▼ (refs specified)
┌─────────────────────────┐                                 │
│ Phase 0: Discovery      │                                 │
│ (Orchestrator)          │                                 │
│ - Query DeepWiki        │                                 │
│ - Search codebase       │                                 │
│ - Select candidates     │                                 │
└───────────┬─────────────┘                                 │
            │                                               │
            ▼                                               │
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Analyzer(s)                                                         │
│ (Opus)                                                                       │
│                                                                              │
│  Run ttnn-operation-analyzer on each reference operation                     │
│  Output: {operation}_analysis.md for each reference                          │
└──────────────────────────────────────────────────┬──────────────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Architect                                                           │
│ (Opus)                                                                       │
│                                                                              │
│  Run ttnn-operation-architect with all analyzer outputs + helper headers      │
│  Output: op_design.md + .tdd_state.json (with registered stages)             │
└──────────────────────────────────────────────────┬──────────────────────────┘
                                                   │
                                         ▼ [USER REVIEW DESIGN]
                                                   │
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Generic Op Builder                                                  │
│ (Opus)                                                                       │
│                                                                              │
│  Input: op_design.md + .tdd_state.json                                       │
│  Output: Python orchestration, ProgramDescriptor, stub kernels, tests        │
└──────────────────────────────────────────────────┬──────────────────────────┘
                              │
                    ▼ [USER REVIEW BUILDER OUTPUT]
                              │
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Kernel Writer (per TDD stage)                                       │
│ (Opus)                                                                       │
│                                                                              │
│  Input:                                                                      │
│  - op_design.md Part 2 (from architect, with TDD Stage Plan)                 │
│  - Stub kernels (from generic_op_builder)                                    │
│  - Pre-registered stages in .tdd_state.json                                  │
│                                                                              │
│  Output:                                                                     │
│  - Working reader, compute, writer kernels                                   │
│  - Following design's USE HELPER / NO HELPER guidance                        │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼ [TEST: E2E passes]
                              │
┌─────────────────────────────────────────────────────────────────────────────┐
│ Done!                                                                        │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────────────────┐
     │ ttnn-riscv-debugger      │ ◄── Can be invoked at any
     │ (Sonnet, hypothesis-     │     phase when kernel issues
     │  driven, reverts code)   │     arise (hangs, wrong output)
     └──────────────────────────┘
```

---

## Agent Responsibilities

### ttnn-operation-analyzer
**Purpose**: Deep architectural analysis of existing TTNN operations as references.

**Input**: Path to reference operation's program factory

**Output**: `{operation_name}_analysis.md` containing:
- Work unit definition
- Data flow pattern
- Circular buffer configuration
- Memory access patterns
- Core distribution strategy

### ttnn-operation-architect
**Purpose**: Design the new operation end-to-end — architecture + kernel implementation strategy.

**Input**: Analyzer outputs + requirements + kernel helper library headers

**Output**: `op_design.md` containing:
- Part 1: Architecture (API, CB layout, work distribution, data flow, test criteria)
- Part 2: Kernel Implementation (helper mappings, TDD stages, per-phase details)
- Also registers TDD stages in `.tdd_state.json`

### ttnn-generic-op-builder
**Purpose**: Create Python-based operation infrastructure using generic_op.

**Input**: `op_design.md` (Part 1) + `.tdd_state.json`

**Output**:
- Python orchestration code using `ttnn.generic_op()`
- ProgramDescriptor configuration
- Stub kernel files (reader, compute, writer)
- CB configuration in Python

**Key Feature**: Python-based for rapid iteration.

### ttnn-kernel-writer
**Purpose**: Implement kernels following the Operation Design Document.

**Input**:
- Operation Design Document (`op_design.md` Part 2) - MANDATORY
- Stub kernels from generic_op_builder

**Output**: Working kernels that:
- Call helpers for phases marked "USE HELPER"
- Use raw calls ONLY for phases marked "NO HELPER"
- **MUST NOT** add raw CB operations around helper calls

---

## Sequential Execution Requirements

### Phase 2 → Phase 3 Sequencing

The architect (Phase 2) must complete before the builder (Phase 3) starts. The architect registers TDD stages in `.tdd_state.json` that the builder reads.

### Dependency Graph

```
analyzer_1 ─┐
analyzer_2 ─┼──► architect ──► op_design.md + .tdd_state.json
analyzer_n ─┘                          │
                                       ▼
                    generic_op_builder ──► stubs + Python + tests
                                       │
                                       ▼
                    kernel_writer (invoked per TDD stage)
```

---

## User Checkpoints

| Checkpoint | Location | What to Review |
|------------|----------|----------------|
| Reference Selection | After Step 4 | Confirm reference operations are appropriate |
| Design Approval | After Phase 2 | Review architecture, helpers, TDD stages |
| Builder Output Review | After Phase 3 | Review generic_op structure, CB config, stubs |
| Final Validation | After Phase 4 | E2E test passes with correct values |

---

## Troubleshooting

### Kernel Writer Dependency Issues

If kernel_writer cannot find required inputs:
- Verify `op_design.md` exists from architect
- Verify stub kernels exist from generic_op_builder
- Check paths match expected locations

### Runtime Errors After Kernel Writing

Invoke `ttnn-riscv-debugger` with:
- Symptom description
- Test command
- Paths to kernels and generic_op code

---

## Additional Resources

- https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/adding_new_ttnn_operation.html - Official docs
