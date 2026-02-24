# Quick Start: Creating a TTNN Operation

This guide walks through the `/create-op` pipeline end-to-end.

## Pipeline at a Glance

```
Phase 0: Discovery     — find reference operations
Phase 1: Analysis      — deep-dive into references (parallel)
Phase 2: Planning      — produce functional spec (~250 lines)
Phase 3: Design→Build  — kernel designer THEN generic-op-builder (sequential)
Phase 4: TDD Kernels   — stage-gated kernel implementation
Phase 5: Report        — summary markdown
```

## Example: Creating a `row_centralize` Operation

### 1. Invoke the skill

```
/create-op row_centralize: subtract the row mean from each element.
  Input: row-major bfloat16, interleaved, rank 2-4.
  Output: same shape, row-major bfloat16.
  Formula: output[i,j] = input[i,j] - mean(input[i,:])
```

### 2. What happens behind the scenes

**Phase 0 — Discovery**: The orchestrator detects row-major input + compute + row-major output, so it selects three references:
- `tilize` (input stage) — converts RM sticks to tiles
- A reduction op like `reduce_w` (compute core) — row-wise reduction
- `untilize` (output stage) — converts tiles back to RM sticks

**Phase 1 — Analysis**: Runs `ttnn-operation-analyzer` on each reference in parallel.

**Phase 2 — Planning**: Runs `ttnn-operation-planner` to produce `row_centralize_spec.md` (~250 lines, no agent names, includes HW constraints checklist).

**Phase 3a — Kernel Designer** (runs first):
- Reads the spec and helper library headers
- Determines TDD stages using H1/H2 heuristics:

  | Stage | Name | What's Added |
  |-------|------|-------------|
  | 1 | `data_pipeline` | Reader + Writer + tilize/untilize passthrough |
  | 2 | `reduce_mean` | Add row-wise mean reduction |
  | 3 | `subtract_mean` | Subtract mean from input (full operation) |

- Registers all stages:
  ```bash
  python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py init row_centralize_spec.md --op-path ttnn/ttnn/operations/row_centralize
  python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py add-stage '{"name":"data_pipeline", ...}' --op-path ...
  python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py add-stage '{"name":"reduce_mean", ...}' --op-path ...
  python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py add-stage '{"name":"subtract_mean", ...}' --op-path ...
  ```
- Writes `kernel_design.md` with Part 1 (TDD Stage Plan) and Part 2 (implementation details)

**Phase 3b — Generic Op Builder** (runs after designer):
- Reads `.tdd_state.json` to discover the 3 registered stages
- Creates Python orchestration, program descriptor, stub kernels
- Writes tests to `tests/ttnn/unit_tests/operations/row_centralize/`

**Phase 4 — TDD Kernels**: For each stage in order:
1. Invoke `ttnn-kernel-writer` with stage-scoped prompt
2. Run `tdd_orchestrator.py test` (uses tt-test.sh --dev with watcher + hang detection)
3. On pass → `advance` + commit; on fail → parse failure, retry or rollback

**Phase 5 — Report**: Generates `REPORT.md` with pipeline execution summary.

### 3. What you get

**Operation directory** (`ttnn/ttnn/operations/row_centralize/`):
```
├── __init__.py
├── row_centralize.py                    # Entry point
├── row_centralize_program_descriptor.py # CB config, kernel setup
├── kernels/
│   ├── row_centralize_reader.cpp
│   ├── row_centralize_compute.cpp
│   └── row_centralize_writer.cpp
├── row_centralize_spec.md               # Functional spec
├── kernel_design.md                     # Kernel design doc
├── .tdd_state.json                      # TDD pipeline state
└── REPORT.md                            # Build report
```

**Test directory** (`tests/ttnn/unit_tests/operations/row_centralize/`):
```
├── test_row_centralize.py               # Integration test
├── test_stage_data_pipeline.py          # Stage 1 test
├── test_stage_reduce_mean.py            # Stage 2 test
└── test_stage_subtract_mean.py          # Stage 3 test
```

### 4. Running tests

```bash
# Run the integration test
pytest tests/ttnn/unit_tests/operations/row_centralize/test_row_centralize.py -v

# Run a specific stage test (with hang detection)
.claude/scripts/tt-test.sh --dev tests/ttnn/unit_tests/operations/row_centralize/test_stage_data_pipeline.py

# Check TDD pipeline status
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py status --op-path ttnn/ttnn/operations/row_centralize
```

## Key Design Decisions

### Why sequential Phase 3?

The kernel designer registers TDD stages in `.tdd_state.json`. The generic-op-builder reads that file to discover stages and generate test files. Running them in parallel meant the builder couldn't see the designer's stages.

### Why separate test directories?

Tests at `tests/ttnn/unit_tests/operations/{op_name}/` instead of colocated with the operation. This avoids pytest autodiscovery issues and follows the repo convention.

### Who determines TDD stages?

The **kernel designer** owns stage determination using two heuristics:
- **H1 (Kernel Complexity Ordering)**: Which kernel to finalize first
- **H2 (Semantic Goal Progression)**: How to break a kernel into testable milestones

The kernel writer implements exactly one assigned stage at a time. It is forbidden from implementing future stages.

### What does the planner spec look like?

~250 lines, two sections:
- **Section A**: API + Validation (parameters, input/output requirements, edge cases)
- **Section B**: CB Config + Data Flow (component sources, work distribution, CB table, kernel args, HW constraints checklist)

No agent names, no implementation strategies — just the contract.

## Automation Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| Interactive (default) | No keyword | Pauses at checkpoints for review |
| Automated | "FULLY AUTOMATED" | Skips review checkpoints |

Example automated invocation:
```
/create-op FULLY AUTOMATED: row_centralize — subtract row mean from each element.
  Input: row-major bfloat16 interleaved rank 2-4.
  Output: same shape row-major bfloat16.
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `.tdd_state.json` missing | Designer didn't run | Re-run Phase 3a (kernel designer) |
| No stage test files | Orchestrator registration failed | Check `tdd_orchestrator.py add-stage` output |
| Test hangs | CB sync mismatch in kernels | `pkill -9 -f pytest && tt-smi -r`, check watcher log |
| Builder can't find stages | Phase 3 ran in parallel | Ensure designer completes before builder starts |
