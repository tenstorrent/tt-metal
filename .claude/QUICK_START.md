# Quick Start: Creating a TTNN Operation

This guide walks through the `/create-op` pipeline end-to-end.

## Pipeline at a Glance

```
Phase 0: Discovery     — find reference operations
Phase 1: Analysis      — deep-dive into references (parallel)
Phase 2: Design        — architect produces op_design.md + registers TDD stages
Phase 3: Build         — generic-op-builder creates Python infra + stubs
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

**Phase 2 — Design**: Runs `ttnn-operation-architect` to produce `op_design.md`:
- **Pass 1 (Architecture)**: Defines CB layout, work distribution, tensor requirements
- **Pass 2 (Implementation)**: Reads helper library headers, maps phases to helpers, validates architecture against helper requirements
- Determines TDD stages using H1/H2 heuristics:

  | Stage | Name | What's Added |
  |-------|------|-------------|
  | 1 | `data_pipeline` | Reader + Writer + tilize/untilize passthrough |
  | 2 | `reduce_mean` | Add row-wise mean reduction |
  | 3 | `subtract_mean` | Subtract mean from input (full operation) |

- Writes `op_design.md` with Part 1 (Architecture) and Part 2 (Kernel Implementation)
- Then registers all TDD stages:
  ```bash
  python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py init op_design.md --op-path ttnn/ttnn/operations/row_centralize
  python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py add-stage '{"name":"data_pipeline", ...}' --op-path ...
  python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py add-stage '{"name":"reduce_mean", ...}' --op-path ...
  python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py add-stage '{"name":"subtract_mean", ...}' --op-path ...
  ```

**Phase 3 — Build** (runs after architect):
- Reads `.tdd_state.json` to discover the 3 registered stages
- Creates Python orchestration, program descriptor, stub kernels
- Writes tests to `tests/ttnn/unit_tests/operations/row_centralize/`

**Phase 4 — TDD Kernels**: For each stage in order:
1. Invoke `ttnn-kernel-writer` with stage-scoped prompt
2. Run `tdd_orchestrator.py test` (uses scripts/tt-test.sh --dev with watcher + hang detection)
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
├── op_design.md                         # Operation design doc
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
scripts/tt-test.sh --dev tests/ttnn/unit_tests/operations/row_centralize/test_stage_data_pipeline.py

# Check TDD pipeline status
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py status --op-path ttnn/ttnn/operations/row_centralize
```

## Key Design Decisions

### Why does the architect run before the builder?

The architect registers TDD stages in `.tdd_state.json`. The generic-op-builder reads that file to discover stages and generate test files. The architect must complete first so the builder can see the registered stages.

### Why separate test directories?

Tests at `tests/ttnn/unit_tests/operations/{op_name}/` instead of colocated with the operation. This avoids pytest autodiscovery issues and follows the repo convention.

### Who determines TDD stages?

The **architect** owns stage determination using two heuristics:
- **H1 (Kernel Complexity Ordering)**: Which kernel to finalize first
- **H2 (Semantic Goal Progression)**: How to break a kernel into testable milestones

The kernel writer implements exactly one assigned stage at a time. It is forbidden from implementing future stages.

### What does the design document look like?

~400-550 lines, two parts:
- **Part 1 (Architecture)**: API, validation, CB layout, work distribution, data flow, test criteria
- **Part 2 (Kernel Implementation)**: TDD stages, helper mappings, per-phase details, critical notes

Part 2 validates Part 1 decisions against actual helper requirements, fixing conflicts immediately.

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
| `.tdd_state.json` missing | Architect didn't run | Re-run Phase 2 (architect) |
| No stage test files | Orchestrator registration failed | Check `tdd_orchestrator.py add-stage` output |
| Test hangs | CB sync mismatch in kernels | `pkill -9 -f pytest && tt-smi -r`, check watcher log |
| Builder can't find stages | Architect didn't complete | Ensure architect finishes before builder starts |
