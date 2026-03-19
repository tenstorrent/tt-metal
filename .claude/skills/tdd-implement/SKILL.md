---
name: tdd-implement
description: Implement TTNN kernels through all TDD stages using a single persistent agent. Replaces per-stage kernel-writer invocations. Launches ttnn-kernel-writer-tdd which owns the full loop. Args = operation path.
argument-hint: "<operation_path>"
---

# TDD Kernel Implementation

Launch a single `ttnn-kernel-writer-tdd` agent that implements all TDD stages in one session.

## Prerequisites

The operation directory MUST contain:
1. `op_design.md` — from the architect
2. Stub kernel files — from the generic-op-builder
3. `.tdd_state.json` — with pre-registered stages
4. Program descriptor and entry point — from the builder

Verify with:
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py status --op-path <path>
```

## Usage

Extract the operation path from the user's arguments. Then launch ONE agent:

```
Agent: ttnn-kernel-writer-tdd
Prompt: |
  Implement all TDD stages for {op_name}.
  Operation path: {op_path}

  Follow the TDD loop exactly: implement stage → test → fix → advance → commit → next stage.
  Do NOT skip stages. Do NOT implement ahead.
```

That's it. The agent owns the full TDD loop internally. Do NOT:
- Loop over stages yourself
- Spawn multiple kernel-writer agents
- Parse failure output yourself
- Call tdd_orchestrator.py yourself

The agent handles all of this.

## When to Add Logging

If the user requests logging/breadcrumbs, append to the prompt:
```
Enable detailed logging.
```

## Result

The agent returns a structured report with:
- Per-stage pass/fail results
- Upstream fixes it made (program descriptor, entry point, CB config)
- Design deviations with justifications
- Files modified

### If a stage fails permanently

The agent will report `HUMAN REVIEW REQUIRED` with the failing stage name and a pointer to `{op_path}/tdd_failure_report.md`. Relay this to the user.

### If all stages pass

The operation is fully implemented and tested. All stages committed. Ready for integration testing or PR.
