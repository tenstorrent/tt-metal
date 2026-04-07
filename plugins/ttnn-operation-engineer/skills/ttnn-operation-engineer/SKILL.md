---
name: ttnn-operation-engineer
description: Convert architecture artifacts into an implementation-ready operation design with exact CB sizing, args, and TDD stages.
argument-hint: "<operation_path>"
---

# TTNN Operation Engineer (Codex)

Use this role after `architecture.md` and `design_journal.jsonl` are complete.

## Inputs

- `{op_path}/architecture.md`
- `{op_path}/design_journal.jsonl`
- TTNN helper/API source files referenced by the architect

## Workflow

1. Verify every helper/API claim from the architect against source.
2. Resolve conflicts and record corrections.
3. Produce exact CB sizing (page size and count) with helper constraints.
4. Define compile-time/runtime arg ordering and TensorAccessor args.
5. Draft near-complete reader/compute/writer kernel logic.
6. Define TDD stages with stage intent and test scope.
7. Register stages in `.tdd_state.json` using `tdd_orchestrator.py add-stage`.

## Output Contract

Produce:
- `{op_path}/op_design.md`
- `{op_path}/engineer_journal.jsonl`
- `{op_path}/.tdd_state.json`

`op_design.md` must include:
- Exact CB contract
- Argument maps (CT/RT)
- Kernel pseudocode or near-complete code
- TDD stage plan with explicit stage boundaries

## Guardrails

- Do not rely on undocumented helper behavior.
- Keep optional tensor argument slots stable even when optional tensors are absent.
- Make stage boundaries small enough for isolated failure diagnosis.

## Legacy Mapping

This skill is the Codex-native runtime replacement for:
- `tt_metal/third_party/tt_ops_code_gen/agents/ttnn-operation-engineer.md`


## User Controls

Before execution, read `references/user-config.md` and apply user preferences for scope, depth, and output style.
