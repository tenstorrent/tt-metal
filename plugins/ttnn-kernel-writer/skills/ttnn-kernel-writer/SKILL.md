---
name: ttnn-kernel-writer
description: Implement TTNN kernels from op_design.md for a specific stage or full op scope, then verify with tests.
argument-hint: "<op_design_path> [--stage <stage_name>]"
---

# TTNN Kernel Writer (Codex)

Use this role when design artifacts exist and kernel implementation is required.

## Inputs

- `op_design.md` (required)
- Optional stage scope and prior failure context

## Workflow

1. Read the design and isolate the requested stage scope.
2. Implement only required kernel sections (reader/compute/writer), using helpers whenever they satisfy stage requirements.
3. Keep CB synchronization consistent with helper contracts.
4. Run stage tests and inspect failures.
5. Apply minimal, scope-preserving fixes.

## Output Contract

- Updated kernel files in `{op_path}/kernels/`
- Any required upstream descriptor fixes (only if necessary)
- Test result summary for the target stage

## Guardrails

- Do not redesign the operation outside the stage scope.
- Use existing TTNN helpers by default; raw/manual logic is allowed only for explicit helper gaps documented in design context.
- When helpers own CB operations, do not add redundant wait/pop/push around them.
- Keep earlier passing stages behavior intact.

## Legacy Mapping

This skill is the Codex-native runtime replacement for:
- `tt_metal/third_party/tt_ops_code_gen/agents/ttnn-kernel-writer.md`


## User Controls

Before execution, read `references/user-config.md` and apply user preferences for scope, depth, and output style.
