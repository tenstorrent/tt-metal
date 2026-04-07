---
name: ttnn-generic-op-builder
description: Build TTNN Python operation scaffolding, program descriptor wiring, and kernel stubs from op_design.md and .tdd_state.json.
argument-hint: "<operation_path>"
---

# TTNN Generic Op Builder (Codex)

Use this role to materialize operation scaffolding from engineered design artifacts.

## Inputs

- `{op_path}/op_design.md`
- `{op_path}/.tdd_state.json`

## Workflow

1. Create operation package under `ttnn/ttnn/operations/{op_name}/`.
2. Implement entry point wrapper and validation logic.
3. Implement program descriptor with CBs, kernels, and runtime args.
4. Create stub kernels in `{op_path}/kernels/`.
5. Add integration test skeleton under `tests/ttnn/unit_tests/operations/{op_name}/`.
6. Verify stage tests referenced in `.tdd_state.json` exist.

## Output Contract

Required outputs:
- `__init__.py`
- `{op_name}.py`
- `{op_name}_program_descriptor.py`
- Kernel stubs (`reader/compute/writer`)
- Integration test scaffold

## Guardrails

- Stub kernels should compile but contain no algorithm logic.
- `ttnn.allocate_tensor_on_device` must use positional args.
- `ttnn.generic_op` invocation must pass output tensor last.

## Legacy Mapping

This skill is the Codex-native runtime replacement for:
- `tt_metal/third_party/tt_ops_code_gen/agents/ttnn-generic-op-builder.md`


## User Controls

Before execution, read `references/user-config.md` and apply user preferences for scope, depth, and output style.
