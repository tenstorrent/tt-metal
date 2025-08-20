Moreh Dot Backward – Program Cache Review

Findings: Reviewed `create(...)`, `override_runtime_arguments(...)`. No program-cache issues found.

- Override updates:
  - Reader: updates output_grad, input, and other buffer base addresses.
  - Writer: updates optional output buffer base addresses if present (`input_grad`, `other_grad`).
- Constants across cache hits: flags for optional outputs, `num_tiles`, and kernel layout are determined at creation and captured in default hash via operation attributes/tensor args.
- Per-core: Single-core program; overrides update core {0,0} only.

References:
- `ttnn/cpp/ttnn/operations/moreh/moreh_dot_backward/device/moreh_dot_backward_program_factory.cpp`.

Status: Reviewed – no issues identified.
