Moreh Fold – Program Cache Review

Findings: Reviewed `create(...)`, `override_runtime_arguments(...)`. No program-cache issues found.

- Override updates:
  - Reader: updates input buffer base address.
  - Writer: updates output buffer base address.
- Constants across cache hits: page sizes, stride/padding/dilation params, and per-core unit counts are shape/attr-derived and hashed; no override needed.
- Per-core coverage: Iteration over `cores` in override mirrors creation order.

References:
- `ttnn/cpp/ttnn/operations/moreh/moreh_fold/device/fold_program_factory_rm.cpp`.

Status: Reviewed – no issues identified.
