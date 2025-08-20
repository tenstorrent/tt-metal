Moreh Dot – Program Cache Review

Findings: Reviewed `create(...)`, `override_runtime_arguments(...)`. No program-cache issues found.

- Override updates:
  - Reader: updates input A and B buffer base addresses.
  - Writer: updates output buffer base address.
- Constants across cache hits: `num_tiles`, mask dims, and kernel config are hashed via default hash (dtype, shapes, memory/compute configs), so they remain constant for cache-hit runs.
- Per-core: Single-core program; overrides update core {0,0} only.

References:
- `ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/moreh_dot_program_factory.cpp`.

Status: Reviewed – no issues identified.
