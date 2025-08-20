Moreh GetItem – Program Cache Review

Findings: Reviewed `create(...)`, `override_runtime_arguments(...)`. No program-cache issues found.

- Override updates:
  - Reader: updates input base address and each defined index tensor base address across 5D-expanded dims.
  - Writer: updates output buffer base address.
- Constants across cache hits: shape-derived strides, unit sizes, index presence flags, and core work-split are determined at creation and covered by default hash via operation attributes (`index_dims`, memory config) and tensor arg shapes; they remain constant for cache-hit runs.
- Per-core coverage: Iterates the same `num_cores` and `core_h` as in creation.

References:
- `ttnn/cpp/ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_rm_factory.cpp`.

Status: Reviewed – no issues identified.
