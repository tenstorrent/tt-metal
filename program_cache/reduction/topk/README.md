TopK — Program Cache Review

Reviewed files:
- `ttnn/cpp/ttnn/operations/reduction/topk/device/topk_op.hpp`
- `ttnn/cpp/ttnn/operations/reduction/topk/device/topk_op.cpp`
- `ttnn/cpp/ttnn/operations/reduction/topk/device/topk_program_factory.cpp`

Findings:
- Uses the old type-erased infra with an override callback in both single-core and multi-core paths.
- Override updates runtime-only values on cache hits:
  - Reader: input buffer base address; optional input-indices buffer base when present.
  - Writer: output values and indices buffer base addresses.
- Other parameters that affect codegen (k, dim, largest/sorted, tile counts, grid splits, dtype/layout, memory configs) are compile-time args and part of default hashing via op attributes, so they remain constant across cache hits.
- No custom program hash; default hashing should be sufficient given attributes and tensor args included.

No program-cache issues identified.

Suggested optional tests:
- Two-run cache test (single-core config) that reallocates input and output tensors between runs; expect identical results and a single program cache entry.
- Two-run cache test (multi-core config with provided input indices) that varies only buffer addresses to exercise override path for optional tensor.
