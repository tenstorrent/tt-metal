Argmax — Program Cache Review

Reviewed files:
- `ttnn/cpp/ttnn/operations/reduction/argmax/device/argmax_op.hpp`
- `ttnn/cpp/ttnn/operations/reduction/argmax/device/argmax_program_factory.cpp`

Findings:
- Uses the old type-erased infra with override callbacks for both single-core and multi-core paths.
- Override updates runtime-only buffer base addresses for input and output on all active cores (and both core groups for multicore).
- Program parameters (reduce-all vs. along a dim, keepdim, core distribution, page sizes) are derived solely from hashed attributes and shapes; default hashing is sufficient.

No program-cache issues identified.

Suggested optional tests:
- Two-run cache tests for single-core and multi-core variants, reallocating buffers to exercise override path while keeping shapes/attrs constant; expect correctness and a single cache entry per variant.
