Generic Reductions — Program Cache Review

Reviewed files:
- `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.hpp`
- `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp`
- `ttnn/cpp/ttnn/operations/reduction/generic/device/multi_core_h/reduce_op_multi_core_h.cpp`
- `ttnn/cpp/ttnn/operations/reduction/generic/device/multi_core_w/reduce_op_multi_core_w.cpp`
- `ttnn/cpp/ttnn/operations/reduction/generic/device/single_core_hw/reduce_op_single_core_hw.cpp`

Findings:
- Uses the old type-erased infra with override callbacks in at least the H-dimension multi-core path; width-sharded path updates dynamic CB addresses for source/output; interleaved path updates reader/writer kernel runtime args per-core.
- Hashing relies on default op hashing with attributes covering reduce op type, dimension, compute kernel config, memory configs, and sub-core grids; tensor args include shapes/dtypes/layouts.

No program-cache issues identified.

Suggested optional tests:
- Two-run cache tests for both interleaved and width-sharded cases that reallocate input/output while keeping attributes constant; assert single cache entry and correctness on cache hit.
