# KDA eltwise-helper migration inventory

This is the working inventory for migrating the KDA compute kernels introduced
by the `mvasilijevicTT` numbered KDA split series.  The order below is the
migration order unless a dependency discovered during the work requires a
change.

| Order | PR | Compute kernel | Accuracy test(s) | Status |
| --- | --- | --- | --- | --- |
| 1 | #52783 | `ttnn/cpp/ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/device/kernels/compute/sigmoid_gated_rms_norm.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_sigmoid_gated_rms_norm.py` | Migrated; Blackhole accuracy passed (31 tests) |
| 2 | #52784 | `ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/compute/qkv_causal_conv1d_silu.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_qkv_causal_conv1d_silu.py` | Migrated; Blackhole accuracy passed (36 tests) |
| 3 | #52786 | `ttnn/cpp/ttnn/operations/experimental/kda/reduce_affine_transforms/device/kernels/compute/reduce_affine_transforms.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_reduce_affine_transforms.py` | Migrated; Blackhole accuracy passed (33 tests) |
| 4 | #52788 | `ttnn/cpp/ttnn/operations/experimental/kda/affine_exclusive_scan/device/kernels/compute/affine_exclusive_scan.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_affine_exclusive_scan.py` | Migrated; Blackhole accuracy passed (36 tests). The remaining raw copies preload affine offsets into DST for matmul accumulation and are not standalone eltwise operations. |
| 5 | #52797 | `ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/kernels/compute/prepare_chunk_recurrence.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_prepare_chunk_recurrence.py` | Migrated; Blackhole accuracy passed (36 tests). The reduction callback and pointer-selected inverse scratch rotation retain raw operations: the former owns DST inside `reduce`, and helper unrolling of the latter exceeds the Tensix kernel configuration budget. |
| 6 | #52798 | `ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/device/kernels/compute/recurrent_chunk_scan.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_recurrent_chunk_scan.py`; `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_summarize_chunk_recurrence.py` | Migrated; Blackhole accuracy passed (40 recurrent-scan tests; 30 summarize-recurrence tests). Runtime state-ring selection dispatches to static helper specializations, so all its eltwise work now uses helpers. |

`ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp`
is a shared compute helper, not a standalone compute kernel.  It is covered
indirectly by the prepare-chunk-recurrence and recurrent/summarize tests.

## DFB lifecycle ownership

Each helper owns the complete DFB window when it is that buffer's sole
consumer. It also owns a final `PopPolicy::AtEnd` when its input was staged by
an upstream operation but it is the final consumer; this removes split
lifecycle bookkeeping without changing the upstream wait. Caller-owned
windows remain where a buffer is reused by a later stage, held across a
multi-output operation, or consumed by a matmul/reduce callback. In
particular, the prepare-chunk-recurrence scratch buffers change semantic roles
between stages, so their ownership remains explicit at those transitions.

Helper input and output specifications leave data-format reconfiguration
enabled, so the helper owns ordinary unpack and pack transitions. Explicit
format reconfiguration remains only for raw matrix, transpose, and the
runtime-selected inverse scratch primitives. The redundant output pack
reconfiguration immediately before recurrent-chunk-scan's helper `add` was
removed.

## Per-kernel migration protocol

For each listed kernel, complete these gates in order:

1. Inline logical wrapper functions so the operation sequence and CB/DFB
   lifecycles are visible at the call site.
2. Replace the relevant raw eltwise LLKs with the eltwise helper(s).
3. Do a second source pass for missed data-format reconfiguration and CB/DFB
   lifecycle ownership, including waits, pops, reserves, pushes, and DST
   synchronization.
4. Compare the resulting helper configuration with the original raw LLKs,
   checking every init, execution, pack, broadcast, and reconfiguration.
5. Run the operation's accuracy tests. Defer performance runs until every
   listed kernel has completed the migration and accuracy gates.

Do not treat the model-only KDA integration PR (#52799) or the reference/test
utility PR (#52781) as standalone device-compute migrations.
