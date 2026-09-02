# KDA eltwise-helper migration inventory

This is the working inventory for migrating the KDA compute kernels introduced
by the `mvasilijevicTT` numbered KDA split series.  The order below is the
migration order unless a dependency discovered during the work requires a
change.

| Order | PR | Compute kernel | Accuracy test(s) | Status |
| --- | --- | --- | --- | --- |
| 1 | #52783 | `ttnn/cpp/ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/device/kernels/compute/sigmoid_gated_rms_norm.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_sigmoid_gated_rms_norm.py` | Migrated; build unavailable (no Docker) and accuracy requires Blackhole (host has n300) |
| 2 | #52784 | `ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/compute/qkv_causal_conv1d_silu.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_qkv_causal_conv1d_silu.py` | Migrated; build unavailable (no Docker) and accuracy requires Blackhole (host has n300) |
| 3 | #52786 | `ttnn/cpp/ttnn/operations/experimental/kda/reduce_affine_transforms/device/kernels/compute/reduce_affine_transforms.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_reduce_affine_transforms.py` | Migrated; build unavailable (no Docker) and accuracy requires Blackhole (host has n300) |
| 4 | #52788 | `ttnn/cpp/ttnn/operations/experimental/kda/affine_exclusive_scan/device/kernels/compute/affine_exclusive_scan.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_affine_exclusive_scan.py` | Migrated; build unavailable (no Docker) and accuracy requires Blackhole (host has n300) |
| 5 | #52797 | `ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/kernels/compute/prepare_chunk_recurrence.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_prepare_chunk_recurrence.py` | Not started |
| 6 | #52798 | `ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/device/kernels/compute/recurrent_chunk_scan.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_recurrent_chunk_scan.py`; `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_summarize_chunk_recurrence.py` | Not started |

`ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp`
is a shared compute helper, not a standalone compute kernel.  It is covered
indirectly by the prepare-chunk-recurrence and recurrent/summarize tests.

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
