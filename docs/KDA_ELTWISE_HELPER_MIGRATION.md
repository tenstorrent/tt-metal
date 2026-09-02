# KDA eltwise-helper migration inventory

This is the working inventory for migrating the KDA compute kernels introduced
by the `mvasilijevicTT` numbered KDA split series.  The order below is the
migration order unless a dependency discovered during the work requires a
change.

| Order | PR | Compute kernel | Accuracy test(s) | Status |
| --- | --- | --- | --- | --- |
| 1 | #52783 | `ttnn/cpp/ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/device/kernels/compute/sigmoid_gated_rms_norm.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_sigmoid_gated_rms_norm.py` | Accuracy-baselined (31 Blackhole tests); helper boundary is sound. |
| 2 | #52784 | `ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/compute/qkv_causal_conv1d_silu.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_qkv_causal_conv1d_silu.py` | Accuracy-baselined (36 Blackhole tests); helper boundary is sound. |
| 3 | #52786 | `ttnn/cpp/ttnn/operations/experimental/kda/reduce_affine_transforms/device/kernels/compute/reduce_affine_transforms.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_reduce_affine_transforms.py` | Accuracy-baselined (33 Blackhole tests); migrate only local copy/send windows. |
| 4 | #52788 | `ttnn/cpp/ttnn/operations/experimental/kda/affine_exclusive_scan/device/kernels/compute/affine_exclusive_scan.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_affine_exclusive_scan.py` | Accuracy-baselined (36 Blackhole tests); helper copy stages preserve destination blocking. Performance sign-off remains pending. |
| 5 | #52797 | `ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/kernels/compute/prepare_chunk_recurrence.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_prepare_chunk_recurrence.py` | Accuracy-baselined (36 Blackhole tests); SFPU `Square` restored and the generic binary facade/dead runtime bindings removed. |
| 6 | #52798 | `ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/device/kernels/compute/recurrent_chunk_scan.cpp` | `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_recurrent_chunk_scan.py`; `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_summarize_chunk_recurrence.py` | Accuracy-baselined (40 recurrent-scan; 30 summarize-recurrence Blackhole tests); raw state-update matmul is hoisted and owns its pack format. |

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

## Architecture map and helper boundary

This is the planning contract for the next migration pass.  Passing accuracy
means that the substitutions execute correctly for the covered shapes; it does
not by itself establish equivalent LLK semantics, code size, or throughput.

| Kernel family | Kernels | Dataflow shape | Correct helper boundary |
| --- | --- | --- | --- |
| Local normalization | sigmoid-gated-RMS-norm | Local pointwise chain interrupted by a reduction | The pointwise links which pack an L1 result belong to a helper chain. Startup, long-lived weight/epsilon constants, and the reduce callback stay raw/caller-owned. |
| Streaming convolution | QKV causal-conv1d-SiLU | Four-tap, offset-addressed weight window and a local partial accumulator | A helper owns one tap's local arithmetic. The caller owns tilize, the four-tap weights window, startup, and its conditional lifetime. |
| Distributed affine scan | reduce-affine-transforms; affine-exclusive-scan | NoC mailbox exchanges interleaved with matmuls and local copies | Only complete local copy/send windows which pack their own L1 output are helper work. NoC protocol, DST preloads, raw matmul, and multi-output lifetimes remain explicit. |
| Chunk-recurrence preparation | prepare-chunk-recurrence | A staged DAG of normalization, gates, scan inputs, pairwise matrices, inverses, and decay outputs | Helper chains cover complete pointwise stages. Raw reduce callbacks, matmuls, transpose, and pointer-selected scratch rotation delimit stages. |
| Runtime state recurrence | recurrent-chunk-scan | Matmul/pointwise recurrence through a double-buffered state ring, plus a summary epilogue | Keep each raw matmul hoisted once; dispatch only the pointwise finishing step to static helper specializations. State-ring and summary-ring lifetimes remain caller-owned. |

A helper is appropriate only when it owns a complete semantic window: input
wait through output push; compile-time DFB identities; no later use of an input
after its helper pop; no required raw DST operation next; and an L1-packed
output.  The caller retains ownership for engine startup, raw
matmul/reduce/transpose and their callbacks, NoC/mailbox protocols, long-lived
or offset-addressed windows, fan-out outputs, cross-stage workspaces, and
runtime-selected buffers.  Raw LLKs in those cases are an intentional boundary,
not unfinished migration.

Ordinary helper inputs and outputs own their unpack and pack reconfiguration.
Every raw operation must instead document or establish its own format state;
in particular, raw matmul pack configuration cannot be silently inherited from
an unrelated helper.  When a helper replaces a batched raw loop, its iteration
shape must explicitly preserve the original destination block size--a bare
`tiles(n)` defaults to one tile and can regress throughput.

Do not introduce a generic `elementwise_binary` facade merely to shorten call
sites.  It is useful only if it removes, rather than duplicates, the binding
between a template DFB id and a runtime handle, has one clear lifecycle policy,
and retains meaningful stage names.  Otherwise direct helper chains express
the graph and its buffer ownership more accurately.

### Follow-up order

1. Measure any proposed static unrolling of prepare-chunk-recurrence's inverse
   rotation before retaining a configuration-budget claim.
2. Validate affine-exclusive-scan's restored copy blocking with multi-tile
   accuracy shapes and a performance measurement.
3. Investigate recurrent-chunk-scan's `final_state` producer/consumer binding
   independently; retain the hoisted raw-matmul and its local pack-format
   ownership.
4. Add only focused safety assertions or tests for discovered invariants (for
   example affine scan double buffering); do not broaden helper ownership
   across the raw boundaries above.

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
