# Archived: `mcast_pipe` migration feedback

This is the review log for issues found in individual `mcast_pipe` migrations.
It is separate from `api_feedback.md`: API feedback concerns the helper contract,
while this file records places where a migration did not use that contract
robustly, left avoidable coupling to an argument layout, or still needs
migration-specific performance validation.

## Status values

- **Open** — feedback is recorded but the migration has not been revised.
- **Accepted** — the required migration change is agreed.
- **Rejected** — the current migration is retained; record the reason.
- **Implemented** — the migration and its relevant tests have been updated.

## MIG-001 — Chain trailing CT arguments from `McastArgs`

- **Date:** 2026-08-04
- **Status:** Implemented
- **Kernel:**
  `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp`
- **Related API feedback:** `api_feedback.md`, API-001

The original v9 migration constructed:

```cpp
constexpr dataflow_kernel_lib::McastArgs<12, 3, num_input_cores> act_mcast_args;
```

but later hard-codes compile-time argument positions belonging to the block
after it:

```cpp
load_config_tensor_if_in_dram<26, 27, 28, cb_reader_indices>(
    noc, reader_indices_dfb, 0);
```

That coupled the migration to the five-word v9 `McastArgs` CT encoding.
For example, API-001's proposed addition of a host-provided rotating span would
shift every following CT argument and silently invalidate these indices.

### Implemented direction

Use the decoder's existing chaining method as the source of truth:

```cpp
constexpr uint32_t post_mcast_ct_base =
    act_mcast_args.next_compile_time_args_offset();
```

The Conv-specific block currently contains nine words after the mcast block
and before the optional config tensor:

```cpp
constexpr uint32_t config_ct_base = post_mcast_ct_base + 9;

load_config_tensor_if_in_dram<
    config_ct_base + 0,
    config_ct_base + 1,
    config_ct_base + 2,
    cb_reader_indices>(noc, reader_indices_dfb, 0);
```

The preferred cleanup is to express all CT fields after `McastArgs` relative to
`post_mcast_ct_base`, with named offsets or a small decoder for the Conv-specific
block. That makes `config_ct_base` the next offset of the preceding block rather
than merely replacing absolute literals `26/27/28` with an unexplained `+9`.

The same principle applies to runtime arguments: a block following
`McastArgs` should begin at `act_mcast_args.next_runtime_args_offset()` rather
than restating the current mcast RT width.

### Resolution

Gate 2 rewrote every following CT field and the config-tensor tail relative to
`act_mcast_args.next_compile_time_args_offset()` and confirmed the factory's
contiguous block order. Gate 4 then changed the kernel to
`McastArgs<12, 3>` under API v10; the sixth `rotating_span` word moves the
following block without any caller-side index update. The exact fresh-JIT
width-sharded case passed at PCC `0.9999992597711427`, and the complete mapped
width inventory passed 48 runnable cases with 16 expected skips plus its
DRAM-config route and the shared 14-case DRAM inventory.

## MIG-002 — Sort row-start handshake remains outside the Pipe

- **Date:** 2026-08-04
- **Status:** Implemented
- **Kernels:** `coordinator_single_row_multi_core.cpp` and
  `reader_single_row_multi_core.cpp`
- **Related API feedback:** `api_feedback.md`, API-003

The sort migration replaced the coordinator-to-workers control multicast with
`send_signal()` / `receive_signal()`, but retained the immediately adjacent
row-start handshake:

```cpp
cores_to_coordinator_ready_sem.wait(number_of_dest);
cores_to_coordinator_ready_sem.set(0);
coordinator_pipe.send_signal();
```

Each reader performs the matching remote readiness increment before its
row-start `receive_signal()`. This is behaviorally the same receiver-ready gate
that a handshaked control Pipe should own. It remained explicit because the
current signal-only methods ignore the Pipe's handshake configuration.

Later sub-stage signals intentionally have no readiness handshake, so the
existing single Pipe cannot simply be changed to `handshake=true`. Revisit the
migration after API-003 is resolved; the likely formulation is separate
handshaked row-start and no-handshake sub-stage control channels. The distinct
writer-done counter should remain operation-owned.

### Resolution

Implemented on 2026-08-05. The host binding emits two contiguous, chained
`Mcast2D` blocks: a handshaked Counter channel for row start and a no-handshake
Counter channel for sub-stage release. The coordinator and reader construct
both Pipe faces once and use the appropriate channel at each protocol point.
The raw reader-ready semaphore and its explicit atomic increment/wait/reset
sequence are gone. The writer-done counter remains independent operation
protocol on semaphore 3.

The exact cold-JIT long case, both `Ht=2` deadlock regressions, and all seven
long cases passed. A durable source audit rejects reintroduction of the raw
row-start readiness name and requires both configured channels in the factory.

## MIG-003 — Let `Mcast2D` own matmul-1D semaphores and splice opaque arg blocks

- **Date:** 2026-08-04
- **Status:** Implemented
- **Host file:**
  `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`
- **Kernel files:** `reader_bmm_tile_layout_in1_sender_writer_padding.cpp`
  and `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp`
- **Related API feedback:** `api_feedback.md`, API-001

### Semaphore ownership

The migrated in1 path creates or declares the two multicast semaphores outside
`Mcast2D`, then passes their IDs back through `McastConfig::sem_ids`. In the
descriptor path it subsequently emits manual `SemaphoreDescriptor`s for those
same IDs.

This reverses the intended default ownership. The caller has no independent
semaphore protocol to preserve here; these are exactly the Pipe's data-ready and
consumer-ready cells. Construct `Mcast2D` without adopted `sem_ids`, let it
assign the IDs from `base_sem_id`, and add the complete result of
`owned_semaphores()` to the descriptor.

The legacy `Program` path currently uses `CreateSemaphore` rather than
`SemaphoreDescriptor`. It should use an equivalent helper-owned application
path instead of allocating IDs first and making `Mcast2D` adopt them. If no
bridge from helper-owned semaphore declarations to `Program` exists, add that
bridge or keep the legacy exception explicit until it can be removed; do not
use the legacy limitation to retain duplicate ownership in the descriptor path.

### CT/RT block insertion

The host currently treats helper output as fixed-size tuples:

```cpp
in1_mcast_compile_time_args[0]
// ...
in1_mcast_compile_time_args[4]

in1_mcast_runtime_args[0]
// ...
in1_mcast_runtime_args[3]
```

That assumes knowledge the caller should not have: five CT words and four RT
words. The assumption will break as soon as the self-describing wire changes,
including API-001's proposed CT-provided rotating span or a future role-specific
RT layout.

Reorder the host/device ABI as needed so each mcast block occupies one clean,
contiguous insertion point. Build the operation-specific prefix, insert the
entire output range from `compile_time_args()` or `runtime_args(core)`, then
append the operation-specific suffix. Do not preserve historical numeric
positions merely to avoid moving downstream fields.

Conceptually:

```cpp
append(prefix, args);
args.insert(args.end(), mcast_args.begin(), mcast_args.end());
append(suffix, args);
```

On the device, use `McastArgs::next_compile_time_args_offset()` and
`next_runtime_args_offset()` as the boundary for the following block. In
particular, replace the sender kernel's manual `rt_args_idx += 4`; the kernel
must not restate the current helper RT width.

### Resolution checklist

- Remove manual semaphore IDs/descriptors from the descriptor path and append
  all of `in1_mcast.owned_semaphores()`.
- Define how the legacy `Program` path consumes helper-owned semaphore
  declarations without allocating/adopting the IDs first.
- Replace every indexed CT/RT extraction from `in1_mcast` in both 1D host
  implementations with whole-range insertion.
- Move surrounding arguments where necessary and update both sender and
  receiver kernels to derive post-mcast offsets through `McastArgs`.
- Audit optional bias/output-sharded tails, TensorAccessorArgs bases, fused CCL
  fields, and program-cache override indices after the ABI reorder.
- Apply the same audit to the analogous matmul 2D bindings; they were migrated
  as the same atomic unit and may repeat the pattern.
- Rebuild host code, run one compile-focused 1D parametrization first, then run
  the mapped matmul inventory sequentially.

### Resolution

Implemented in Gates 2 and 3. Both matmul-1D/2D legacy and descriptor paths
insert the helper's complete CT/RT ranges at one opaque ABI boundary. Sender
and receiver kernels resume through `next_compile_time_args_offset()` and
`next_runtime_args_offset()`, while cached output and bias patch indices are
derived from the emitted range size.

The helper now owns the in1 semaphore pair. Descriptor factories append all
of `owned_semaphores()`, and the legacy `Program` bridge creates declarations
in ID order and rejects an allocated ID that differs from the declaration.
Matmul-2D also uses the single offset-aware `Mcast1D` described by API-004.
The host build, `McastHostFixture` 25/25, focused 1D and both 2D orientations,
and `MM-IN1-ALL` at 302 passed / 188 expected skips all passed.

## MIG-004 — Validate GroupNorm's fixed three-rectangle sender path

- **Date:** 2026-08-04
- **Status:** Implemented — production configurations are zero-edge; wrapped splitter paths have synthetic host coverage
- **Host file:**
  `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_sharded_program_factory.cpp`
- **Sender kernels:** `reader_mcast_sender_unary_sharded_gn_v2.cpp` and
  `welford_reader_mcast_sender_unary_sharded_gn_v2.cpp`

Before migration, the sender always constructed and sent the middle rectangle,
but sent first/last edge rectangles only when those groups existed. The host
helper now always emits three `Mcast2D` blocks. A missing edge becomes a
sender-only singleton, so its `send()` is behaviorally a no-op but still incurs
pipe construction and a degenerate call/branch path on every reduction value.

The common rectangular case therefore changed from one send call to three,
with two degenerate calls. A one-edge case changed from two to three. A case
with both edges still performs three real sends and may benefit from pipe
construction moving out of the hot loop.

### Existing evidence

On Blackhole p100a at AICLK 800, with three warmups and 20 real-time-profiler
records per case, baseline `4a1d6a97ca9` and migrated `28356d43846` measured:

| SDXL `(1, 1920, 32, 32)` case | Baseline median (ns) | Migrated median (ns) | Delta |
| --- | ---: | ---: | ---: |
| Legacy | 48,593.704 | 48,714.444 | +0.248% |
| Welford | 261,695.556 | 260,426.667 | -0.485% |

These 8x8 block-sharded cases use rectangular column groups and execute two
degenerate edge sends. No material regression was observed for that shape.

### Resolution

The supported production inventory generates only zero-edge rectangles. The
factory requires the shard grid to merge to one dense rectangle, and its
block-sharded sender boundaries align with the row/column traversal partitions
enforced by the batch/group divisibility checks. The mapped height-sharded
production case uses an 8x1 grid. Inspecting the host-generated groups for both
orientations and every mapped GroupNorm v2 configuration found no first or last
edge rectangle.

One- and two-edge partitions remain valid defensive behavior for a wrapped
coordinate sequence, but no mapped production configuration reaches them.
`GroupNormMcastGeometry` now exercises the same production splitter directly
with zero-edge, one-edge, and two-edge coordinate sequences; all three cases
pass. `McastHostFixture` remains green at 25/25, and `./build_metal.sh` passes.

The only supported production performance class is therefore the zero-edge
class already measured above. Legacy is +0.248% and Welford is -0.485% versus
the matched baseline, both within the 1.5% gate. No new wrapped baseline run or
hot-path change is warranted.
