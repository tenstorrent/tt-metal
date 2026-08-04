# `mcast_pipe` migration feedback

This is the review log for issues found in individual `mcast_pipe` migrations.
It is separate from `api_feedback.md`: API feedback concerns the helper contract,
while this file records places where a migration did not use that contract
robustly or left avoidable coupling to an argument layout.

## Status values

- **Open** — feedback is recorded but the migration has not been revised.
- **Accepted** — the required migration change is agreed.
- **Rejected** — the current migration is retained; record the reason.
- **Implemented** — the migration and its relevant tests have been updated.

## MIG-001 — Chain trailing CT arguments from `McastArgs`

- **Date:** 2026-08-04
- **Status:** Open
- **Kernel:**
  `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp`
- **Related API feedback:** `api_feedback.md`, API-001

The migrated kernel constructs:

```cpp
constexpr dataflow_kernel_lib::McastArgs<12, 3, num_input_cores> act_mcast_args;
```

but later hard-codes compile-time argument positions belonging to the block
after it:

```cpp
load_config_tensor_if_in_dram<26, 27, 28, cb_reader_indices>(
    noc, reader_indices_dfb, 0);
```

This couples the migration to the current five-word `McastArgs` CT encoding.
For example, API-001's proposed addition of a host-provided rotating span would
shift every following CT argument and silently invalidate these indices.

### Expected direction

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

### Resolution checklist

- Rewrite CT reads at indices 17 through 28 as a chained, named layout rooted at
  `act_mcast_args.next_compile_time_args_offset()`.
- Confirm the host factory appends fields in exactly the same block order.
- Check the rest of the migration for absolute offsets that cross an
  `McastArgs` boundary.
- Compile the kernel through the mapped width-sharded Conv test after the
  migration is updated.

## MIG-002 — Sort row-start handshake remains outside the Pipe

- **Date:** 2026-08-04
- **Status:** Open
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

Until then, the migration is behaviorally validated but should not be described
as having absorbed the complete row-start multicast-handshake block.

## MIG-003 — Let `Mcast2D` own matmul-1D semaphores and splice opaque arg blocks

- **Date:** 2026-08-04
- **Status:** Open
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
