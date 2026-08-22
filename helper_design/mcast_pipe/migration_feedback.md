# `mcast_pipe` migration review feedback

This is the active review log for migrated operations. It records requested
follow-up work; an item being listed here does not mean that the production code
has been changed.

## Cross-cutting feedback

### MCAST-001 — Put multicast arguments at the end

- Date: 2026-08-22
- Status: Resolved (2026-08-22)
- Scope: Every kernel migrated to `mcast_pipe`, including every host-side
  producer of its compile-time and runtime arguments

Arrange each migrated kernel's argument ABI in this order:

1. All operation-owned compile-time arguments, followed by all multicast helper
   compile-time argument blocks.
2. All operation-owned runtime arguments, followed by all multicast helper
   runtime argument blocks.

The kernel source should read the arguments in that same natural order: read and
name the operation arguments first, then construct/read the `McastArgs` blocks at
the end. The ordering must be updated in lockstep in the kernel and in every
factory, descriptor path, legacy path, runtime override, and other host-side
argument producer.

If a kernel has multiple multicast blocks, keep their internal order stable, but
place the complete group after all operation-owned arguments.

Implementation suggestion: define named boundaries for the operation argument
blocks and use those boundaries as the `McastArgs<CT_BASE, RT_BASE>` bases. This
makes the tail layout explicit and avoids unexplained numeric offsets.

Resolution: normalized every migrated/source-integrated kernel and host
producer to operation-first CT/RT prefixes followed by opaque helper tails.
This included TensorAccessor descriptors, optional branches, cache rebinding,
multiple helper families, sparse and descriptor Matmul paths, Argmax, Conv2D
weight sharing, and Conv3D's variable worker-coordinate prefix. A dynamic
runtime-base constructor lets `McastArgs` follow genuinely variable operation
prefixes without splitting the ABI. The ledger-wide source audit now enforces
natural operation-before-helper declaration order.

### MCAST-002 — Make sender and receiver pipe types directly nameable

- Date: 2026-08-22
- Status: Resolved (2026-08-22)
- Scope: Migrated mixed-role kernels that conditionally store sender and
  receiver pipes

Avoid call-site type recovery like:

```cpp
using In0SenderPipe = decltype(in0_mcast_args.sender(noc));
using In0ReceiverPipe = decltype(in0_mcast_args.receiver(noc));
```

The natural kernel vocabulary should be `SenderPipe` and `ReceiverPipe`. The
helper already derives their concrete template configuration from `McastArgs`,
so callers should not have to invoke `sender(noc)` or `receiver(noc)` inside a
`decltype` merely to name the corresponding storage type.

Investigate an API that lets mixed-role kernels declare their optional storage
using directly exposed sender/receiver pipe type names. Ideally the call site
can simply say `SenderPipe` and `ReceiverPipe`; if retaining compile-time
specialization prevents that exact spelling, expose concrete aliases from
`McastArgs` that remain explicit and do not require expression-based type
deduction.

The first reviewed occurrence is
`ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp`.
The same pattern also exists in the convolution width-sharded activation reader,
so treat this as helper/API feedback rather than a Matmul-only cleanup.

Resolution: `McastArgs` now exposes concrete `SenderPipe` and `ReceiverPipe`
aliases, plus `SenderPipeFor<NOC_ID>` for the uncommon explicit-NoC type. Every
migrated mixed-role kernel now declares optional storage from those aliases;
none recovers a type by invoking `sender(noc)` or `receiver(noc)` in `decltype`.
The cross-operation audit also found Group Attention Matmul eagerly constructing
a receiver on sender-only participants. It now constructs each face only when
the corresponding runtime role permits it, retaining the helper's role asserts
and fixing the reproduced device hang. A source audit enforces the aliases and
role-conditional storage.

### MCAST-003 — Append host argument blocks like `TensorAccessorArgs`

- Date: 2026-08-22
- Status: Resolved (2026-08-22)
- Scope: Every host-side producer of multicast compile-time and runtime
  arguments

Follow the `TensorAccessorArgs` construction pattern for multicast arguments.
The host should first build all operation-owned compile-time or per-core runtime
arguments, including any branch-specific fields, and then append the multicast
helper block in one obvious operation at the end of the completed vector.

Ideally the helper exposes append-style APIs analogous to
`TensorAccessorArgs::append_to(...)` for both compile-time and runtime wires,
rather than requiring callers to obtain a temporary vector and manually
`insert` it. Place those append calls after the relevant `if`/`else` structure
whenever the argument block is common to all branches. Do not duplicate or
interleave multicast insertion inside individual branches merely because their
operation-owned prefixes differ.

The resulting host source should visually match the ABI required by MCAST-001:
operation arguments are constructed first, and the opaque multicast block is a
single tail append. Where genuinely different kernels or multicast families
require different wires, keep separate completed vectors and apply the same
tail-append rule to each one.

Resolution: added `append_compile_time_args_to(...)` and
`append_runtime_args_to(...)` to both host multicast helpers, with support for
ordinary integer vectors and descriptor runtime vectors containing buffer
bindings. Converted every migrated host producer to build its complete
operation-owned prefix first and append each opaque multicast block once at the
tail. Existing getters remain only for legitimate wire queries and compatibility;
a ledger-wide source audit rejects non-append production bindings. Focused host,
source, compile, cache-reuse, and device tests passed across Matmul, Sparse
Matmul, Group Attention Matmul, Conv2D/Conv3D, normalization, reduction, sorting,
movement, and SDPA families.

### MCAST-004 — Make presence part of the `McastArgs` wire

- Date: 2026-08-22
- Status: Resolved (2026-08-22)
- Scope: The `mcast_pipe` host and kernel APIs and every migrated operation

Make optionality an intrinsic property of every `McastArgs` block instead of
using the separate `OptionalMcastArgs` type or operation-owned named flags such
as `in0_mcast_args_present` and `in1_mcast_args_present`.

The first positional compile-time word of every multicast block should be a
compile-time presence tag owned and decoded by `McastArgs` itself. A present
block contains that tag followed by the existing multicast compile-time payload
and its runtime payload. An absent block contains only a false presence tag and
no remaining multicast compile-time or runtime payload. This deliberately makes
an absent helper a one-word tagged block rather than a zero-width block; treat
that as the library-wide optional-helper ABI, not as a Matmul-only exception.

Presence is known at kernel compile time. `McastArgs` must therefore avoid
decoding payload fields when its tag is false, and attempting to instantiate or
obtain a sender or receiver from an absent block must fail with a clear
compile-time assertion rather than a runtime assertion. Its
`next_compile_time_args_offset()` and `next_runtime_args_offset()` results must
continue to describe the actual encoded width for both present and absent
blocks so chained helpers remain correct.

Update `Mcast1D` and `Mcast2D` host serialization to emit the true tag for
ordinary multicast blocks and provide the corresponding opaque way to append an
absent tagged block. Remove `OptionalMcastArgs` and migrate all of its consumers
to ordinary `McastArgs`; remove the operation-specific named presence flags and
make the helper wire the single source of truth. Apply the ABI change to every
migrated operation, including mandatory multicast blocks, and audit all chained
compile-time offsets, runtime bases, descriptor and legacy producers, cache
overrides, and source tests.

Resolution: helper API v13 makes the presence tag the first compile-time word
of every `McastArgs` block. Present `Mcast1D`/`Mcast2D` blocks serialize the tag
plus their six-word payload; the host absent-block append API serializes only a
false tag and no runtime words. `McastArgs` selects a compile-time
specialization from that tag: the absent specialization advances compile-time
offsets by one, leaves runtime offsets unchanged, reads no payload, and rejects
sender/receiver construction with a dependent compile-time assertion.
`OptionalMcastArgs` and the Matmul-owned presence flags were removed. All
migrated producers inherit the tagged present serialization, while the five
inactive Matmul/Sparse shared-kernel bindings emit the opaque absent block.
Host build, 25 source audits, 80 helper device tests, 36 host gtests, both 1D
Matmul directions, Sparse Matmul, and chained 2D Matmul passed; the device gates
ran sequentially under the safe wrapper with Watcher.

## Matmul

### MATMUL-001 — Apply MCAST-001 to the in0 tile-layout receiver

- Date: 2026-08-22
- Status: Resolved (2026-08-22)
- Kernel:
  `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp`
- Host producers:
  - `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`
  - `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp`

The receiver currently introduces `McastArgs<4, 0>` at the beginning of
`kernel_main`, then reads operation compile-time arguments from indices 0–3. Its
compile-time wire layout also splits the operation arguments around the multicast
block: the first four operation arguments precede it, while `batch` and
`get_batch_from_reader` follow it.

Move all six operation compile-time arguments into one contiguous prefix and put
the multicast compile-time block after them. Apply the runtime ordering rule as
well, and move the multicast decoding in the kernel source below the
operation-argument reads so the source order mirrors the ABI. Update both the 1D
and 2D host producers, including their descriptor and legacy construction paths.

### MATMUL-002 — Do not model an inactive 1D operand as a multicast family

- Date: 2026-08-22
- Status: Resolved (2026-08-22)
- Factory:
  `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`
- Affected paths: Both the legacy/program and descriptor builders

In the 1D `MCAST_IN0` path, in1 is not multicast: every work core reads its own
in1 slice, and the in1 sender/writer kernel is compiled with `SKIP_MCAST`.
Nevertheless, the factory constructs an `in1_mcast` as a one-core,
no-handshake `Mcast2D` and emits its compile-time and runtime wire blocks. This
object is only an ABI filler for the kernel's unconditional `McastArgs` decoder;
it does not represent operation behavior.

Remove this synthetic in1 multicast model. The inactive operand should not have
an `in1_mcast` object or multicast argument block merely to satisfy a shared
kernel layout. Adjust the kernel/factory interface so `SKIP_MCAST` paths do not
need to construct or decode multicast arguments. Apply the symmetric cleanup to
the 1D `MCAST_IN1` path, where the current `in0_mcast` self-rectangle is likewise
only a disabled placeholder. Preserve the real `in1_mcast` in the actual
`MCAST_IN1` variant.

### MATMUL-003 — Preserve the divergent fixed-sender ACK-count override

- Date: 2026-08-22
- Status: Resolved (2026-08-22)
- Factory:
  `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`
- Affected paths: Both the legacy/program and descriptor builders for 1D
  `MCAST_IN0`

In the non-sharded in0 path, the factory passes `num_cores - 1` as the final
argument to the fixed-sender `Mcast2D` constructor. That argument is an explicit
handshake ACK-count override, but for this dense multicast it is the same
`area - 1` fan-out that `Mcast2D` already derives from the receiver rectangle
and the sender's placement inside it.

Remove the explicit `num_cores - 1` argument and use the constructor's default
dense ACK-count derivation. The factory should provide the multicast geometry
and sender; it should not restate a count already owned and derived by the
helper.

Resolution correction: the stated dense-geometry premise does not hold for
every supported 1D shape. With `uneven_width=2`, the multicast rectangle is a
bounding box containing inactive landing cores, while only `num_cores - 1`
active receivers run the receiver kernel and acknowledge. Removing the override
deterministically hung with the sender waiting for excess consumer ACKs; the
same exact node passed after restoring it. The explicit override is therefore
required by the divergent-count guardrail in both legacy and descriptor
builders. The symmetric MCAST_IN1 builders retain it for the same potentially
partial bounding-box geometry.

### MATMUL-004 — Append the common in0 multicast compile-time tail once

- Date: 2026-08-22
- Status: Resolved (2026-08-22)
- Factory:
  `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`
- Affected paths: Both the legacy/program and descriptor 1D `MCAST_IN0`
  builders

The sharded and interleaved branches construct different operation-owned
prefixes for `in0_sender_compile_time_args`, but both branches currently end by
calling
`in0_mcast.append_compile_time_args_to(in0_sender_compile_time_args)`. The
multicast block is identical and is a common tail, so appending it separately in
both branches obscures that invariant and needlessly duplicates the binding.

Keep only the branch-specific construction of the operation arguments inside
the `if (in0_is_sharded)` / `else` statement. Move the multicast append after
the complete conditional and call it exactly once. Apply this cleanup to both
copies of the builder so the source directly expresses the MCAST-003 rule that
a common helper block is appended once after all operation-owned branches are
complete. Audit other migrated factories for the same duplicated common-tail
pattern and apply the same cleanup where the helper wire is genuinely identical
across branches.

Resolution: both 1D builders now finish their sharded/interleaved operation
prefix before appending the identical in0 helper tail once. The cross-operation
audit found the same pattern in both 2D Matmul builders and applied the same
cleanup there; no other migrated producer had a genuinely identical helper tail
duplicated across branches. A source regression test requires exactly one append
after each of the four conditionals. The release host build, all 26 source
audits, and focused 1D and 2D Matmul gates under Watcher passed.
