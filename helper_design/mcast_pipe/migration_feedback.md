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

Resolution correction (2026-08-23): MCAST-005 refines the runtime ordering for
operation tails whose length is known only at runtime. Compile-time arguments
remain operation-first/helper-last. Runtime arguments use a fixed operation
prefix, then the helper block, then the variable operation tail; this makes the
helper base compile-time-constant without encoding its wire width in the
operation. Fixed-width ABIs continue to put the helper at the end.

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

Resolution correction (2026-08-23): the append remains one opaque operation at
one contiguous boundary, but MCAST-005 permits a variable-length operation tail
to be appended after it. The helper is still never indexed, split, or
interleaved with another block.

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

### MCAST-005 — Make the runtime base template-owned

- Date: 2026-08-23
- Status: Resolved (2026-08-23)
- Scope: The `McastArgs` kernel API and every migrated operation with a
  runtime-sized operation argument region

`McastArgs<CT_BASE, RT_BASE>` must have one source of truth for its runtime
base. Remove the constructor that accepts a second runtime base and the mutable
stored base/end accessors. Every helper read and every chained boundary must use
the `RT_BASE` template argument.

For an ABI containing operation data whose length is known only at runtime,
order its runtime arguments as:

1. the fixed-size operation prefix;
2. the complete opaque multicast helper block;
3. the variable-size operation tail.

The kernel derives the tail start with
`McastArgs::next_runtime_args_offset()`. The host emits the same order. This is
the deliberate cross-operation exception to MCAST-001's usual helper-tail rule;
it is preferable to carrying the same base both as a template argument and as a
constructor value.

Apply this to every existing dynamic-base consumer, not only Matmul. The audit
must include Matmul fused all-gather/reduce-scatter fields, the block-sharded
receiver tail, DRAM width-sharded in1 bank/stride arrays, descriptor and legacy
producers, and Conv3D reducer/worker coordinate arrays.

Resolution: API v14 removes the runtime-base constructor and stores no runtime
base in `McastArgs`. Five consumers now use compile-time bases: four Matmul
kernel layouts and the Conv3D writer. Their matching host producers emit fixed
operation fields, the helper block, then the runtime-sized fused/DRAM/worker
tail. The in1 DRAM-width-sharded parser now also advances across every emitted
bank/stride pair before decoding a fused-operation tail. A ledger-wide source
guard rejects a second runtime base and permits only the five registered,
derived variable-tail layouts. The release build, 27 source audits, 36 host
gtests, all 80 helper device tests under Watcher, focused 1D, block-sharded 2D,
and DRAM-width-sharded in1 Matmul nodes, and the focused Conv3D node passed;
Conv3D's known Watcher skip (#37184) passed unchanged without Watcher.

### MCAST-006 — Put multicast before a genuinely optional compile-time tail

- Date: 2026-08-23
- Status: Resolved (2026-08-23)
- Scope: Every migrated operation with an optional compile-time argument block
  after a fixed operation prefix

Do not emit dummy operation arguments solely to keep a multicast helper at the
end of a fixed compile-time ABI. When an operation-owned compile-time tail is
genuinely absent from one separately compiled kernel variant, order the wire as:

1. the fixed operation prefix;
2. the complete opaque multicast helper block;
3. the optional operation tail, when that variant uses it.

The kernel should derive the optional tail's first index from
`McastArgs::next_compile_time_args_offset()`, and the host should emit the same
order. This is a deliberate compile-time counterpart to MCAST-005's
variable-runtime-tail exception to MCAST-001. It keeps the helper block opaque
without requiring operation fields that have no behavior in the compiled
variant.

The first reviewed occurrence is the width-sharded Conv2D activation reader.
Its non-DRAM factory path currently emits two explicit zero fields and a null
two-word `TensorAccessorArgs` block only so the following activation-multicast
block starts at the same offset as in the DRAM variant. Place the multicast
block before the optional config-tensor address, page size, and accessor tail;
emit that tail only for `CONFIG_TENSOR_IN_DRAM`; and derive the config-tensor
indices from the multicast block's compile-time end.

Audit every other migrated operation for zero placeholders, null
`TensorAccessorArgs`, or other filler introduced only to stabilize a following
multicast offset. Apply this layout where the fields are truly absent in a
separate compile-time variant. Retain fixed optional descriptors when their
presence is itself part of the operation ABI or when kernel behavior consumes
the encoded absence; do not remove them merely because their values are zero.

Update the durable argument-ordering guardrail, which currently describes
multicast as always following operation compile-time arguments, to register
this optional-tail exception. Update
`test_migrated_kernels_keep_fixed_operation_prefixes_before_helper_decoders`
at the same time as the production layout: it should continue rejecting an
unexplained operation tail after `McastArgs`, but accept a registered optional
tail only when its first index is derived from
`McastArgs::next_compile_time_args_offset()`.

Resolution: the width-sharded Conv2D activation ABI now emits its 21-word
fixed operation prefix, the opaque activation-multicast block, and then the
DRAM config-tensor address, page size, and `TensorAccessorArgs` only when
`CONFIG_TENSOR_IN_DRAM` is compiled. The kernel derives all three tail indices
from `act_mcast_args.next_compile_time_args_offset()`; the non-DRAM factory no
longer emits four filler words. The ledger-wide audit found no other filler
introduced solely to stabilize a following migrated helper: the similar
sharded-Conv and SDPA descriptors are operation-owned fixed ABIs rather than
post-helper optional tails. The source audit registers this single exception
and rejects an unregistered compile-time tail. The release build, all 28 source
audits, the exact non-DRAM width-sharded node under Watcher (PCC 0.999956503),
and the exact DRAM-config node (PCC 0.998234911) passed; the DRAM node retained
its known unrelated Watcher/C++17 `ASSERT` compile incompatibility and passed
unchanged without Watcher.

### MCAST-007 — Chain offsets through the existing `McastArgs` object

- Date: 2026-08-23
- Status: Resolved (2026-08-23)
- Scope: Every kernel migrated to `mcast_pipe`

Prefer object-qualified offset chaining when a kernel already constructs a
named constexpr `McastArgs` object. For example, write:

```cpp
constexpr McastArgs<CT_BASE, RT_BASE> mcast_args;
uint32_t argidx = mcast_args.next_runtime_args_offset();
```

rather than introducing a type alias solely to make a type-qualified static
call:

```cpp
using WeightMcastArgs = McastArgs<CT_BASE, RT_BASE>;
constexpr WeightMcastArgs weights_mcast_args;
uint32_t argidx = WeightMcastArgs::next_runtime_args_offset();
```

Keep `next_compile_time_args_offset()` and `next_runtime_args_offset()` static
and constexpr: the boundaries belong to the compile-time wire type and must not
become runtime object state. C++ permits a static member to be called through
the existing constexpr object, which gives the clearer kernel spelling without
changing the helper API or undoing MCAST-005's single-source runtime-base rule.

Audit all migrated kernels and replace type-qualified offset calls with the
corresponding named object call. Declare chained helper objects in order so the
next helper type can use the preceding object's offsets. Remove a `using` alias
unless it independently names a nested `SenderPipe` or `ReceiverPipe` type; do
not create otherwise-unused helper objects merely to remove an alias. This is a
kernel call-site cleanup only; it requires no helper API version bump or host
ABI change.

Resolution: audited the complete migrated-kernel inventory and converted every
type-qualified offset call, including the Matmul, Conv3D, width-sharded Conv2D,
GroupNorm, and Move cases. All aliases without a nested pipe-type use were
removed; block-sharded Matmul, width-sharded Conv2D, Group Attention, and
Argmax retain only the aliases needed for `SenderPipe`/`ReceiverPipe`. A
fleet-wide source guard enforces both rules. The helper methods remain static
constexpr, API v14 and every wire ABI are unchanged. All 33 source audits
passed, as did focused Conv3D, Matmul, Conv2D, tiled/row-major Move,
sharded/interleaved legacy and Welford GroupNorm, and pre/post-allgather
LayerNorm gates. The interleaved GroupNorm parameters passed without `--dev`
after independently encountering the known Watcher/C++17 `ASSERT` compile
incompatibility while building `eltwise_typecast`.

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

## Conv

### CONV-001 — Preserve terminal write barriers in data-movement kernels

- Date: 2026-08-23
- Status: Resolved (2026-08-23)
- Scope: Conv data-movement kernels migrated to `mcast_pipe`

Do not remove the explicit `noc.async_write_barrier()` or equivalent terminal
barrier from a Conv data-movement kernel as part of the multicast migration.
The completion semantics inside `SenderPipe::send()` cover the transaction
owned by that helper call; they do not replace the kernel-wide drain of every
outstanding NoC write, including writes issued outside the multicast helper.

Preserve or reinstate the original barrier after the kernel's data-movement
loops and ensure that every exit following an issued write reaches it. Audit
the 1D and 2D sender and receiver variants rather than relying on the helper's
per-send synchronization as a substitute for the end-of-kernel barrier.

Resolution: compared all four migrated Conv2D weights kernels with their actual
pre-migration implementations and restored the terminal
`noc.async_write_barrier()` in every sender and receiver. Their only early exits
precede all issued writes, so every write-producing path reaches the drain. A
source guard requires the terminal barrier to remain the last kernel statement.
The exact height-sharded route passed under Watcher at PCC 0.999988205 and a
block-sharded sender/receiver route passed under Watcher at PCC 0.999944614.
An initially selected BFLOAT16 block-sharded parameter still exhibits an
independent pre-existing hang; removing only the restored 2D barriers reproduced
the same hang, while the BFLOAT8_B route proves the migrated 2D kernels compile
and execute with the barriers present.

### CONV-002 — Derive a multicast sender role from `McastArgs`

- Date: 2026-08-23
- Status: Resolved (2026-08-23)
- Scope: Conv data-movement kernels migrated to `mcast_pipe`

Audit every separately passed `is_sender_core` runtime argument. When the flag
means that the core is the sender for the same multicast family represented by
an `McastArgs`, remove the duplicate argument and query
`mcast_args.can_send()` instead. Remove the corresponding host-side argument
and update the runtime-argument offsets with it.

Do not mechanically replace flags that describe another role. In
`writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` and
`writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp`,
the current `is_sender_core` flag gates split-reader activation work, including
in the weights-multicast receiver kernel. That suggests it may describe
activation ownership rather than the weights multicast sender role. Confirm
the semantics from the host producer. If it is an independent role, rename it
to that precise role or derive it from the correct helper metadata; do not
leave it as an ambiguous multicast-looking `is_sender_core` boolean.

Resolution: audited every `is_sender_core` in migrated Conv code. The
width-sharded activation family already derives its same-family sender role
from `act_mcast_args.can_send()`, and Conv3D carries an explicit operation role
enum. In both block-sharded weights kernels, the scalar comes from
`input_cores.contains(core)` and gates split-reader activation work even on a
weights receiver; it is therefore independent input-shard ownership rather
than a duplicate weights-multicast role. Renamed the kernel and matching host
binding to `has_sharded_input` without changing ABI width or order. A source
guard rejects the ambiguous name in these kernels and the exact block-sharded
route passed under Watcher at PCC 0.999944614.

### CONV-003 — Remove migration-added streaming-source flushes

- Date: 2026-08-23
- Status: Resolved (2026-08-23)
- Kernel:
  `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp`
- Audit scope: All operations migrated to `mcast_pipe`

The migration added `weight_sources_are_persistent` and a conditional
`noc.async_writes_flushed()` before loading the next weight block. The
pre-migration kernel did not perform this mid-loop source-lifetime flush; it
issued the raw weight and ready-signal multicasts and retained only its terminal
write barrier.

Remove the migration-added persistence classification and conditional flush so
the migrated kernel retains the original synchronization behavior. Do not
replace it with the default per-send helper flush. Keep the weight and bias
sends caller-managed, and handle the original terminal barrier independently as
required by CONV-001.

Audit the other migrated operations for the same pattern. If a migration added
an `async_writes_flushed()`, write barrier, persistence classification, or
equivalent source-lifetime synchronization that the pre-migration kernel did
not have, remove that added behavior as well. Compare against each kernel's
actual pre-migration implementation rather than assuming that every explicit
flush in the migrated tree was part of the original operation contract.

Resolution: removed `weight_sources_are_persistent` and its conditional
mid-loop `noc.async_writes_flushed()` from the 1D weights sender. Both weight
and bias sends remain `SourceL1Guard::CallerManaged`, and the independently
restored original terminal write barrier remains. The cross-operation history
audit found this was the only migration follow-up commit that introduced a
production source-lifetime flush or persistence classification; current
barriers and flushes in other migrated operations were already part of their
operation contracts or unrelated protocols. A source guard fixes this policy,
all 31 audits at this gate passed, and the exact height-sharded route passed
under Watcher at PCC 0.999988205.

### CONV-004 — Derive dense ACK populations from multicast geometry

- Date: 2026-08-23
- Status: Resolved (2026-08-23)
- Scope: Conv2D and Conv3D multicast factories and kernels

Do not explicitly pass or retain an operation-owned count of receiver cores
that send consumer-ready acknowledgements when every non-sender core reached by
the multicast rectangle or line acknowledges. In that dense case, let
`Mcast1D` or `Mcast2D` derive the ACK population from the multicast geometry and
sender placement, and remove any duplicate host/kernel count ABI.

Use the block-sharded Conv2D weights multicast and the Conv3D weights multicast
as reference cases. The block-sharded weights grid is validated as dense and
`Mcast1D` derives the per-line `span - 1` count. Conv3D includes otherwise-idle
rectangle members as passive handshake participants, so `Mcast2D` derives
`area - 1` without a custom count. Preserve that formulation and remove any
remaining duplicate count associated with those families. The block-sharded
Conv2D activation multicast also carries a raw row/column destination count
that is geometric; when it is migrated to the helper, derive it from the line
rather than preserving the scalar ABI.

Do not mechanically remove an override from a divergent multicast. The
width-sharded Conv2D activation rectangle contains cores that return before the
handshake, and the height-sharded/default weights rectangle contains noop cores
that do not acknowledge. Those counts remain behavior-specific unless the
kernels and semaphore placement are deliberately changed so the extra landing
cores participate passively.

Resolution: audited every Conv2D and Conv3D helper construction and found the
reference dense families already use the required form. Block-sharded Conv2D
weights construct `Mcast1D` without an ACK override after asserting a single
dense zero-anchored output grid, so the helper derives `span - 1`. Conv3D's
template and per-group `Mcast2D` objects likewise omit an override, and idle
rectangle members retain their passive receive loops, so the helper derives
`area - 1`. The width-sharded activation and height/default weights overrides
remain because their landing and acknowledging populations diverge. The raw
block-sharded activation scalar remains unchanged because that family is still
deferred and was not authorized for migration by this feedback. A source guard
enforces all five dispositions. All 32 audits passed; the exact block-sharded
Conv2D route passed under Watcher at PCC 0.999944614, and the focused Conv3D
route passed at PCC 0.999991419 after its known Watcher skip (#37184).

## Group Attention Matmul

### GROUP-ATTN-MATMUL-001 — Use multicast role queries for rotating send and receive branches

- Date: 2026-08-23
- Status: Open
- Kernel:
  `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/reader_mcast_transformer_group_attn_matmul.cpp`

Use `in1_mcast_args.should_send(tile_row_id)` to select the sender for each
rotating multicast round. Do not identify the sender by comparing the local NoC
coordinates with `sender_x(tile_row_id)` and `sender_y(tile_row_id)`. That
coordinate comparison predates `McastArgs::should_send()` and now duplicates
the role and sender-phase information owned by the helper.

Likewise, remove the legacy-named `in1_sender_in_receiver_grid` local. The
branch is testing whether this core has the receiver role, not whether a sender
is geometrically inside the receiver grid, so express it directly with
`in1_mcast_args.can_receive()` (the kernel-side receiver-role query). The
resulting dispatch should have the form:

```cpp
if (in1_mcast_args.should_send(tile_row_id)) {
    // sender work
} else if (in1_mcast_args.can_receive()) {
    // receiver work
}
```

Keep the existing role-conditional construction of `SenderPipe` and
`ReceiverPipe`; this feedback only removes duplicated role selection and the
misleading legacy alias.

## SDPA Decode

### SDPA-DECODE-001 — Represent replicated-Q K sharing as 1D multicast families

- Date: 2026-08-23
- Status: Open
- Factory:
  `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`
- Affected path: Column-major, locally available replicated-Q MLA K multicast

Do not construct one fixed-sender `Mcast2D` host object for every individual
vertical K-sharing group. The groups form regular column families: for
`P = q_heads_parallel_factor`, each `P`-row band has the first core in every
column send to the other `P - 1` cores in that column. Represent each band with
one fixed-sender `Mcast1D` using `Mcast1DShape::PerColumn` and sender index zero.

For a configured `Gx × Gy` grid, the current nested loops construct

```text
N(Mcast2D) = Gx × (Gy / P)
```

objects, where this path requires `P > 1` and `Gy % P == 0`. The proposed
family representation needs only

```text
N(Mcast1D) = Gy / P
```

objects: one per `P`-row band, with each object deriving the `Gx` independent
column multicasts. Thus the host-object reduction is exactly a factor of `Gx`.
Only `num_active_cores / P` groups can perform work for a particular invocation;
the current host loop nevertheless creates objects for all `Gx × Gy / P`
grid groups and supplies zeroed reader arguments to idle cores.
For example, an `8 × 8` grid uses 16 `Mcast2D` objects at `P = 4`, but needs
only two `Mcast1D` families. At the current largest unharvested Blackhole
compute-with-storage grid (`13 × 10`), the maximum is 65 `Mcast2D` objects at
`P = 2`, reducible to five `Mcast1D` families. More generally, the operation
has no smaller fixed object-count cap than `Gx × Gy / P`; future grids scale
the current construction proportionally.

Preserve the existing behavior: one fixed sender at the top of each band and
column, the same `1 × (P - 1)` receiver population, no consumer-ready
handshake, the operation-owned K semaphore, and the existing caller-managed
source-lifetime/barrier policy. A program has one `P`, so its groups differ
only in coordinates, not in shape or protocol; that regularity is why a
per-band `Mcast1D` family is the natural host model.

## All-gather concat heads

### ALL-GATHER-CONCAT-001 — Reuse one completion semaphore across all concat receivers

- Date: 2026-08-23
- Status: Future feedback — migration not present on this branch
- Factory:
  `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/all_gather_concat_program_factory.cpp`
- Kernels:
  `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/llama_all_gather_concat_writer.cpp`
  and
  `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/llama_concat_reader.cpp`

> ⭐ **Future-only: this cannot be resolved on the current
> `sjovic/mcast-migration` branch because that branch does not contain the
> all-gather-concat helper migration. Keep this item for the future migration
> or when `sjovic/mcast-migration-multi-device` is reconciled; do not mark it
> resolved from the current legacy source.**

Model the required behavior as one event: after the global all-gather
completion count is satisfied, release the same 16 local concat receivers to
consume the remote rows. Do not treat the two legacy semaphore IDs as two
required protocol events merely because the current implementation publishes
both.

The legacy factory allocates `concat_semaphore_id` and
`concat_semaphore_id2`. The writer sets both to the same Flag value and
multicasts them back-to-back, at the same completion point, to the same three
dense receiver rectangles. The reader selects between the two IDs only through
the compile-time `ROWS_TO_READ` branch, while this factory passes the constant
`first_phase = 1`. Therefore the current generated reader waits only on the
first semaphore; the second publication has no distinct timing, payload,
receiver population, or active consumer in this factory.

When this path is migrated, first verify that no separately compiled or
out-of-tree variant supplies `ROWS_TO_READ = 2`. If that audit holds, retain one
operation-owned completion semaphore, pass that one ID to every concat reader,
and construct only one no-handshake Flag `Mcast2D` per existing receiver
rectangle:

```text
3 dense receiver rectangles × 1 completion semaphore = 3 Mcast2D objects
```

Do not construct six `Mcast2D` objects solely to preserve the legacy
`3 rectangles × 2 semaphore IDs` implementation. This is an operation cleanup,
not a missing helper capability: one semaphore ID denotes a separate local L1
cell on every participating core, each receiver resets its own cell after the
wait, and all receivers are observing the same completion event.

Validate the future cleanup on the target `test_concat_fuse_6u` 8×4,
four-device ring, `num_links=4` route. Keep the three receiver rectangles and
their 2/6/8 fan-outs unchanged; this feedback removes only the redundant second
semaphore family and its three duplicate signal publications.
