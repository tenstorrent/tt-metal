# `mcast_pipe` API feedback

This is the review log for the current `mcast_pipe` API. Record concerns here
before they become accepted design changes, and retain implemented decisions
when their contracts are important to future migrations. Document resulting
API revisions in `changelog.md`; use `api_feasibility.md` when a resolution
depends on census or production-kernel evidence.

Record issues in individual ports, including brittle CT/RT offset handling, in
The completed migration-specific review log is archived at
`archive/migration_feedback.md`; its durable rules are in `migration_guardrails.md`.

## Status values

- **Open** — feedback is recorded but no decision has been made.
- **Accepted** — the direction is agreed but may not be implemented yet.
- **Rejected** — the current API is retained; record the reason.
- **Implemented** — the accepted change is present in the helper and its callers.

## API-001 — Derive rotating span from the compile-time wire

- **Date:** 2026-08-04
- **Status:** Implemented
- **Surface:** `dataflow_kernel_lib::McastArgs`
- **Feedback:** `McastArgs` should need only the compile-time and runtime base
  offsets. The rotating sender span should be encoded in the compile-time
  arguments emitted by the host helper instead of supplied independently as a
  third template argument.

Current spelling:

```cpp
McastArgs<CT_BASE, RT_BASE, SPAN>
```

Preferred spelling:

```cpp
McastArgs<CT_BASE, RT_BASE>
```

### Rationale

`SPAN` describes the host-generated runtime wire: it selects fixed versus
rotating layout, determines the number of sender-coordinate pairs, and therefore
determines the next runtime-argument offset. Requiring the kernel call site to
repeat it creates two sources of truth that can disagree.

The value must be compile-time information because it determines
`ReceiverPipe`'s type and its `receive(round)` bounds check, but it does not
need to be a template argument supplied by the caller.
`get_compile_time_arg_val()` is constexpr and the decoder already uses values
read from the CT block as non-type template arguments.

### Implemented wire change

Add `rotating_span` to the uniform CT block:

```text
[active, data_ready_sem_id, consumer_ready_sem_id, num_active, flags, rotating_span]
```

- `rotating_span == 0`: fixed sender, four-word RT block.
- `rotating_span > 0`: rotating sender, `4 + 2 * rotating_span` RT words.

This cannot be derived generally from `num_active`: a divergent `Mcast2D` may
have fewer acknowledging receivers than its geometric span, and fixed and
rotating dense multicasts can have the same acknowledgement count.

### Resolution

Implemented on 2026-08-05 as `MCAST_PIPE_API_VERSION=10`. The host emits the
dedicated sixth CT word rather than packing the span into flags. `McastArgs`
now takes only `CT_BASE` and `RT_BASE` and derives fixed/rotating mode,
receiver type, runtime width, and both next offsets from `rotating_span`.

All 13 migrated kernels and 12 host bindings were audited after Gate 2 made
their helper boundaries opaque. Exact host-wire tests cover fixed, rotating,
divergent, and degenerate layouts; the complete helper device suite and all
mapped Matmul, Conv, GroupNorm, and Sort production inventories passed. A
durable source audit rejects any reintroduction of a third `McastArgs` template
argument.

## API-002 — Enforce the kernel's permitted sender/receiver face

- **Date:** 2026-08-04
- **Status:** Open
- **Surface:** host mcast wire and `dataflow_kernel_lib::McastArgs`
- **Trigger:** `activation_reader_width_sharded.cpp` constructs both
  `act_mcast_args.sender(noc)` and `act_mcast_args.receiver(noc)`. That is valid
  for its rotating protocol, but it raises the split-kernel case: what prevents
  a receiver-only kernel from constructing a sender pipe, or vice versa?

### Current behavior

The CT wire does not describe which face a kernel is allowed to construct.
Consequently, both methods are always available:

```cpp
auto sender = args.sender(noc);
auto receiver = args.receiver(noc);
```

For a fixed sender, the four RT words are a destination rectangle. For a fixed
receiver, the same four slots contain `[sender_x, sender_y, 0, 0]`. A mistaken
`sender()` call in a receiver-only kernel therefore interprets receiver data as
a rectangle; there is no compile-time diagnostic.

### Desired contract

The host binding should declare the permitted face for each kernel:

- **SenderOnly**
- **ReceiverOnly**
- **Both** — required for rotating protocols such as width-sharded Conv and for
  any single binary that legitimately performs either role.

This is host-known compile-time metadata and should ride on the CT wire rather
than become another caller-supplied `McastArgs` template argument. With a
constexpr face mask, `sender()` and `receiver()` should reject use of a face not
permitted by the kernel binding, preferably with `static_assert` when that
method is instantiated.

### Scope

This feedback tracks only compile-time sender/receiver-face enforcement. Keep
the current uniform CT/RT block sizes and role-neutral runtime layout. `Both`
permits construction of both faces; per-core runtime-role validation and RT
wire compaction are not part of this item.

### Resolution checklist

- Inventory sender-only, receiver-only, and Both kernels.
- Encode the permitted face in the existing self-describing CT metadata without
  changing the current block widths.
- Add negative compile tests for calling `sender()` on ReceiverOnly and
  `receiver()` on SenderOnly.
- Re-run the helper suites and all migrated production inventories.

## API-003 — Signal-only operations silently ignore handshake configuration

- **Date:** 2026-08-04
- **Status:** Implemented
- **Surface:** `SenderPipe::send_signal()` and
  `ReceiverPipe::receive_signal()`
- **Migration evidence:** sort's row-start control channel in
  `coordinator_single_row_multi_core.cpp` / `reader_single_row_multi_core.cpp`

### Current behavior

`McastConfig::handshake` is encoded as `McastArgs::pre_handshake`, and the data
operations honor it:

- `SenderPipe::send()` waits for and resets `consumer_ready`.
- `ReceiverPipe::receive()` increments `consumer_ready` before waiting for the
  data-ready signal.

The signal-only operations do not:

- `SenderPipe::send_signal()` immediately broadcasts the data-ready signal.
- `ReceiverPipe::receive_signal()` waits for that signal without sending a
  readiness acknowledgement.

Thus a Pipe configured with `handshake=true` silently behaves as
no-handshake when the caller chooses the signal-only methods. That is surprising
because handshake appears to be channel configuration carried by the CT wire,
not an operation-dependent exception.

### Production consequence

Sort's row-start sequence is structurally a handshaked control signal:

```text
worker readers -> ready increments -> coordinator wait/reset -> start multicast
```

The migration could not express that sequence with `send_signal()` /
`receive_signal()`, so it retained a raw `cores_to_coordinator_ready_sem` around
the migrated multicast. Later sub-stage signals intentionally have no readiness
handshake, so merely changing the one existing Pipe to handshaked would also be
incorrect.

### Expected direction

Make the signal-only operations consistent with the data operations:

- When the channel is configured with `handshake=true`,
  `receive_signal()` sends the consumer-ready acknowledgement and
  `send_signal()` waits for and resets the configured acknowledgement count
  before broadcasting the signal.
- When the channel is configured with `handshake=false`, both methods retain
  the current no-handshake Counter/Flag behavior.

The caller should choose whether a channel needs a handshake through
`McastConfig::handshake`; choosing `send_signal()` rather than `send()` must not
silently override that channel policy. Audit existing signal-only callers before
changing the semantics. The current production and test callers explicitly use
`handshake=false`, so their intended behavior should remain unchanged.

With that behavior, sort can use two host-described channels:

- a handshaked row-start Counter channel, whose `consumer_ready` semaphore
  replaces `cores_to_coordinator_ready_sem`; and
- a no-handshake sub-stage Counter channel.

The writer-to-coordinator done counter remains operation-owned: it counts
completed compare/write pairs and is not receiver readiness.

### Resolution

Implemented on 2026-08-05. `send_signal()` now waits for and resets the
configured consumer-ready count when `PRE_HANDSHAKE=true`; `receive_signal()`
acknowledges the current sender coordinate before waiting. The no-handshake
specialization is unchanged and emits no readiness traffic.

The control-only Counter matrix now runs both policies over 1x2/1x8 rectangles
and 2/32 rounds. The complete helper suite passed 77/77 from a cold cache.
Sort now has a handshaked row-start Counter Pipe and a separate no-handshake
sub-stage Counter Pipe. Its raw reader-ready semaphore was removed; the
writer-done counter remains operation-owned. Because no existing handshaked
signal caller depended on the old exception and no call-site spelling changed,
the rollout remains `MCAST_PIPE_API_VERSION=10`.

## API-004 — Support offset grids in `Mcast1D`

- **Date:** 2026-08-04
- **Status:** Implemented
- **Surface:** `ttnn::kernel_lib::host::Mcast1D`
- **Production evidence:** the in1 multicast family in
  `matmul_multicore_reuse_mcast_2d_program_factory.cpp`

### Current limitation

`Mcast1D` requires its grid bounding box to start at logical core `(0,0)` and
derives its dimensions directly from the absolute end coordinate. This prevents
using it for otherwise ordinary row/column multicast families placed on a
non-zero subdevice origin.

The matmul-2D host path works around that restriction by constructing a
`std::vector<Mcast2D>`, one object per in1 multicast line. Every object has the
same CT configuration and adopted semaphore IDs; only the per-line runtime
geometry differs. The factory consequently takes CT args from
`in1_mcasts.front()` and selects one vector entry only to obtain RT args.

### Required behavior

Allow `Mcast1D` over any single rectangular logical-core range. Store the
rectangle origin and dimensions separately, and interpret fixed sender indices
relative to the broadcast axis of that rectangle:

- `PerRow`: sender x-position is relative to `bounding_box.start_coord.x`.
- `PerColumn`: sender y-position is relative to
  `bounding_box.start_coord.y`.
- Uniform and diagonal placement operate on relative positions and translate
  to absolute logical coordinates only when producing/querying a core.
- `runtime_args(core)`, `is_sender(core)`, and line indexing account for both
  x/y origin offsets.
- Semaphore ownership covers exactly the supplied grid.

The zero-origin case must remain behaviorally identical.

### First production migration test: matmul-2D

Use matmul-2D's in1 family as the first migration after implementing offset-grid
support. Its existing vector should become one `Mcast1D` over
`all_cores_with_work`:

```cpp
const auto shape = transpose_mcast ? Mcast1DShape::PerRow
                                   : Mcast1DShape::PerColumn;
Mcast1D in1_mcast(
    device,
    CoreRangeSet(all_cores_with_work),
    shape,
    /*relative sender index=*/0,
    config);
```

This migration is a strong acceptance test because it covers both multicast
orientations, non-zero subdevice origins, one shared CT/semaphore family, and
per-line host-generated RT geometry. Use `all_cores_with_work`, not the
potentially enlarged `all_cores` used to accommodate extra in0-sharded cores.

### Resolution checklist

- Add host-unit coverage for offset `PerRow` and `PerColumn` grids on both NoCs.
- Cover uniform and diagonal fixed senders on an offset grid.
- Verify sender/receiver RT coordinates, sender classification, line indexing,
  semaphore ranges, and degenerate one-core spans.
- Replace both legacy and descriptor matmul-2D in1
  `std::vector<Mcast2D>` constructions with one `Mcast1D`.
- Apply MIG-003's semaphore ownership and opaque CT/RT block insertion while
  changing the matmul binding.
- Rebuild host code, run one compile-focused matmul-2D parametrization, then
  sequentially run both transpose orientations and the mapped matmul inventory.
- If caller-facing semantics change, bump `MCAST_PIPE_API_VERSION` and record
  the result in `changelog.md`.

### Resolution

Implemented in Gate 3. `Mcast1D` now accepts any dense rectangular
`CoreRangeSet`, retains its logical origin, and derives sender placement, line
indices, coordinates, multicast bounds, and semaphore coverage relative to
that origin. Sparse sets are rejected by comparing their size with the
bounding rectangle.

Both legacy and descriptor matmul-2D paths now use one offset-aware `Mcast1D`
over `all_cores_with_work`, selecting `PerRow` or `PerColumn` from
`transpose_mcast`. Host coverage exercises offset grids on both NoCs, uniform
and diagonal placement, coordinate and line-index lookup, semaphore ranges,
and one-core degeneracy. `McastHostFixture` passed 25/25; focused 1D and both
2D orientations passed; and `MM-IN1-ALL` passed 302 cases with 188 expected
skips. The change did not alter the wire, so the API remained v9 until the
independent self-describing-wire v10 bump in Gate 4.

## API-005 — Make payload source-L1 lifetime explicit per send

- **Date:** 2026-08-04
- **Status:** Implemented
- **Surface:** `dataflow_kernel_lib::SenderPipe::send()`
- **Trigger:** the SDXL VAE Conv migration regressed by 2.730%, prompting review
  of why `send()` calls `async_writes_flushed()`.

The flush is not required to order multicast data before its linked ready
signal. It is a source-lifetime guard: it makes `src_l1` safe to overwrite or
reuse when `send()` returns. Some kernels keep the source immutable through a
later barrier or kernel completion, so forcing SENT completion after every send
unnecessarily serializes their hot path.

The accepted API is an opt-in method-template policy:

```cpp
pipe.send(src_l1, dst_l1, size);  // guarded; source is reusable on return
pipe.send<SourceL1Guard::CallerManaged>(src_l1, dst_l1, size);
```

For `CallerManaged`, the caller must keep the payload source unchanged until a
later NoC completion point. The policy cannot be inferred from the addresses:
it depends on the caller's CB/storage lifetime. It skips only the remote-only
SENT fence and does not weaken real-loopback destination completion,
rotating-Flag signal-source lifetime, or Counter atomic acknowledgement
requirements.

`MCAST_PIPE_API_VERSION` remains 9 because existing callers retain their exact
spelling and guarded semantics; only an explicitly opted-in call changes its
contract.

### Implementation and validation

The height-sharded Conv sender opts into caller-managed lifetime for weights
and bias. Bias is immutable. Fully buffered weight sources are not reused, and
the streaming configuration flushes once at the actual CB overwrite boundary.
The complete send hot path is `FORCE_INLINE`.

With three warmups and 20 real-time-profiler records, the SDXL VAE median
improved from 28,719.126 ns to 28,161.499 ns, within +0.736% of the 27,955.899
ns reverse pre-migration baseline. The exact nightly correctness node passed at
PCC 0.9999325 against 0.985, and the complete helper suite passed 73/73 under
`--dev`, including loopback and rotating cases.

## API-006 — Let `ReceiverPipe` borrow stable sender coordinates

- **Date:** 2026-08-04
- **Status:** Implemented
- **Surface:** `dataflow_kernel_lib::McastArgs::receiver()` and `ReceiverPipe`
- **Trigger:** the SegFormer width-sharded Conv migration regressed by 1.761%; a
  rotating `SPAN=18` receiver copied the same 36 RT coordinate words twice
  during pipe construction.

`McastArgs::receiver()` now constructs `ReceiverPipe` with a non-owning pointer
directly into its RT coordinate block. `ReceiverPipe` retains that view instead
of owning another fixed-size array.

This requires an explicit lifetime contract: the coordinate storage must
outlive the pipe. `McastArgs` satisfies it because `get_arg_addr()` addresses
the kernel's RT block in L1, which remains allocated and unchanged throughout
the kernel invocation. A direct `ReceiverPipe` caller must keep its own array
alive through every pipe use. The helper tests' by-hand receiver does so by
declaring the array in `kernel_main()` before constructing and using the pipe.

The change does not remove `NUM_SENDERS` from the pipe type: it still provides
the compile-time bound for `receive(round)` and remains tied to API-001's span
discussion. It only removes redundant coordinate storage and copies.

`MCAST_PIPE_API_VERSION` remains 9 because existing array arguments decay to
the accepted pointer and require no call-site rewrite. The ownership/lifetime
semantics changed and are documented on the constructor.

### Implementation and validation

The complete helper suite passed 73/73 under `--dev`, including rotating spans
and direct construction. The exact 576-channel SegFormer width-sharded nightly
node passed at PCC 0.9998909 against 0.985.

Three independent real-time-profiler medians were 38,362.905, 38,377.304, and
38,414.444 ns. Their median, 38,377.304 ns, is +0.958% versus the immediate
pre-migration parent at 38,013.031 ns and improves on the migrated 38,682.593
ns result (+1.761%).

## API-007 — Let Flag signals carry caller-supplied control values

- **Date:** 2026-08-06
- **Status:** Implemented
- **Surface:** `dataflow_kernel_lib::SenderPipe::send_signal()` and
  `ReceiverPipe::receive_signal()`
- **Migration evidence:** Matmul in0 sparsity batch-validity exchange in
  `reader_bmm_tile_layout_in0_sender_padding.cpp` /
  `reader_bmm_tile_layout_in0_receiver.cpp`

### Required behavior

The Matmul in0 semaphore cell carries both ordinary data-ready notifications
and a three-state batch-validity control value: `INVALID` (0), `VALID` (1), or
`IGNORE_BATCH` (2). The receiver waits for a non-zero value and branches on the
observed value. Replacing this with a payload transfer would add packets and L1
storage to the skipped-batch path, so the helper must preserve the existing
one-word semaphore protocol.

Extend the Flag specialization so the sender can provide a small non-zero
value while retaining `VALID` as the default:

```cpp
sender.send_signal(value);  // value defaults to VALID
const uint32_t value = receiver.receive_signal();
```

The receiver waits for `>= VALID`, returns the observed value, and clears the
Flag to `INVALID` exactly once. Counter mode remains monotone and does not gain
typed values. Handshake behavior remains governed solely by the channel's
existing configuration.

### Resolution

Implemented on 2026-08-06 as `MCAST_PIPE_API_VERSION=11`. Flag
`send_signal(value = VALID)` writes the caller-supplied non-zero value before
the existing signal multicast. Flag `receive_signal()` waits for `>= VALID`,
returns the observed value, and clears the cell to `INVALID` exactly once.
Counter remains a monotone `+1` event channel and requires the default
argument. Handshake behavior and the host wire are unchanged.

Fresh-JIT focused cells passed for default `VALID` and `IGNORE_BATCH`; the
complete helper device suite passed 79/79 and `McastHostFixture.*` passed
28/28. The existing v10 fleet must now be remigrated as Tier 0 before net-new
Matmul work resumes.

## API-008 — Use Counter mode for race-free TopK readiness

- **Date:** 2026-08-06
- **Status:** Implemented
- **Surface:** existing no-handshake `DataReadyMode::Counter` signal channel
- **Migration evidence:** multicore TopK final-coordinator readiness exchange

### Resolution

No new helper feature is required. TopK's readiness notification is a repeated
event, not a level whose historical `VALID`/`INVALID` spelling must be
preserved. Express it with the existing no-handshake Counter channel, adopt the
host-created readiness semaphore initialized to zero, and remove per-round
receiver clears. The worker-to-coordinator arrival counter remains
operation-owned.

This formulation both avoids constructor initialization of a remotely written
Flag and closes the existing lost-wakeup window in which a worker can clear the
next round's `VALID` after its prior-round arrival has released the
coordinator. The helper API and wire remain at v10 for this migration.

## API-009 — Decouple rotating senders from the receiver rectangle

- **Date:** 2026-08-07
- **Status:** Implemented
- **Surface:** `ttnn::kernel_lib::host::Mcast1D`, `Mcast2D`, and the rotating
  host/runtime wire
- **Migration evidence:** block-sharded Matmul in0 multicast

### Current limitation

The rotating helpers derive the sender set from the multicast rectangle:
`Mcast1D` rotates over every core on the line, while `Mcast2D` requires a
rotating sender to be inside its rectangle. They therefore cannot represent
the original block-sharded Matmul topology when the shard has more sender
cores than the output has receiver cores: later senders are outside the fixed
output receiver rectangle.

Expanding the receiver rectangle to include those senders is not an equivalent
migration. It makes shard-only cores receive and acknowledge every other
sender's rounds, increases the payload and semaphore fan-out, and turns
formerly external senders into loopback senders.

### Required behavior

Support an ordered rotating sender set that is independent of the fixed
receiver rectangle. The protocol must derive or encode the correct per-sender
acknowledgement count and include/exclude-source mode: a sender inside the
receiver rectangle waits for all other receivers, while a sender outside it
waits for every receiver and does not loop data back to itself.

Matmul should restore its original rectangles after this capability exists;
until then, the API limitation must remain explicit rather than being hidden by
widening the multicast geometry.

### Resolution

Implemented on 2026-08-07 without changing the v11 device wire. `Mcast1D` now
accepts one explicit ordered sender sequence per receiver line, and `Mcast2D`
accepts one explicit ordered sender sequence for its fixed rectangle. Sender
sequences may contain cores outside the receiver rectangle; helper-owned
semaphores cover the union of receivers and senders. Empty, duplicate,
misaligned, non-dense, and inconsistent line configurations fail on the host.

The existing rotating RT layout already carries a fixed destination rectangle
followed by ordered sender coordinates, so no wire expansion was needed. The
host emits the existing `0xFFFFFFFF` dense-fan-out sentinel for the ACK field.
`SenderPipe` then derives the correct count per round: rectangle area minus one
for an inside sender and the full rectangle area for an outside sender.

Block-sharded Matmul now supplies its shard sender order separately while
retaining the original output-work receiver rectangles in both legacy and
descriptor builders. A focused device case rotates across an inside sender and
an outside sender for multiple rounds and verifies payload order plus handshake
completion; complete helper normal and Watcher suites passed 80/80. Host wire,
validation, role-query, and semaphore-union coverage passed 30/30. The full
Matmul suite passed 816 tests with 310 expected skips and 2 known xfails, and
the sparse Matmul suite passed 18/18 after its shared-kernel ABI bindings were
audited. Matched 3-warmup/20-record Blackhole performance deltas versus
`4a1d6a97ca9` were +0.643% (2D SDXL), +0.809% (1D SDXL), and -0.045%
(transposed 2D), all within the 1.5% gate.
