# `mcast_pipe` API feedback

This is the intake log for review feedback on the current `mcast_pipe` API.
Record concerns here before they become accepted design changes. Once an item is
resolved, preserve the outcome here and document any resulting API revision in
`changelog.md`; use `api_feasibility.md` when the resolution depends on census or
production-kernel evidence.

Record issues in individual ports, including brittle CT/RT offset handling, in
`migration_feedback.md`.

## Status values

- **Open** — feedback is recorded but no decision has been made.
- **Accepted** — the direction is agreed but may not be implemented yet.
- **Rejected** — the current API is retained; record the reason.
- **Implemented** — the accepted change is present in the helper and its callers.

## API-001 — Derive rotating span from the compile-time wire

- **Date:** 2026-08-04
- **Status:** Open
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
`ReceiverPipe`'s type and coordinate storage, but it does not need to be a
template argument supplied by the caller. `get_compile_time_arg_val()` is
constexpr and the decoder already uses values read from the CT block as
non-type template arguments.

### Candidate wire change

Add `rotating_span` to the uniform CT block:

```text
[active, data_ready_sem_id, consumer_ready_sem_id, num_active, flags, rotating_span]
```

- `rotating_span == 0`: fixed sender, four-word RT block.
- `rotating_span > 0`: rotating sender, `4 + 2 * rotating_span` RT words.

This cannot be derived generally from `num_active`: a divergent `Mcast2D` may
have fewer acknowledging receivers than its geometric span, and fixed and
rotating dense multicasts can have the same acknowledgement count.

### Resolution checklist

- Decide whether to add a sixth CT word or pack the value into the unused flag
  bits.
- Audit CT-block construction and any manually calculated downstream CT
  offsets.
- Audit rotating call sites and remove their third `McastArgs` template
  argument.
- Update helper tests for fixed, dense rotating, and divergent rotating wires.
- If accepted, bump `MCAST_PIPE_API_VERSION` and record the change in
  `changelog.md`.

## API-002 — Encode and enforce the kernel's permitted mcast face

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
a rectangle; there is no compile-time or runtime diagnostic.

The current fixed receiver wire does not carry an actual sender rectangle, but
it is still padded to four words even though only two are meaningful. The
rotating union wire carries both a four-word rectangle and all sender-coordinate
pairs, so a split sender-only or receiver-only kernel receives a larger block
than its face needs.

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

For a `Both` binary whose role varies per core at runtime, a CT-only assertion
cannot detect calling the wrong face on an individual core. Decide separately
whether such wires need a per-core role tag and a debug/runtime assertion.

### Wire-size question

Role enforcement and wire compaction are related but separable:

1. The minimum change is to add permitted-face metadata while retaining the
   current uniform RT union layout. This provides misuse detection without
   complicating offsets.
2. A stronger change makes the host emit face-specific RT projections:
   fixed sender = rectangle, fixed receiver = one coordinate pair, rotating
   sender = rectangle, rotating receiver = sender-coordinate list, and Both =
   their union. That avoids sending unused fields but requires
   `compile_time_args` and `runtime_args` to be emitted for a specific kernel
   binding rather than from a role-neutral helper object.

The safety contract should be decided independently of whether the saved RT
words justify face-specific layouts.

### Resolution checklist

- Inventory sender-only, receiver-only, Both, and per-core mixed-role kernels.
- Decide whether permitted face is a dedicated CT word or part of a general
  self-describing mcast metadata word alongside API-001's rotating span.
- Add negative compile tests for calling `sender()` on ReceiverOnly and
  `receiver()` on SenderOnly.
- Decide whether mixed-role binaries need a per-core runtime assertion.
- Measure the value and complexity of compact face-specific RT layouts before
  coupling that optimization to role enforcement.

## API-003 — Signal-only operations silently ignore handshake configuration

- **Date:** 2026-08-04
- **Status:** Open
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

### Resolution checklist

- Inventory every `send_signal()` / `receive_signal()` caller and its configured
  handshake flag.
- Make the existing signal-only methods honor the configured handshake policy.
- Add positive and negative tests proving signal-only behavior for both
  handshake modes.
- Re-evaluate the sort migration and its `fully end-to-end` classification.
- If the API changes, bump `MCAST_PIPE_API_VERSION` and update `changelog.md`.

## API-004 — Support offset grids in `Mcast1D`

- **Date:** 2026-08-04
- **Status:** Open
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
