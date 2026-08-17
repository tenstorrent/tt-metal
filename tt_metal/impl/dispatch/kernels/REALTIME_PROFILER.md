# Blackhole Real-Time Profiler Device Path

## Scope

The real-time profiler is enabled only on Blackhole with worker
dispatch and one command queue. It records device wall-clock start and end
ticks for programs on independent dispatch streams. Wormhole, Quasar,
multi-command-queue, trace-replay concurrency, and model integration are not
part of this feature; profiler eligibility rejects those architectures.

## Device pipeline

| Stage | Producer | Consumer | Full behavior |
| --- | --- | --- | --- |
| Per-stream start queue | dispatch_s NCRISC | dispatch_s TRISC0 | Count `start_descriptor_drop_count`; launch continues |
| Completed interval queue | dispatch_s TRISC0 | dispatch_s NCRISC | Count `completed_record_drop_count`; observation continues |
| Per-stream watermark slot | dispatch_s NCRISC | dispatch_s TRISC0 / dispatch_s NCRISC | Count `watermark_request_drop_count`; dispatch continues |
| Mailbox B | dispatch_s NCRISC | profiler-core BRISC | Leave record queued until a later bounded service point |
| Reserved-core L1 ring | profiler-core BRISC | profiler-core NCRISC | Intervals reserve one control slot and count `transport_drop_count`; watermarks retain mailbox ownership until the control slot is available |
| D2H socket | profiler-core NCRISC | host receiver | May backpressure the reserved profiler core, never dispatch |

## Component map

| Component | File | Core / transport |
| --- | --- | --- |
| Start publisher and record forwarder | `cq_dispatch_subordinate.cpp` | dispatch_s NCRISC, NOC 1 |
| Reset-epoch publisher and ID producer | `cq_dispatch.cpp` | dispatch_d BRISC, shared L1 |
| Completion observer | `cq_dispatch_subordinate_compute.cpp`, `cq_realtime_profiler_dispatch_subordinate.hpp` | dispatch_s TRISC0, shared L1 |
| Counter/wrap helpers | `realtime_profiler_protocol.hpp` | host tests and device firmware |
| Mailbox reader | `cq_realtime_profiler.cpp` | reserved profiler BRISC, NOC 0 |
| D2H pusher | `cq_realtime_profiler_push.cpp` | reserved profiler NCRISC, NOC 1 / PCIe |
| Host manager | `realtime_profiler_manager.cpp` | receiver and callback threads |

dispatch_s receives the exact worker contribution in each go command. Just
before sending the go signal, it publishes a descriptor containing the runtime
ID, device start tick, stream, reset generation, and modular completion target:

```text
target = (previous completion count + workers in this launch) mod 2^17
```

TRISC0 continuously samples every configured completion stream. When a target
is reached using the stream counter's half-range modular comparison, TRISC0
captures the end tick and publishes an eight-word completed interval. If
multiple targets became ready between samples, only the newest can receive an
accurate sampled end tick; the ambiguous older intervals are counted as
completion-observer drops.

dispatch_s calls `service_realtime_profiler_once()` at existing progress
points. One call forwards at most one completed interval or watermark and returns immediately
when mailbox B is occupied. There is no steady-state profiler drain or
acknowledgement loop on dispatch_s.

On Blackhole an inline write to profiler-core L1 uses the NCRISC spoof command
buffer. The default multicast go path therefore defers profiler service while
`wait_for_workers()` runs between `init_state` and `with_state`; its next normal
progress point services any queued record. Device-print builds retain their
existing wait-before-`init_state` ordering. Neither progress path can overwrite
live go-signal NOC state.

The profiler-core BRISC NOC-reads the mailbox-B payload. Intervals treat the
ring as full with one slot remaining, count a transport drop, and acknowledge
the mailbox. That reserved slot carries an ordered watermark. If it is
temporarily occupied, BRISC retains the watermark notification until NCRISC
makes space; dispatch_s does not wait for the acknowledgement. The NCRISC
remains responsible for coalesced D2H socket writes and may wait for host FIFO
space without propagating that wait back to application dispatch.

The NCRISC snapshots the reserved-core ring indices and sends every available
entry in a coalesced `push_entries_to_host()` call. Transfers are split only at
ring wrap, host-FIFO wrap, and the NOC burst limit, then committed with one
socket page notification and NOC write barrier.

The host receiver drains D2H pages, decodes interval and sync records, and
publishes intervals to a `BroadcastRing`. Each registered callback has its own
consumer thread. A slow callback increments only that consumer's drop count; it
cannot stall the socket receiver, reserved profiler core, or application
dispatch.

## Ordering and cache visibility

Each local queue is single-producer/single-consumer. The producer writes the
payload, executes `fence w,w`, and publishes its write index last. Blackhole has
no separate uncached L1 alias, so a cross-RISC consumer invalidates its L1 cache
before reading the producer index and invalidates again before reading a reused
payload slot.

The existing NOC barriers order mailbox-B payload reads and reserved-ring
publication. The dispatch_s inline state notification is sent only after the
32-byte mailbox-B payload is stored and fenced. The payload remains 16-byte
aligned in the shared message; firmware compile-time checks reject a layout or
generated address that would make the Blackhole NOC read align down to the
wrong words.

## Stream reset and wrap

Natural 17-bit completion-counter wrap is forward progress. It does not reset
profiler state. For an explicit `CLEAR_STREAM`, co-located dispatch_d increments
the stream's reset generation after clearing the hardware counter. Descriptors
carry that generation; TRISC0 consumes and counts stale-generation descriptors
instead of correlating them with the reset counter.

Queue producer and consumer indices use unsigned 32-bit distance, with bounded
capacities far below the half range.

## Finish watermark

Every explicit exact collection allocates one nonzero 32-bit watermark ID and
emits a `CQ_DISPATCH_CMD_RT_PROFILER_FLUSH` carrying that ID for each selected
stream.
After the existing worker wait, dispatch_s publishes a per-stream request.
TRISC0 completes it only after the target is reached and every preceding start
descriptor was emitted or counted as loss. Descriptors for later completion
targets do not delay the watermark. TRISC0 snapshots the completed-record
producer index, successful sequence, reset generation, and cumulative
descriptor-, observer-, and record-stage loss. dispatch_s forwards that
watermark only after its completed-record consumer reaches the snapshot, and
prioritizes a now-eligible watermark over records accepted after the snapshot.

A request from a reset generation newer than TRISC0's adopted generation waits
for the next observer pass. A request from an older, already-quiesced generation
uses a dedicated protocol-error watermark marker. Watermark request/protocol
counters remain control-plane diagnostics; they are not folded into the
descriptor-stage interval-loss delta.

The host consumes watermark pages internally. A batch is complete only after
the exact ID was observed for every registered stream on every active device;
FIFO or ring emptiness is never completion evidence. Normal Finish sends the
existing flush with watermark ID zero, so it neither creates collection state
nor executes the device watermark path.
`FinishAndCollectProgramRealtimeProfiler()` explicitly waits with a host
control-plane timeout and returns stream masks plus source/transport loss
deltas. Source loss is also reported as descriptor, completion-observer, and
completed-record deltas. The command-queue identity field is fixed at zero for
this single-command-queue protocol. It does not create host-derived operation
durations.

Watermarks already armed for a stream remain serviceable if a later sub-device
manager temporarily reduces the active stream count: the stage scanner covers
the fixed eight-stream protocol domain and uses the pending mask to avoid work
on idle streams.

## Termination

After application work is quiesced, dispatch_s requests the TRISC0 completion
observer to stop and waits only for a device-cycle-bounded acknowledgement. It
then performs an item- and device-cycle-bounded handoff attempt. This
post-quiescence loop may poll an occupied mailbox B for its IDLE acknowledgement
until the device-cycle deadline so the last accepted record can advance. Any
completed records or descriptors still queued after a successful observer stop
are counted before dispatch termination proceeds. Observer-stop timeout is
reported separately. If profiler-core termination finds the final mailbox
occupied while its ring is completely full, it counts that transport loss
before acknowledging the mailbox and exiting.
After the completion observer stops, any request or ready watermark still in a
per-stream slot is counted as watermark-request loss before dispatch exits.

## Resource bounds

- Per-stream start depth: 4 descriptors across at most 8 streams.
- Completed interval depth: 128 records.
- Dispatch-core profiler message: 5,536 bytes.
- Reserved profiler-core L1 layout: unchanged at 262,336 bytes.

The protocol, ordering proof, rejected alternatives, and milestone gates are
documented in `docs/realtime_profiler_concurrent_subdevice_protocol.md`.
