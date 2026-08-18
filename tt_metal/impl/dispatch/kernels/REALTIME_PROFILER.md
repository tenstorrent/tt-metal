# Real-Time Profiler: Architecture and Performance Analysis

## Overview

The real-time profiler observes correlated per-program device intervals during
execution. dispatch_s captures the start tick, a resident TRISC0 observes the
matching completion counter, and the existing reserved-core/D2H path delivers
both endpoints to callbacks. Every firmware queue is bounded and profiler
pressure drops and counts data instead of blocking application progress. Trace
replay is intentionally unprofiled.

### Components

| Component | File | Core | NOC |
|-----------|------|------|-----|
| **dispatch_s** (signal source) | `cq_dispatch_subordinate.cpp` | NCRISC | NOC 1 |
| **completion observer** | `cq_dispatch_subordinate_compute.cpp` | dispatch_s TRISC0 | register space / local L1 |
| **BRISC reader** (fast path) | `cq_realtime_profiler.cpp` | reserved profiler tensix BRISC | NOC 0 |
| **NCRISC pusher** (slow path) | `cq_realtime_profiler_push.cpp` | reserved profiler tensix NCRISC | NOC 1 |
| **host manager** | `realtime_profiler_manager.cpp` | CPU threads | PCIe |

The data mover is split across two RISCs on a reserved dedicated tensix core — an otherwise-unused core taken from the back of the dispatch core pool.
The BRISC reader pulls timestamps off dispatch_s and drops them into an L1 ring
buffer; the NCRISC pusher drains that ring to the host over PCIe. Splitting the
work this way decouples the fast NOC read from the PCIe push, so that transient bursts
can be absorbed without dropping records.

### Data Flow

```
dispatch_s       TRISC0 observer       BRISC reader       NCRISC pusher       host
    |                    |                   |                   |              |
    |-- descriptor ----->|                   |                   |              |
    |-- GO               |                   |                   |              |
    |                    |-- completed ----->|                   |              |
    |<-- bounded service-|   ring            |                   |              |
    |-- A/B payload + PUSH_A/PUSH_B -------->|                   |              |
    |                    |                   |-- L1 ring ------->|              |
    |                    |                   |                   |-- D2H page -->|
    |                    |                   |                   |              |-- callbacks
```

## Concurrent Device-Local Protocol

For each accepted ordinary GO, dispatch_s reserves a slot in that sub-device
stream's depth-four SPSC ring, captures one device-clock tick and stages local
payload immediately before GO, then publishes the producer index and wakeup
epoch immediately after the first GO write. The descriptor includes runtime
identity, completion target, generation, and start tick; its ring supplies the
stream identity. A resident TRISC0 observer stays parked
until the profiler is activated, tracks only active streams, and samples the
same device clock on the first poll that sees a target complete. It publishes
the correlated interval to a bounded 64-record device-local SPSC ring.

Every producer path is one-shot: a full descriptor or completed-record ring
increments its named loss counter and drops the record. No profiler path waits
for space. A generation handshake around a real `CLEAR_STREAM` prevents a
queued descriptor from crossing the counter-reset window. Both timing
endpoints are device ticks; host timing and tensor readback are never fallback
endpoints.

## Public Transport

dispatch_s checks the completed-publication scratch register at fixed safe
service points. A clear signal returns without invalidating L1. When a record is
pending and the A/B transport is acknowledged, one service action invalidates,
copies exactly one eight-word interval, publishes the low-24-bit completed-ring
consumer index, and notifies the reserved BRISC. The BRISC acknowledges only
after consuming or explicitly dropping that payload, so dispatch_s never
overwrites an in-flight mailbox. There is no program-ID FIFO and no profiler
command emitted by ordinary `Finish`.

The **BRISC reader** polls its state mailbox. On `PUSH_A`/`PUSH_B` it issues a
`noc_async_read` of the 32-byte transport record from the indicated dispatch_s
buffer into the next ring slot, composes its cumulative device-ring loss, then
advances `write_index`. If the ring is full it increments `device_ring`, drops
the interval, and acknowledges immediately. The reader also services host
clock-sync requests, enqueueing sync-marker records into the same ring.

The **NCRISC pusher** owns the slow PCIe path. Each iteration it snapshots
`write_index`/`read_index`, and if the ring is non-empty it pushes *all*
available entries in one `push_entries_to_host` call, then advances `read_index`
by the number drained.

`push_entries_to_host` reserves the pages in the D2H socket, then issues
coalesced NOC writes over PCIe — up to `NOC_MAX_BURST_SIZE` per write, chunked at
ring-wrap, host-FIFO-wrap, and burst-size boundaries — followed by a single
`socket_push_pages` + `socket_notify_receiver` + `noc_async_write_barrier`.

## Measured Timing

### Signal cost (dispatch_s side)

| Metric | Value |
|--------|-------|
| Full enabled M1 GO-tail increment over the same binary disabled | **128 cycles (~95 ns at 1.35 GHz)** |
| Disabled M1 GO-tail median versus exact clean baseline | **817 versus 927 cycles** |
| Full enabled M3 GO-tail increment over the same binary disabled | **161 cycles (~119 ns at 1.35 GHz)** |
| M3 device observation error, median GO lead / polling lag | **568 / 370 cycles** |
| M3 observer loop, idle / one active / eight active streams | **24 / 90 / 464 cycles** |
| M3 production observer TRISC0 text | **1,848 / 2,048 bytes** |

### Push cost (NCRISC pusher side)

| Metric | Value |
|--------|-------|
| `push_entries_to_host` per drain | **~420 ns** |

### Throughput

The signal-to-record path uses bounded descriptor/completed queues followed by
a fast NOC read into the deep reserved-core L1 ring. The PCIe pusher drains
pending entries in coalesced bursts. None of these queues claims losslessness
under unbounded pressure; every capacity failure has a monotonic stage-specific
counter and no interval producer waits for space.

## Implementation Notes

- The host side runs a receiver thread that drains device→host pages and
  publishes decoded records onto a `BroadcastRing`; separate per-callback
  consumer threads read from the ring and invoke the registered callbacks. A slow
  callback only drops records for that consumer (tracked in `Consumer::dropped`);
  it never stalls page draining or dispatch.
- Shutdown asks the observer to stop, waits only within a fixed cycle budget,
  forwards at most the 64-record completed-ring capacity, and counts abandoned
  descriptors, records, or stop timeouts. It does not publish a dummy record or
  substitute a host timestamp.
