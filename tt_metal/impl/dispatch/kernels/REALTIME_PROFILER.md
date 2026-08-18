# Real-Time Profiler: Architecture and Performance Analysis

## Overview

The real-time profiler streams per-program device-side timestamps to the host
during execution. In clean-room Milestone 1, each profiled ordinary GO produces
a correctly identified device start tick and `end_timestamp == 0`. Tracy skips
these start-only records; Milestone 2 restores duration zones with a correlated
completion-gated endpoint. Trace replay is intentionally unprofiled.

### Components

| Component | File | Core | NOC |
|-----------|------|------|-----|
| **dispatch_s** (signal source) | `cq_dispatch_subordinate.cpp` | NCRISC | NOC 1 |
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
dispatch_s            BRISC reader          NCRISC pusher          host
(NCRISC, NOC 1)       (profiler tensix,     (profiler tensix,      (receiver thread)
                       NOC 0)                NOC 1)
    |                      |                      |                      |
    |-- inline_dw_write -->|                      |                      |
    |   PUSH_A / PUSH_B    |                      |                      |
    |   (NOC 1)            |                      |                      |
    |                      |-- noc_async_read     |                      |
    |                      |   (read timestamps,  |                      |
    |                      |    NOC 0)            |                      |
    |                      |-- write_index++ ---->| L1 ring buffer       |
    |                      |                      |                      |
    |                      |                      |-- drain all pending  |
    |                      |                      |   push_entries_to_host
    |                      |                      |   (coalesced PCIe    |
    |                      |                      |    writes, ~420 ns) ->| hugepage
    |                      |                      |                      |-- read pages
    |                      |                      |                      |   -> callbacks
```

## Double-Buffer Protocol

dispatch_s maintains two timestamp buffers in its own L1 (A/B). On each
profiled `CQ_DISPATCH_CMD_SEND_GO_SIGNAL` it:

1. Waits for prior same-stream workers.
2. Captures the device start tick immediately before the first go-signal NOC
   write.
3. Issues GO, writes the command-carried runtime ID, and sends a `PUSH_A` or
   `PUSH_B` state to the reserved profiler tensix via a NOC inline dword write.

There is no program-ID FIFO and no M1 completion timestamp. The public host
record explicitly sets the end timestamp to zero.

This alternation only hands one in-flight record to the reader at a time;
dispatch_s never blocks on the profiler.

The **BRISC reader** polls its state mailbox. On `PUSH_A`/`PUSH_B` it issues a
`noc_async_read` of the 32-byte transport record from the indicated dispatch_s
buffer into the next ring slot, then advances `write_index` (records for
unprofiled programs are read but not committed). If the ring is full it spins
(heartbeat `ring_full_wait_count`); in practice this does not happen, because the
host drains records faster than they are produced. The reader also services host
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

### Push cost (NCRISC pusher side)

| Metric | Value |
|--------|-------|
| `push_entries_to_host` per drain | **~420 ns** |

### Throughput

The signal-to-record path uses a fast NOC read into a deep L1 ring, decoupled
from the PCIe push; the pusher drains pending entries in coalesced bursts. This
does not claim losslessness under unbounded pressure. Milestone 1 qualification
in `test_realtime_profiler_stress.cpp` uses repeated ordinary non-trace launches
and verifies lossless delivery for the measured peak-load window. Later
milestones add explicit device-side loss counters and drop behavior.

## Implementation Notes

- The host side runs a receiver thread that drains device→host pages and
  publishes decoded records onto a `BroadcastRing`; separate per-callback
  consumer threads read from the ring and invoke the registered callbacks. A slow
  callback only drops records for that consumer (tracked in `Consumer::dropped`);
  it never stalls page draining or dispatch.
