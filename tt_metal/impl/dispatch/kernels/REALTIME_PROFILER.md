# Real-Time Profiler: Architecture and Performance Analysis

## Overview

The real-time profiler streams per-program device-side timestamps to the host
during execution, enabling 1:1 correlation between host-side Tracy zones
(`EnqueueMeshWorkload`) and device-side program execution windows.

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
    |   (NOC 1, ~92 cyc)   |                      |                      |
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
`CQ_DISPATCH_CMD_WAIT` completion it:

1. Writes the program's start/end timestamps into the next buffer (alternating A/B)
2. Sends a `PUSH_A` or `PUSH_B` state to the reserved profiler tensix via a NOC
   inline dword write

This alternation only hands one in-flight record to the reader at a time;
dispatch_s never blocks on the profiler.

The **BRISC reader** polls its state mailbox. On `PUSH_A`/`PUSH_B` it issues a
`noc_async_read` of the 32-byte timestamp pair from the indicated dispatch_s
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
| `signal_realtime_profiler_and_switch` duration | **~92 cycles (~0.09 us)** |
| Signal = NOC 1 inline dword write | Negligible overhead on dispatch |

### Push cost (NCRISC pusher side)

| Metric | Value |
|--------|-------|
| `push_entries_to_host` per drain | **~420 ns** |

### Throughput

The profiler fully keeps up with dispatch. The signal-to-record path is a fast
NOC read into a deep L1 ring, decoupled from the PCIe push; the pusher
then drains the entire pending backlog per iteration and coalesces it into a few
large bursts (~420 ns per push). Because the ring absorbs bursts and the push is cheap,
signals never outrun the drain and no records are lost.

This is verified under load by `test_realtime_profiler_stress.cpp`, which replays
a 4096-program blank-kernel trace back-to-back — the peak signal rate dispatch can
sustain, since blank kernels minimize per-program dispatch overhead — and asserts
every record arrives with the device ring and host D2H FIFO never filling.

## Implementation Notes

- The host side runs a receiver thread that drains device→host pages and
  publishes decoded records onto a `BroadcastRing`; separate per-callback
  consumer threads read from the ring and invoke the registered callbacks. A slow
  callback only drops records for that consumer (tracked in `Consumer::dropped`);
  it never stalls page draining or dispatch.
