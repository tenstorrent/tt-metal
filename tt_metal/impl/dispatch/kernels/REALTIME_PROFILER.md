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
    |   record_wr_idx      |                      |                      |
    |   (NOC 1)            |                      |                      |
    |                      |-- noc_async_read     |                      |
    |                      |   (read record slot, |                      |
    |                      |    NOC 0)            |                      |
    |<-- inline_dw_write --|                      |                      |
    |    record_rd_idx     |                      |                      |
    |                      |-- write_index++ ---->| L1 ring buffer       |
    |                      |                      |                      |
    |                      |                      |-- drain all pending  |
    |                      |                      |   push_entries_to_host
    |                      |                      |   (coalesced PCIe    |
    |                      |                      |    writes, ~420 ns) ->| hugepage
    |                      |                      |                      |-- read pages
    |                      |                      |                      |   -> callbacks
```

## Record Ring Protocol

dispatch_s hands records to the BRISC reader through a single-producer,
single-consumer ring of `REALTIME_PROFILER_RECORD_SLOTS` (16) record slots in its
own L1 (`records[]` in `realtime_profiler_msgs.h`). Two free-running indices
drive it, and each has exactly one writer:

- `record_wr_idx`, written only by dispatch_s. On the dispatch core it names the
  open slot that dispatch_s and its compute helper are filling; dispatch_s pushes
  it to the reader's copy with a NOC inline dword write on every publish.
- `record_rd_idx`, written only by the reader. It pushes its count back to
  dispatch_s once a slot's NOC read has landed, which frees the slot.

After each command, dispatch_s publishes the open slot and opens the next one. It
never reuses a slot the reader may still be reading: if all other slots are
unread it waits (watcher waypoint `RPFW`) instead of dropping a record. Because
the published value is a count, a reader that polls late still sees every record,
and no write can overwrite another side's signal. At `CQ_DISPATCH_CMD_TERMINATE`
dispatch_s publishes the final count with `REALTIME_PROFILER_RECORD_WR_IDX_TERMINATE`
set in the same word, so the reader drains every record before it exits.

This replaced an A/B ping-pong handoff that had no acknowledgement and dropped
records whenever the reader fell behind dispatch_s (issue #57632).

The **BRISC reader** polls its `record_wr_idx`. For each published slot it issues a
`noc_async_read` of the 32-byte record into the next ring slot, then advances
`write_index` (records for unprofiled programs are read but not committed). It
keeps draining while servicing a clock sync. If the ring is full it spins
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
| `publish_realtime_profiler_record` duration (BH p100a, device-profiler zone, p50) | **~128 cycles (~0.09 us)** |
| Former A/B signal, same zone and board | ~105 cycles |
| Peak production rate, `RealtimeProfilerStress` (ring vs. A/B) | ~1.068 M rec/s both |

The zone figures include the zone's own overhead. `record_full_wait_count` (host:
`RealtimeProfilerManager::record_ring_full_wait_count()`) counts how often dispatch_s
waited for a free slot; it stays 0 at the stress test's peak rate.

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
