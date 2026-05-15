# Real-Time Profiler: Architecture and Performance Analysis

## Overview

The real-time profiler streams per-program device-side timestamps to the host
during execution, enabling 1:1 correlation between host-side Tracy zones
(`EnqueueMeshWorkload`) and device-side program execution windows.

### Components

| Component | File | Core | NOC |
|-----------|------|------|-----|
| **dispatch_s** (signal source) | `cq_dispatch_subordinate.cpp` | NCRISC | NOC 1 |
| **profiler kernel** (data mover) | `cq_realtime_profiler.cpp` | BRISC | NOC 0 |
| **host receiver thread** | `mesh_device.cpp` | CPU thread | PCIe |

### Data Flow

```
dispatch_s                    profiler kernel                  host
(NCRISC, NOC 1)               (BRISC, NOC 0)                  (receiver thread)
    |                              |                              |
    |-- inline_dw_write(PUSH) ---->|                              |
    |   (NOC 1, ~92 cycles)        |                              |
    |                              |-- noc_async_read ----------->| dispatch_s L1
    |                              |   (read timestamps, NOC 0)   |
    |                              |                              |
    |                              |-- noc_wwrite (PCIe) -------->| hugepage
    |                              |   (push 64B page, ~50-80 us) |
    |                              |                              |
    |                              |                              |-- read page
    |                              |                              |   buffer record
```

## Double-Buffer Protocol

dispatch_s maintains two timestamp buffers in L1 (`kernel_start_a/b`,
`kernel_end_a/b`). On each `CQ_DISPATCH_CMD_WAIT` completion it:

1. Writes the program's start/end timestamps into the next buffer (alternating A/B)
2. Sends a `PUSH_A` or `PUSH_B` state via NOC inline write to the profiler kernel

The profiler kernel polls its mailbox, reads the indicated buffer from
dispatch_s's L1 via `noc_async_read`, and pushes a 64-byte page to the host
over PCIe via the D2H socket.

## Measured Timing (ResNet50 on Wormhole, 317 programs)

### Signal Cost (dispatch_s side)

| Metric | Value |
|--------|-------|
| `signal_realtime_profiler_and_switch` duration | **~92 cycles (~0.09 us)** |
| Signal = NOC 1 inline dword write | Negligible overhead on dispatch |

### Push Cost (profiler kernel side)

The profiler's full push cycle (read from dispatch_s + PCIe write + barrier)
dominates the latency:

| Percentile | Cycle time (cycles) | Cycle time (us) |
|------------|--------------------:|----------------:|
| Fastest    |             23,469  |          23.5   |
| P10        |             48,212  |          48.2   |
| P25        |             59,359  |          59.4   |
| **P50**    |         **79,985**  |      **80.0**   |
| P75        |            164,416  |         164.4   |
| P90        |            192,757  |         192.8   |

The bottleneck is steps 4-8 of the push path:

```
 1. noc_async_read(dispatch_s buffer)      ~100-200 cycles (fast)
 2. noc_async_read_barrier()               ~100 cycles
 3. noc_write_init_state()                 trivial
 4. socket_reserve_pages(1)             ** can block if D2H socket full **
 5. noc_wwrite_with_state() (PCIe)      ** 64B write over PCIe **
 6. socket_push_pages(1)                   trivial
 7. pcie_socket_notify_receiver()          trivial
 8. noc_async_write_barrier()           ** wait for PCIe completion **
```

### Signal Rate vs. Push Rate

Programs in ResNet50 complete every **48-290 us** (at the fastest). The profiler
push takes **50-80 us** at P50. When signals arrive faster than the profiler can
push, the second signal overwrites the first in the mailbox, causing data loss.

## The Double-Buffer Race Condition

### Mechanism

When dispatch_s fires signals faster than the profiler can drain them:

```
Time 0 us:   dispatch_s sends PUSH_B (program 174)
Time 54 us:  dispatch_s sends PUSH_A (program 175) -- profiler still pushing 173
Time 108 us: dispatch_s sends PUSH_B (program 176) -- overwrites 174's data in buffer B
Time 160 us: profiler finishes push for 173, reads mailbox: sees PUSH_B (176)
             program 174 and 175 are LOST
```

### Measured Impact

From a ResNet50 run (317 programs, no ack mechanism):

| Metric | Value |
|--------|------:|
| Total signals sent by dispatch_s | 317 |
| Records received by host | 250-276 |
| **Missing records** | **41-55 (13-17%)** |

Missing IDs cluster in bursts (e.g., programs 174-215) because once the
profiler falls behind, every subsequent signal overwrites the previous before
it can be read, creating a cascading loss pattern.

### Gap Analysis at the Loss Region

```
Signal # | Gap from prev (us) | Captured?
---------|-------------------:|-----------
  173    |            161.8   | Yes
  174    |             54.2   | MISSING  <-- profiler still pushing 173
  175    |            286.7   | MISSING  <-- buffer A overwritten
  176    |            204.7   | MISSING  <-- buffer B overwritten
  ...    |              ...   | MISSING  (cascade continues)
  188    |             50.6   | Yes      <-- profiler catches up
```

## Fix: Acknowledge-Before-Signal Protocol

### Design

Add a `profiler_ack` field to the mailbox. The profiler writes `1` after
completing a push; dispatch_s waits for `ack==1` before sending the next signal.

**dispatch_s (wait -> reset -> send):**
```
while (profiler_ack == 0) { }   // wait for previous push to complete
profiler_ack = 0                // reset
inline_dw_write(PUSH_A/B)      // send next signal
```

**profiler kernel (ack after full push):**
```
noc_async_read(data)            // read timestamps
noc_async_read_barrier()
... PCIe push ...               // slow path (~50-80 us)
noc_async_write_barrier()
mailbox->state = IDLE           // mark self idle BEFORE ack
send_profiler_ack()             // NOC inline write to dispatch_s (cmd buf 3)
```

### Expected Overhead

With the ack mechanism, dispatch_s waits ~50-80 us per program (the profiler's
push cycle time). For 317 programs:

| Metric | Value |
|--------|------:|
| Per-program overhead | 50-80 us |
| Total overhead (317 programs) | 16-25 ms |
| ResNet50 inference time | ~22 ms |
| **Relative slowdown** | **~1.7-2.1x** |

This is acceptable for a profiling mode (profiling always adds overhead).

### Implementation Notes

- `profiler_ack` is initialized to `1` by the host so the first signal passes
  without waiting.
- The ack uses `noc_inline_dw_write` on **cmd buf 3** (`write_at_cmd_buf`).
  Using cmd buf 0 (`write_cmd_buf`) corrupts in-flight PCIe data writes.
- The profiler writes `IDLE` to its own mailbox **before** sending the ack.
  This prevents dispatch_s's subsequent PUSH (sent after seeing the ack) from
  being overwritten by a late IDLE write.
- The host receiver thread must use **C++ record buffering** (not Python
  callbacks) to avoid a GIL deadlock: TTNN ops hold the GIL during
  `EnqueueProgram`, so a Python callback from the receiver thread would block
  on GIL acquisition while dispatch_s is stalled waiting for the ack.

### NOC Coordinate Considerations (Wormhole)

- dispatch_s runs on NCRISC (NOC 1); profiler kernel runs on BRISC (NOC 0).
- On Wormhole, NOC 0 and NOC 1 have **transposed** coordinate systems.
- The host must encode `realtime_profiler_core_noc_xy` in **NOC 1 space**
  (for dispatch_s to address the profiler), and `DISPATCH_CORE_NOC_X/Y` in
  **NOC 0 space** (virtual coordinates, for the profiler to address dispatch_s).

## Future Optimization: L1 FIFO Buffering

To reduce the ack-induced overhead, the profiler kernel could maintain a local
L1 FIFO (e.g., 8-16 entries). On receiving a PUSH, it would quickly read the
timestamps into the FIFO (~200 cycles) and ack immediately, then push FIFO
entries to the host asynchronously. This would reduce the ack latency from
~50-80 us (full PCIe push) to ~0.2 us (just the read), eliminating the
dispatch_s stall almost entirely.
