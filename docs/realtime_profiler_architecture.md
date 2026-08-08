# Real-Time Profiler: Dispatch Core, Profiler Core, and Host Interaction

This document describes how the **dispatch core** (dispatch_s), **real-time profiler core**, and **host** interact to stream program execution timestamps and metadata to the host for profiling (e.g. Tracy).

---

## 1. High-Level Architecture

```
+-----------------------------------------------------------------------------+
| HOST                                                                        |
|                                                                             |
|   +-------------------+  +------------------+  +-------------------------+  |
|   | Init               |  | D2H Socket       |  | Receiver thread         |  |
|   | - Pick profiler   |  | - Config buffer  |  | - Drain pages           |  |
|   |   core            |  | - Page flow      |  | - Probe device clock    |  |
|   | - Create D2H      |  |   (PCIe)         |  | - Map + publish records |  |
|   |   socket          |  |                  |  |   to the record ring    |  |
|   | - Warm up clock   |  |                  |  |                         |  |
|   | - Start recv      |  |                  |  |                         |  |
|   +--------+----------+  +--------+---------+  +------------+------------+  |
|            |                     |                          |               |
|            | L1 write            | PCIe read                | PCIe read     |
|            | (config_buffer_addr)| (timestamp pages)        | (timestamp    |
|            |                     |                          |  pages +      |
|            |                     |                          |  clock reg)   |
+------------+---------------------+--------------------------+---------------+
             |                     |                          |               |
             v                     |                          |               |
+-----------------------------------------------------------------------------+
| DEVICE (per chip)                                                           |
|                                                                             |
|   +---------------------------------------------------------------------+   |
|   | REAL-TIME PROFILER CORE (Tensix, closest to PCIe)                   |   |
|   | Kernel: cq_realtime_profiler.cpp                                    |   |
|   |                                                                     |   |
|   |   +---------------+  +-------------------------------------------+  |   |
|   |   | Mailbox (L1)  |  | Loop:                                     |  |   |
|   |   | - config_buf  |  |   PUSH_A -> NOC read buf A -> L1 ring     |  |   |
|   |   |   _addr       |  |   PUSH_B -> NOC read buf B -> L1 ring     |  |   |
|   |   | - state (R/W) |  |   TERMINATE -> exit                       |  |   |
|   |   +-------+-------+  +-------------------------------------------+  |   |
|   |           |                        ^ NOC read (timestamp data)      |   |
|   +-----------+------------------------+--------------------------------+   |
|               |                        |                                    |
|               | state (PUSH_A/B)       |                                    |
|               | NOC write              |                                    |
|               v                        |                                    |
|   +---------------------------------------------------------------------+   |
|   | DISPATCH CORE (dispatch_s)                                          |   |
|   | Kernel: cq_dispatch_subordinate.cpp                                 |   |
|   |                                                                     |   |
|   |   L1 carve-out realtime_profiler_msg_t:                              |   |
|   |     Ping-pong: kernel_start_a/b, kernel_end_a/b                     |   |
|   |     program_id_fifo, realtime_profiler_core_noc_xy,                 |   |
|   |     realtime_profiler_remote_state_addr                             |   |
|   |                                                                     |   |
|   |   Per-command: record start ts, FIFO program id, process cmd,       |   |
|   |     record end ts, signal_realtime_profiler_and_switch()            |   |
|   |     ... process command ...                                         |   |
|   |     record_realtime_timestamp(false); signal_realtime_profiler_and_ |   |
|   |     switch();  (NOC-write state to profiler core)                   |   |
|   +---------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------+
```

---

## 2. Data Flow: Program Timestamp to Host

```
  DISPATCH_S                 REAL-TIME PROFILER CORE              HOST
  (dispatch_s)               (cq_realtime_profiler)               (receiver thread)

       |                              |                                  |
       | 1. Record start ts,          |                                  |
       |    program_id into           |                                  |
       |    mailbox buf A or B        |                                  |
       | 2. Process command           |                                  |
       | 3. Record end ts             |                                  |
       | 4. Update state PUSH_A/B     |                                  |
       | 5. NOC write state --------> |                                  |
       |                              | 6. See state PUSH_A or PUSH_B    |
       |                              | 7. NOC read timestamp data       |
       | <----------------------------|    from dispatch_s L1 (buf A/B)  |
       |                              | 8. Push page to D2H socket       |
       |                              |    (PCIe write to host buffer)   |
       |                              | -------------------------------> | 9. wait_for_pages
       |                              |                                  |    get_read_ptr
       |                              |                                  | 10. Parse start/end ts,
       |                              |                                  |     program_id
       |                              |                                  | 11. InvokeProgramRealtime
       |                              |                                  |     Callbacks(record)
       |                              | <------------------------------- | pop_pages, notify_sender
```

---

## 3. Sync (Timestamp Calibration)

Host and device timestamps are aligned so consumers can relate device cycles to host time. The host retains a per-chip ring of clock **probes** — paired (host time, device ticks) reads — and places each record between the two probes around it: every record is published with its own affine mapping `device_cycle = frequency * host_ns + device_cycle_offset`, whose rate is the secant of the probe pair (or, for a record spanning several pairs, of its own two placements). There is no fitted model and no re-anchor policy: the mapping is pure interpolation over what was measured (`DeviceClockMapping`).

The tensix free-running cycle counter is a hardware register the NOC serves directly, so the host reads it through its own uncached TLB window with a single load, bracketed between two `steady_clock` reads. The device timestamp is known to have been sampled somewhere inside that bracket, so the probe goes at its midpoint and contributes half its width to `sync_error`.

```
  HOST                                   REAL-TIME PROFILER CORE

    |  t1 = steady_clock::now()            |
    |  load WALL_CLOCK_L  ---------------> |  (served by the NOC; no RISC involved)
    | <--------------------- D ----------- |
    |  t2 = steady_clock::now()            |
    |  load WALL_CLOCK_H (latched by the L read, so outside the bracket)
    |  anchor D at (t1+t2)/2               |
```

Reading the low word latches the high word, so the pair is coherent and only the low read has to sit inside the bracket. The window is allocated once at configure time and the mapped address resolved once: the generic UMD register read holds a chip-wide mutex and rewrites the TLB configuration registers over PCIe on every call, all of which would land inside the bracket and widen it by ~450 ns. The bracket *is* the error bound, so that width is the whole quantity being minimised.

Because no device software is in the path, a sample cannot be delayed by whatever the profiler core's push loop is doing, and sync costs the device nothing at all.

**Probe cadence.** The receiver probes a device immediately after every non-empty page read — the previous probe and that one bracket every record the read returned, so nothing waits on publication — and on a 500 us sync-interval floor while the device is idle. Each probe is the tightest of up to a few reads, taken only while the bracket is still wider than reads have recently been coming back at; a warm-up before the receiver thread starts gives the first record a pair to land between. Probes are retained in a 2-second ring; a record whose start outlives the ring is anchored at its end (or at a start pinned earlier by peeking dispatch_s's in-flight start timestamp — see `LongProgramIsDeliveredIntact`).

**sync_error** is what the placement itself could be off by: the anchoring probes' half-brackets, plus the clock's measured departure from their secant, read at the neighbouring probe the secant was not fitted to. It is an estimate that tracks the measured clock, not a worst-case ceiling; the didt suite bounds what remains in practice.

What the probe spacing has to resolve is set by how the clock actually misbehaves: DVFS steps AICLK on the ARC firmware's 1 ms timer (~5200 ppm per PLL multiplier step on Blackhole), so probe pairs well under a millisecond wide straddle at most one step, and a step inside a pair misplaces a record by at most step * width / 4.

Sync runs on the receiver thread rather than its own. A read is ~700 ns, far less than a second thread costs in wakeups, and it leaves the mapping owned by one thread -- the receiver both maintains it and stamps records with it, so no publication protocol is needed.

---

## 4. Carve-out layout (conceptual)

| Location | Contents (`realtime_profiler_msg_t`) |
|----------|----------------------------------------|
| **Dispatch_s L1** | Ping-pong buffers, program_id_fifo, **realtime_profiler_core_noc_xy**, **realtime_profiler_remote_state_addr**, realtime_profiler_state. Host writes NOC XY and the profiler tensix L1 address of `realtime_profiler_state` for NOC signaling. |
| **Profiler tensix L1** | **config_buffer_addr**, **realtime_profiler_state**. |

Layout: `tt_metal/hw/inc/hostdev/realtime_profiler_msgs.h`. HAL: `tt::tt_metal::realtime_profiler_msgs`. Not in `mailboxes_t`.

---

## 5. File / Component Reference

| Component | File(s) |
|-----------|--------|
| Dispatch_s (timestamp record + signal) | `tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp`, `realtime_profiler.hpp` |
| Profiler-core kernels (BRISC reader + NCRISC pusher/sync) | `tt_metal/impl/dispatch/kernels/cq_realtime_profiler.cpp`, `cq_realtime_profiler_push.cpp` |
| Host init and receiver thread (per MeshDevice) | `tt_metal/impl/realtime_profiler/realtime_profiler_receiver.cpp` |
| Clock read path + probe-interpolation mapping (`DeviceClockSync` / `DeviceClockMapping`) | `tt_metal/impl/realtime_profiler/device_clock_sync.cpp` |
| Shared struct + HAL accessors | `realtime_profiler_msgs.h` → `realtime_profiler_msgs` (generated) |
| Public API (register / unregister / is-active) | `tt_metal/impl/realtime_profiler/realtime_profiler.cpp` |
| Record fan-out (service, Tracy, user callbacks) | `tt_metal/impl/realtime_profiler/realtime_profiler_service.cpp`, `realtime_profiler_tracy_consumer.cpp` |
| Kernel-source metadata (runtime_id → sources) | `tt_metal/impl/dispatch/data_collector.cpp` |
