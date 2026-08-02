# Real-Time Profiler: Dispatch Core, Profiler Core, and Host Interaction

This document describes how the **dispatch core** (dispatch_s), **real-time profiler core**, and **host** interact to stream program execution timestamps and metadata to the host for profiling (e.g. Tracy).

---

## 1. High-Level Architecture

```
+-----------------------------------------------------------------------------+
| HOST                                                                        |
|                                                                             |
|   +-------------------+  +------------------+  +-------------------------+  |
|   | Init/Calibration  |  | D2H Socket       |  | Receiver thread         |  |
|   | - Pick profiler   |  | - Config buffer  |  | - wait_for_pages()      |  |
|   |   core            |  | - Page flow      |  | - Parse timestamps      |  |
|   | - Create D2H      |  |   (PCIe)         |  | - InvokeProgramRealtime |  |
|   |   socket          |  |                  |  |   Callbacks()           |  |
|   | - Run sync        |  |                  |  |                         |  |
|   | - Start recv      |  |                  |  |                         |  |
|   +--------+----------+  +--------+---------+  +------------+------------+  |
|            |                     |                          |               |
|            | L1 writes           | PCIe read                | PCIe read     |
|            | (sync_request,      | (timestamp pages)        | (timestamp    |
|            |  sync_host_ts,      |                          |  pages)       |
|            |  config_buffer_addr)|                          |               |
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
|   |   | - config_buf  |  |   IDLE + sync_request -> sync(); push     |  |   |
|   |   |   _addr       |  |   PUSH_A -> NOC read buf A -> D2H push    |  |   |
|   |   | - state (R/W) |  |   PUSH_B -> NOC read buf B -> D2H push    |  |   |
|   |   | - sync_req    |  |   TERMINATE -> exit                       |  |   |
|   |   | - sync_host_ts|  |                                           |  |   |
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

Host and device timestamps are aligned so consumers can relate device cycles to host time. The host keeps a per-chip affine mapping `device_cycle = frequency * host_ns + device_cycle_offset`: `frequency` is fit once at init; `device_cycle_offset` is re-anchored whenever a probe finds the mapping has visibly moved.

The tensix free-running cycle counter is a hardware register the NOC serves directly, so the host reads it through its own uncached TLB window with a single load, bracketed between two `steady_clock` reads. The device timestamp is known to have been sampled somewhere inside that bracket, so the anchor goes at its midpoint and the reported `sync_error` is half its width.

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

**Init** takes ~100 samples at 5 ms spacing and fits `frequency` by linear regression, each sample the tightest of a few reads. A settled clock fits that line to ~2 ns rms; a fit taken while AICLK is ramping still uses every sample but lands a frequency tens of ppm off, so a residual far above that floor is rejected and the calibration retried.

**Steady state:** every 1 ms the receiver takes one read per device and asks what the standing mapping predicted for it. A probe cannot locate the clock better than half its own bracket, so a miss smaller than that says nothing and the mapping is left alone -- re-anchoring on it would trade a good anchor for a noisier one. A larger miss is real, and the probe is taken as the new anchor. No assumed drift rate enters anywhere: the decision is made on what the probe measured. A slow read rejects itself, since its own miss is always smaller than the resolution it would bring.

What this bounds is set by how the clock actually misbehaves. Blackhole's DVFS throttler runs on a 1 ms tick and steps AICLK by one PLL multiplier (1350 -> 1343 MHz, ~5200 ppm) for one or more of those ticks under load; the mapping error a step leaves is that rate times however long it goes uncorrected, and those cycles are never given back. Measured worst case: 14.2 us when re-anchoring every 10 ms, 5.7 us at 1 ms. Steps arrive in bursts, so the interval drops to 250 us for 5 ms once one is seen.

Sync runs on the receiver thread rather than its own. A read is ~700 ns against a drain-gap threshold of 5 ms, so it costs the drain far less than a second thread costs in wakeups, and it leaves the mapping owned by one thread -- the receiver both maintains it and stamps records with it, so no publication protocol is needed.

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
| Clock mapping: fit, re-anchor policy, drift, error bar | `realtime_profiler_clock_model.cpp` |
| Clock read path, clock model and re-anchor rule, calibration cache | `realtime_profiler_clock_sync.cpp` |
| Shared struct + HAL accessors | `realtime_profiler_msgs.h` → `realtime_profiler_msgs` (generated) |
| Public API (register / unregister / is-active) | `tt_metal/impl/realtime_profiler/realtime_profiler.cpp` |
| Record fan-out (service, Tracy, user callbacks) | `tt_metal/impl/realtime_profiler/realtime_profiler_service.cpp`, `realtime_profiler_tracy_consumer.cpp` |
| Kernel-source metadata (runtime_id → sources) | `tt_metal/impl/dispatch/data_collector.cpp` |
