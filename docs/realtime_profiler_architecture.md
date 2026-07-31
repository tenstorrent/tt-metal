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

Host and device timestamps are aligned so consumers (Tracy, callbacks) can relate device cycles to host time. The host keeps a per-chip affine mapping `device_cycle = frequency * host_ns + device_cycle_offset`: `frequency` is fit once at init; `device_cycle_offset` is re-anchored continuously from the sync thread (~every 50 ms per device).

Nothing runs on device for this. The tensix free-running cycle counter is a hardware register the NOC serves directly, so the host reads it through its own uncached TLB window with a single load, bracketed between two `steady_clock` reads. The device timestamp is known to have been sampled somewhere inside that bracket, so the anchor goes at its midpoint and the reported `sync_error` is half its width.

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

**Init** takes ~100 samples at 5 ms spacing and fits `frequency` by linear regression. A settled clock fits that line to ~2 ns rms; a fit taken while AICLK is ramping still uses every sample but lands a frequency tens of ppm off, which the servo would then spend the session correcting as though it were drift, so a residual far above that floor is rejected and the calibration retried. **Steady state:** each device is resynced every 10 ms and `device_cycle_offset` re-anchored. The frequency is held from bring-up while AICLK moves ~123 ppm under load, so the error accrued between anchors is that rate times the interval -- which is why the cadence is what bounds it, and why it is 10 ms rather than 50 ms.

Each resync takes a burst of 8 reads and anchors on the tightest bracket. Ranked rather than compared against a threshold: under record load the whole bracket distribution shifts, and an absolute threshold would reject every read in a pass instead of picking that pass's best. Syncing runs on its own thread, not the receiver's: it is a slow control loop where draining is a data path. The receiver reads each device's mapping through a seqlock the sync thread publishes into.

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
| Clock read path, probe burst, published mapping, calibration cache | `realtime_profiler_clock_sync.cpp` |
| Shared struct + HAL accessors | `realtime_profiler_msgs.h` → `realtime_profiler_msgs` (generated) |
| Public API (register / unregister / is-active) | `tt_metal/impl/realtime_profiler/realtime_profiler.cpp` |
| Record fan-out (service, Tracy, user callbacks) | `tt_metal/impl/realtime_profiler/realtime_profiler_service.cpp`, `realtime_profiler_tracy_consumer.cpp` |
| Kernel-source metadata (runtime_id → sources) | `tt_metal/impl/dispatch/data_collector.cpp` |
