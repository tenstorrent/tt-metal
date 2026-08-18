# Real-Time Profiler: Dispatch Core, Profiler Core, and Host Interaction

This document describes how the **dispatch core** (dispatch_s), **real-time profiler core**, and **host** interact to stream program execution timestamps and metadata to the host for profiling (e.g. Tracy).

Milestone 1 of the Blackhole clean-room protocol publishes a correctly
identified device **start tick only**. Public records have `end_timestamp == 0`,
so Tracy deliberately emits no duration zone until Milestone 2 adds the
completion-gated endpoint. Trace-replay commands are unprofiled.

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
|   |     runtime ID slot, realtime_profiler_core_noc_xy,                 |   |
|   |     realtime_profiler_remote_state_addr                             |   |
|   |                                                                     |   |
|   |   Per profiled GO: wait for prior same-stream workers, record       |   |
|   |     start ts immediately before the first go-signal write, issue    |   |
|   |     GO, publish the command-carried runtime ID, then signal the      |   |
|   |     profiler core. No completion timestamp is published in M1.      |   |
|   +---------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------+
```

---

## 2. Data Flow: Program Timestamp to Host

```
  DISPATCH_S                 REAL-TIME PROFILER CORE              HOST
  (dispatch_s)               (cq_realtime_profiler)               (receiver thread)

       |                              |                                  |
       | 1. Wait for prior workers    |                                  |
       | 2. Record device start ts    |                                  |
       | 3. Issue GO                  |                                  |
       | 4. Publish runtime ID        |                                  |
       | 5. Update state PUSH_A/B     |                                  |
       | 6. NOC write state --------> |                                  |
       |                              | 7. See state PUSH_A or PUSH_B    |
       |                              | 8. NOC read timestamp data       |
       | <----------------------------|    from dispatch_s L1 (buf A/B)  |
       |                              | 9. Push page to D2H socket       |
       |                              |    (PCIe write to host buffer)   |
       |                              | -------------------------------> | 10. wait_for_pages
       |                              |                                  |     get_read_ptr
       |                              |                                  | 11. Parse start ts and
       |                              |                                  |     runtime ID; set public
       |                              |                                  |     end timestamp to zero
       |                              |                                  | 12. InvokeProgramRealtime
       |                              |                                  |     Callbacks(record)
       |                              | <------------------------------- | pop_pages, notify_sender
```

---

## 3. Sync (Timestamp Calibration)

Host and device timestamps are aligned so that Tracy (or other consumers) can relate device cycles to host time.

```
  HOST                              REAL-TIME PROFILER CORE

    |  Write sync_request = 1 (L1)        |
    | ---------------------------------> |  Poll sync_request
    |  Write sync_host_timestamp = T     |
    | ---------------------------------> |  See host_ts > 0
    |                                    |  Capture device wall clock (D)
    |                                    |  Push page: (D_hi, D_lo, T,
    |                                    |    REALTIME_PROFILER_SYNC_MARKER_ID)
    |                                    |  Clear sync_host_timestamp
    |  wait_for_pages(1)                 |
    | <--------------------------------- |  (D2H page arrives)
    |  Parse device_time D, host_time T  |
    |  Repeat for N samples              |
    |  Write sync_request = 0 (L1)       |
    | ---------------------------------> |  Exit sync loop
    |  Linear regression -> frequency,   |
    |  first_timestamp for this device   |
```

---

## 4. Carve-out layout (conceptual)

| Location | Contents (`realtime_profiler_msg_t`) |
|----------|----------------------------------------|
| **Dispatch_s L1** | Ping-pong timestamp/runtime-ID buffers, **realtime_profiler_core_noc_xy**, **realtime_profiler_remote_state_addr**, realtime_profiler_state. There is no program-ID FIFO. Host writes the remote state address before publishing the nonzero NOC XY activation release. |
| **Profiler tensix L1** | **config_buffer_addr**, **realtime_profiler_state**, sync_request, sync_host_timestamp. |

Layout: `tt_metal/hw/inc/hostdev/realtime_profiler_msgs.h`. HAL: `tt::tt_metal::realtime_profiler_msgs`. Not in `mailboxes_t`.

---

## 5. File / Component Reference

| Component | File(s) |
|-----------|--------|
| Dispatch_s (timestamp record + signal) | `tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp`, `realtime_profiler.hpp` |
| Real-time profiler kernel | `tt_metal/impl/dispatch/kernels/cq_realtime_profiler.cpp` |
| Host init, sync, receiver thread | `mesh_device.cpp`, `realtime_profiler_manager.cpp` |
| Shared struct + HAL accessors | `realtime_profiler_msgs.h` → `realtime_profiler_msgs` (generated) |
| Callbacks (Tracy, user) | `tt_metal/impl/dispatch/data_collector.cpp`, `realtime_profiler_tracy_handler.cpp` |
