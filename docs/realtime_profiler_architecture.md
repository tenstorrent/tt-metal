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
|   |   |   _addr       |  |   PUSH_B -> NOC read buf B -> ring push   |  |   |
|   |   | - state (R/W) |  |   terminate_requested -> drain and exit   |  |   |
|   |   | - sync_req    |  |                                           |  |   |
|   |   | - sync_host_ts|  |                                           |  |   |
|   |   +-------+-------+  +-------------------------------------------+  |   |
|   |           |                        ^ NOC read (timestamp data)      |   |
|   +-----------+------------------------+--------------------------------+   |
|               |                        |                                    |
|               | state (PUSH_B)         |                                    |
|               | NOC write              |                                    |
|               v                        |                                    |
|   +---------------------------------------------------------------------+   |
|   | DISPATCH CORE (dispatch_s)                                          |   |
|   | Kernel: cq_dispatch_subordinate.cpp                                 |   |
|   |                                                                     |   |
|   |   L1 carve-out realtime_profiler_msg_t:                              |   |
|   |     per-stream start/watermark slots, completed-record ring,        |   |
|   |     mailbox B, drop counters, reset generations, termination        |   |
|   |                                                                     |   |
|   |   NCRISC: publish start descriptor before go signal; service at     |   |
|   |     most one completed record into mailbox B per progress point     |   |
|   |   TRISC0: observe each stream completion counter, capture device    |   |
|   |     end tick, publish intervals, and order Finish watermarks        |   |
|   +---------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------+
```

---

## 2. Data Flow: Program Timestamp to Host

```
  DISPATCH_S                 REAL-TIME PROFILER CORE              HOST
  (dispatch_s)               (cq_realtime_profiler)               (receiver thread)

       |                              |                                  |
       | 1. Publish start descriptor  |                                  |
       | 2. Send go signal            |                                  |
       | 3. TRISC0 observes stream    |                                  |
       |    completion, records end   |                                  |
       | 4. Publish completed record  |                                  |
       | 5. Move one record to buf B  |                                  |
       | 6. NOC write PUSH_B -------->|                                  |
       |                              | 7. NOC read timestamp data       |
       | <----------------------------|    from dispatch_s L1 (buf B)    |
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
| **Dispatch_s L1** | Per-stream start rings, completed-record ring, mailbox B, loss counters, reset generations, termination handshake, program_id_fifo, **realtime_profiler_core_noc_xy**, and **realtime_profiler_remote_state_addr**. Host writes NOC XY and the profiler tensix L1 address of `realtime_profiler_state` for NOC signaling. |
| **Profiler tensix L1** | **config_buffer_addr**, **realtime_profiler_state**, **terminate_requested**, sync_request, sync_host_timestamp. |

Layout: `tt_metal/hw/inc/hostdev/realtime_profiler_msgs.h`. HAL: `tt::tt_metal::realtime_profiler_msgs`. Not in `mailboxes_t`.

---

## 5. File / Component Reference

| Component | File(s) |
|-----------|--------|
| Dispatch_s NCRISC (start publication + transport signal) | `tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp`, `realtime_profiler.hpp` |
| Dispatch_d (program-ID + reset-epoch publication) | `tt_metal/impl/dispatch/kernels/cq_dispatch.cpp` |
| Dispatch_s TRISC0 (completion observer) | `tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate_compute.cpp`, `cq_realtime_profiler_dispatch_subordinate.hpp` |
| Profiler-core BRISC (mailbox reader) | `tt_metal/impl/dispatch/kernels/cq_realtime_profiler.cpp` |
| Profiler-core NCRISC (D2H pusher) | `tt_metal/impl/dispatch/kernels/cq_realtime_profiler_push.cpp` |
| Host init, sync, receiver thread | `mesh_device.cpp`, `realtime_profiler_manager.cpp` |
| Shared struct + HAL accessors | `realtime_profiler_msgs.h` → `realtime_profiler_msgs` (generated) |
| Wrap and queue protocol helpers | `tt_metal/impl/dispatch/kernels/realtime_profiler_protocol.hpp` |
| Callbacks (Tracy, user) | `tt_metal/impl/dispatch/data_collector.cpp`, `realtime_profiler_tracy_handler.cpp` |
