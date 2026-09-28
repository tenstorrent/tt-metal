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
|   |   | - config_buf  |  |   rd_idx != wr_idx -> NOC read slot(s)    |  |   |
|   |   |   _addr       |  |     -> ring -> D2H push; ack rd_idx       |  |   |
|   |   | - record_wr   |  |   sync_request -> sync() (still drains)   |  |   |
|   |   |   _idx        |  |   TERMINATE bit + all read -> exit        |  |   |
|   |   | - sync_req    |  |                                           |  |   |
|   |   | - sync_host_ts|  |                                           |  |   |
|   |   +-------+-------+  +-------------------------------------------+  |   |
|   |           ^                        | NOC read (record slots)        |   |
|   |           |                        | NOC write record_rd_idx        |   |
|   +-----------+------------------------+--------------------------------+   |
|               |                        |                                    |
|               | record_wr_idx          |                                    |
|               | NOC write              v                                    |
|               v                        |                                    |
|   +---------------------------------------------------------------------+   |
|   | DISPATCH CORE (dispatch_s)                                          |   |
|   | Kernel: cq_dispatch_subordinate.cpp                                 |   |
|   |                                                                     |   |
|   |   L1 carve-out realtime_profiler_msg_t:                              |   |
|   |     records[16] (SPSC ring), record_wr_idx, record_rd_idx,          |   |
|   |     program_id_fifo, realtime_profiler_core_noc_xy,                 |   |
|   |     realtime_profiler_remote_wr_idx_addr                            |   |
|   |                                                                     |   |
|   |   Per-command: record start ts + program id into the open slot,     |   |
|   |     process cmd (end ts written while waiting on workers),          |   |
|   |     publish_realtime_profiler_record(): wait while the ring is      |   |
|   |     full, open the next slot, NOC-write record_wr_idx to the        |   |
|   |     profiler core                                                   |   |
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
       |    program_id into the       |                                  |
       |    open record slot          |                                  |
       | 2. Process command           |                                  |
       | 3. Record end ts             |                                  |
       | 4. Wait for a free slot,     |                                  |
       |    advance record_wr_idx     |                                  |
       | 5. NOC write wr_idx -------> |                                  |
       |                              | 6. See rd_idx != wr_idx          |
       |                              | 7. NOC read each pending slot    |
       | <----------------------------|    from dispatch_s L1, then      |
       | <------ record_rd_idx -------|    ack rd_idx (frees the slots)  |
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
| **Dispatch_s L1** | Record ring (`records[16]`, `record_wr_idx` = open slot, `record_rd_idx` = reader's ack), program_id_fifo, **realtime_profiler_core_noc_xy**, **realtime_profiler_remote_wr_idx_addr**, realtime_profiler_state (stops the compute helper). Host writes the profiler tensix L1 address of `record_wr_idx`, then NOC XY (which enables publishing), after the reader kernels launch. |
| **Profiler tensix L1** | **config_buffer_addr**, **record_wr_idx** (published count; written only by dispatch_s, terminate flag in bit 31), sync_request, sync_host_timestamp. |

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
