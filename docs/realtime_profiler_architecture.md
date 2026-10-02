# Real-Time Profiler: Dispatch Core, Profiler Core, and Host Interaction

This document describes how the **dispatch core** (dispatch_s), **real-time profiler core**, and **host** interact to stream program execution timestamps and metadata to the host for profiling (e.g. Tracy).

---

## 1. High-Level Architecture

```
+--------------------------------------------------------------+
| dispatch_s  (dispatch core)                                  |
| Stamps each program's start and end time into a 4-slot ring. |
| If all 4 slots are still unread, it waits.                   |
+--------------------------------------------------------------+
            |  record ready                       ^  slot free
            v                                     |
+--------------------------------------------------------------+
| BRISC  (profiler core)                                       |
| Copies each ready record into a big ring in its own L1,      |
| then tells dispatch_s the slot is free.                      |
| Also answers the host's clock-sync requests.                 |
+--------------------------------------------------------------+
            |  records
            v
+--------------------------------------------------------------+
| NCRISC  (profiler core)                                      |
| Sends the ring's records to the host over PCIe               |
| and tells the host how many it sent.                         |
+--------------------------------------------------------------+
            |  records (PCIe)                     ^  records read
            v                                     |
+--------------------------------------------------------------+
| Receiver thread  (host)                                      |
| Reads the records and hands each one to the callbacks        |
| (e.g. Tracy), then tells the NCRISC how many it has read.    |
+--------------------------------------------------------------+
```

---

## 2. Data Flow: The Three Queues

A record leaves the device through three single-producer, single-consumer queues. Each index has exactly one
writer, and a full queue makes its producer wait, so a slow host slows dispatch down instead of losing records.

```
  dispatch_s              BRISC                   NCRISC                  Host
  (dispatch core)         (profiler core)         (profiler core)         (receiver thread)

   |                       |                       |                       |
   | 1. Write the record   |                       |                       |
   |    into the open slot |                       |                       |
   |    (own L1)           |                       |                       |
   | 2. If no slot is      |                       |                       |
   |    free: WAIT         |                       |                       |
   | 3. Advance wr_idx,    |                       |                       |
   |    NoC-write it to    |                       |                       |
   |    the profiler core  |                       |                       |
   | ------ wr_idx ------->|                       |                       |
   |                       | 4. See wr_idx move    |                       |
   |                       | 5. If the ring is     |                       |
   |                       |    full: WAIT         |                       |
   |                       | 6. NoC-read the slot  |                       |
   |                       |    into the ring,     |                       |
   |                       |    bump write_index   |                       |
   | --- record (32 B) --->|                       |                       |
   |                       | ---- write_index ---->|                       |
   |                       | 7. NoC-write rd_idx   |                       |
   |                       |    back: slot free    |                       |
   | <------ rd_idx -------|                       |                       |
   |                       |                       | 8. See write_index    |
   |                       |                       |    move               |
   |                       |                       | 9. If the host FIFO   |
   |                       |                       |    is full: WAIT      |
   |                       |                       | 10. NoC-write the     |
   |                       |                       |     entries to the    |
   |                       |                       |     host FIFO (PCIe)  |
   |                       |                       | ------ records ------>|
   |                       |                       | 11. Write bytes_sent  |
   |                       |                       |     to the host, bump |
   |                       |                       |     read_index        |
   |                       |                       | ---- bytes_sent ----->|
   |                       | <---- read_index -----|                       |
   |                       |                       |                       | 12. See bytes_sent;
   |                       |                       |                       |     read the pages,
   |                       |                       |                       |     run the callbacks
   |                       |                       |                       | 13. Write bytes_acked
   |                       |                       |                       |     into profiler L1
   |                       |                       | <---- bytes_acked ----|
```

| Queue | Lives in | Size | Producer advances | Consumer advances | When full |
|-------|----------|------|-------------------|-------------------|-----------|
| Record ring | dispatch core L1 | 4 slots of 32 B | `record_wr_idx` (dispatch_s, NoC write to the profiler core) | `record_rd_idx` (BRISC, NoC write back) | dispatch_s waits |
| BRISC→NCRISC ring | profiler core L1 | 16,384 entries of 64 B | `write_index` (BRISC) | `read_index` (NCRISC) | BRISC waits |
| D2H socket FIFO | pinned host memory | 32,768 pages of 64 B (2 MiB) | `bytes_sent` (NCRISC, PCIe write) | `bytes_acked` (host, write into profiler core L1) | NCRISC waits |

---

## 3. Sync (Timestamp Calibration)

Host and device timestamps are aligned so that Tracy (or other consumers) can relate device cycles to host time.

```
  Host                                        BRISC  (profiler core)

   |                                           |
   | 1. Start sync                             |
   | ---------------- sync on ---------------->|
   | 2. Send host time T                       |
   | ------------------- T ------------------->|
   |                                           | 3. Stamp device time D
   | <-------- D and T, via the NCRISC --------|
   | 4. Repeat 2-3 N times                     |
   | 5. Stop sync                              |
   | --------------- sync off ---------------->|
   | 6. Fit a line through                     |
   |    the (T, D) pairs:                      |
   |    device clock rate                      |
   |    and offset                             |
```

---

## 4. Carve-out layout (conceptual)

| Location | Contents (`realtime_profiler_msg_t`) |
|----------|----------------------------------------|
| **Dispatch_s L1** | Record ring (`records[4]`, `record_wr_idx` = open slot, `record_rd_idx` = reader's ack), program_id_fifo, **realtime_profiler_core_noc_xy**, **realtime_profiler_remote_wr_idx_addr**, realtime_profiler_state (stops the compute helper). Host writes the profiler tensix L1 address of `record_wr_idx`, then NOC XY (which enables publishing), after the reader kernels launch. |
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
