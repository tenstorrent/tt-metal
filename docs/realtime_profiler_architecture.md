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
  dispatch_s                   BRISC                        NCRISC                       Host
  (dispatch core)              (profiler core)              (profiler core)              (receiver thread)

   |                            |                            |                            |
   | 1. Store the record in the |                            |                            |
   |    open slot (own L1)      |                            |                            |
   | 2. No free slot: WAIT      |                            |                            |
   | 3. Bump wr_idx and         |                            |                            |
   |    NoC-write it to the     |                            |                            |
   |    profiler core           |                            |                            |
   | --------- wr_idx --------->|                            |                            |
   |                            | 4. See wr_idx move         |                            |
   |                            | 5. Ring full: WAIT         |                            |
   |                            | 6. Copy the slot into the  |                            |
   |                            |    ring (NoC read) and     |                            |
   |                            |    bump write_index        |                            |
   | ----- record (32 B) ------>|                            |                            |
   |                            | ------ write_index ------->|                            |
   |                            | 7. NoC-write rd_idx back   |                            |
   |                            |    (slot is free)          |                            |
   | <--------- rd_idx ---------|                            |                            |
   |                            |                            | 8. See write_index move    |
   |                            |                            | 9. Host FIFO full: WAIT    |
   |                            |                            | 10. NoC-write the entries  |
   |                            |                            |     to the host FIFO       |
   |                            |                            |     over PCIe              |
   |                            |                            | -------- records --------->|
   |                            |                            | 11. Send bytes_sent to the |
   |                            |                            |     host; bump read_index  |
   |                            |                            | ------- bytes_sent ------->|
   |                            | <------- read_index -------|                            |
   |                            |                            |                            | 12. See bytes_sent, read the
   |                            |                            |                            |     pages, run the callbacks
   |                            |                            |                            | 13. Write bytes_acked into the
   |                            |                            |                            |     profiler core L1
   |                            |                            | <------ bytes_acked -------|
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

## 4. Records and Memory

### 4.1 The record

dispatch_s writes one 32-byte record per command (`realtime_profiler_record_t`): a start entry and an end entry
(`realtime_profiler_timestamp_t`, 16 B each), each holding a 64-bit device time (8 B), the program id and a header
word. That is eight 4-byte words; offsets below are in bytes. Records for commands that are not a profiled program
carry id 0, and the BRISC drops them before they reach the host.

| Offset | Field | Contents |
|--------|-------|----------|
| 0 | `kernel_start.time_hi` | Device wall clock when dispatch_s started the command, high 32 bits |
| 4 | `kernel_start.time_lo` | Same, low 32 bits |
| 8 | `kernel_start.id` | Program runtime id; 0 = not a profiled program |
| 12 | `kernel_start.header` | Always 0 in a program record (marks marker entries, see 4.2) |
| 16 | `kernel_end.time_hi` | Last worker completion seen while the slot was open, high 32 bits |
| 20 | `kernel_end.time_lo` | Same, low 32 bits |
| 24 | `kernel_end.id` | Same program runtime id |
| 28 | `kernel_end.header` | Cycles dispatch_s waited for a free slot before publishing; 0 = no wait. The BRISC turns it into a stall marker and zeroes it |

The BRISC never lets an end time go backwards: a slot that saw no completion while open is given the previous
record's end time.

### 4.2 Ring entries and markers

The BRISC-to-NCRISC ring and the host FIFO hold 64-byte entries, because 64 bytes is the D2H socket page size: the
32-byte record followed by 32 bytes of padding. The BRISC also writes two kinds of marker entry, told apart by word 3:

| Entry | Word 3 | Words 0-1 | Word 2 |
|-------|--------|-----------|--------|
| Program record | 0 | Start time | Program runtime id |
| Sync marker | `0xFFFFFFFF` | Device time of the sample | Host time it answers |
| Dispatch-stall marker | `0xFFFFFFFE` | Time the stall ended (the next record's start) | Stall length in cycles |

On the host, each program record becomes a `ProgramRealtimeRecord` (runtime id, chip id, 64-bit start and end
timestamps, clock frequency, kernel source paths): 48 bytes on a 64-bit host.

### 4.3 Memory used

| Where | What | Size |
|-------|------|------|
| Dispatch core L1 | `realtime_profiler_msg_t`: 4 record slots (128 B), program-id FIFO of 32 ids (128 B), indices and control words (44 B) | 300 B |
| Prefetch and profiler core L1 | The same struct at the same address, because the dispatch memory map lays it out on every core it covers. Only the config and sync words are used on the profiler core | 300 B each |
| Profiler core L1 | BRISC-to-NCRISC ring: 64 B header plus 16,384 entries of 64 B | 1,048,640 B |
| Profiler core L1 | D2H socket config | 128 B |
| Profiler core L1, total | `RealtimeProfilerCoreL1` (ring plus socket config) | 1,048,768 B (~1 MiB) |
| Host, pinned memory | D2H socket FIFO: 32,768 pages of 64 B | 2 MiB |
| Host, heap | Record ring for the callback threads: 4 x min(2^20, 32,768 x devices) records of 48 B, capped at 2^22 records | 6 MiB for 1 device, at most 192 MiB |
| Host, heap | Per callback consumer: a batch buffer of min(2^20, 32,768 x devices) records | 1.5 MiB per consumer for 1 device |

`realtime_profiler_msg_t` is not part of `mailboxes_t`, so worker cores carry none of this: on a worker the same
address range is ordinary allocatable L1.

### 4.4 Mailbox fields by core

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
