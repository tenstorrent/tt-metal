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

> **Note:** the profiler core runs two kernels — a BRISC *reader* (`cq_realtime_profiler.cpp`) that NOC-reads each timestamp pair from dispatch_s into an L1 ring buffer, and an NCRISC *pusher* (`cq_realtime_profiler_push.cpp`) that drains the ring to the host over the D2H socket and services the sync handshake (§3). The single loop drawn above is split across those two kernels so the reader never blocks on the PCIe push.

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

Host and device timestamps are aligned so consumers (Tracy, callbacks) can relate device cycles to host time. The host keeps a per-chip affine mapping `device_cycle = frequency * host_ns + device_cycle_offset`: `frequency` is fit once at init; `device_cycle_offset` is re-anchored continuously by a free-running servo on the receiver thread (~every 50 ms).

Each handshake is one-shot and rides a host-pinned ACK word; the device never writes a sync record into the timestamp FIFO. The host writes a 32-bit token into `sync_host_timestamp`; the profiler core's NCRISC pusher (`cq_realtime_profiler_push.cpp`, so the drop-critical dispatch_s read path is never stalled by sync work) sees it non-zero on its next loop iteration, captures the device WALL_CLOCK, and NOC-writes it — then the token — straight into the host's pinned ACK buffer (a device->host write that bypasses the record FIFO).

```
  HOST                                   REAL-TIME PROFILER CORE (NCRISC pusher)

    |  Write sync_host_timestamp = T (L1)  |
    | -----------------------------------> |  see sync_host_timestamp != 0
    |                                      |  capture device wall clock (D)
    |                                      |  stage D in sync_ack_device_time (L1)
    |  Poll pinned ACK word until == T;    |  NOC-write D, then T, into the host's
    | <---------- D, then T -------------- |    pinned ACK buffer (bypasses FIFO)
    |  read D from the ACK buffer;         |  clear sync_host_timestamp
    |  re-anchor offset at the midpoint    |
```

The device writes `D` before the token, so once the host observes the token `D` has already landed. The offset is re-anchored at the round-trip midpoint (minimax placement, error <= RTT/2 without assuming a symmetric latency); the reported `sync_error_ns` is that half-RTT.

**Init** repeats the handshake ~100 times (reading `D` from the ACK buffer each time) and fits `frequency` by linear regression. **Steady state:** the servo issues one handshake per device every 50 ms and re-anchors `device_cycle_offset` to track clock drift. A device whose host ACK word could not be set up (`sync_ack_pcie_xy_enc == 0`) is simply left unsynced — there is no record-FIFO fallback.

---

## 4. Carve-out layout (conceptual)

| Location | Contents (`realtime_profiler_msg_t`) |
|----------|----------------------------------------|
| **Dispatch_s L1** | Ping-pong buffers, program_id_fifo, **realtime_profiler_core_noc_xy**, **realtime_profiler_remote_state_addr**, realtime_profiler_state. Host writes NOC XY and the profiler tensix L1 address of `realtime_profiler_state` for NOC signaling. |
| **Profiler tensix L1** | **config_buffer_addr**, **realtime_profiler_state**, sync_host_timestamp (host->device token), sync_ack_device_time and sync_ack_* (device->host WALL_CLOCK + pinned-ACK address), sync_request (L1 staging for the ACK NOC-write). |

Layout: `tt_metal/hw/inc/hostdev/realtime_profiler_msgs.h`. HAL: `tt::tt_metal::realtime_profiler_msgs`. Not in `mailboxes_t`.

---

## 5. File / Component Reference

| Component | File(s) |
|-----------|--------|
| Dispatch_s (timestamp record + signal) | `tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp`, `realtime_profiler.hpp` |
| Profiler-core kernels (BRISC reader + NCRISC pusher/sync) | `tt_metal/impl/dispatch/kernels/cq_realtime_profiler.cpp`, `cq_realtime_profiler_push.cpp` |
| Host init, sync, receiver thread | `realtime_profiler_manager.cpp` |
| Shared struct + HAL accessors | `realtime_profiler_msgs.h` → `realtime_profiler_msgs` (generated) |
| Public API (register / unregister / is-active) | `tt_metal/impl/realtime_profiler/realtime_profiler.cpp` |
| Record fan-out (service, Tracy, user callbacks) | `tt_metal/impl/realtime_profiler/realtime_profiler_service.cpp`, `realtime_profiler_consumer.hpp`, `realtime_profiler_tracy_consumer.cpp` |
| Kernel-source metadata (runtime_id → sources) | `tt_metal/impl/dispatch/data_collector.cpp` |
