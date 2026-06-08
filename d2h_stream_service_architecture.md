# D2H Stream Service — Architecture Summary

## Problem

Repeatedly move a **fixed-shape** device tensor to the host without **rebuilding/dispatching a program every iteration**. Mirror the existing **H2DStreamService** pattern for the device→host direction, with optional **multi-core production** and **cross-process** reads.

---

## Design thesis

**Separate concerns into layers:**

1. **Stream service** — long-lived orchestration (backing tensor, sync, persistent kernel)
2. **Socket** — transport (pinned host FIFO + flow control)
3. **Kernels** — device-side behavior (stream out; optionally produce backing in parallel)

Setup once; hot path is **signal → produce → ack → stream → read → barrier**.

---

## Component diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         D2HStreamService (host)                          │
│  • global_spec + mapper → backing device tensor                          │
│  • service cores + semaphores/counters                                   │
│  • one D2HSocket per mesh coord                                          │
│  • launches persistent_d2h_sender (MeshWorkload, non-blocking)         │
│  API: read_from_tensor*, barrier, export_descriptor / connect            │
└───────────────┬──────────────────────────────┬───────────────────────────┘
                │ owns                         │ owns
                ▼                              ▼
┌───────────────────────────┐    ┌──────────────────────────────────────────┐
│ Backing device tensor     │    │ D2HSocket (× mesh coords)                │
│ Fixed DRAM buffer         │    │ • pinned host FIFO (fifo_size_bytes)     │
│ Source of truth on device │    │ • bytes_sent / bytes_acked flow control  │
└───────────────▲───────────┘    │ • L1 config on service core              │
                │                └──────────────────▲───────────────────────┘
                │ write                           │ PCIe push / host read
     ┌──────────┴──────────┐                      │
     │ Workers (optional)  │         ┌────────────┴────────────┐
     │ or host CPU copy    │         │ persistent_d2h_sender   │
     └─────────────────────┘         │ (service core, loop)    │
                                     └─────────────────────────┘
```

---

## Core abstractions


| Abstraction           | Responsibility                                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Backing tensor**    | Fixed device DRAM buffer; all D2H data is read from here                                                                  |
| **Service core**      | Off-grid core for persistent I/O (sender + socket config + sync words)                                                    |
| **D2HSocket**         | Ring buffer in pinned host memory; device DMA in, host copies out                                                         |
| **Chunk plan**        | Derives `socket_page_size` × `num_socket_pages` from tensor pages + scratch CB budget                                     |
| **Metadata master**   | Designated worker core; **only when metadata enabled**; writes its slice, fans its local replicated metadata **in** to the service-core staging region, then acks — like every other worker but with the metadata write |
| **Owner / connector** | Same API surface; owner holds device + kernel; connector attaches via `/dev/shm` descriptor                               |


---

## Operating modes

### A. Host-only (Python test)

- **1 kernel:** `persistent_d2h_sender`
- Host writes backing → `read_from_tensor()` → internally `notify_backing_ready()` (write_ack++) → sender streams → host drains FIFO → `barrier()`

### B. Worker-sync, no metadata

- **1 worker kernel role** on every core in `worker_cores`
- Handshake per iteration:

```
sender: transfer_done++ (multicast)  →  workers write slices  →  each worker write_ack++
sender: wait num_workers acks         →  stream backing → FIFO  →  host read + barrier
```

### C. Worker-sync + metadata

- **1 worker kernel**, same on every core; an `is_master` runtime arg picks which core also forwards
  the metadata. There is **no cross-talk between worker cores** — no roll-call, no peer-release.
- Each worker writes its slice; the master additionally reads its **local replicated** metadata copy
  and writes it **into the service-core staging region** (fan-in, host-ward) before its own ack. The
  service core waits for all `num_workers` acks, so the master writing metadata before acking is
  sufficient ordering — no inter-worker semaphore is needed.

```
sender: transfer_done++  →  workers write slices (master also fans metadata in to service core)  →  all workers write_ack++
sender: stream backing + metadata page  →  host read + barrier
```

Worker programs are **launched by the model/test**, not by `D2HStreamService`.

---

## Sync primitives


| Signal                       | Location                    | Purpose                                                       |
| ---------------------------- | --------------------------- | ------------------------------------------------------------- |
| `transfer_done_sem`          | Worker L1 (GlobalSemaphore) | Sender unlocks workers after prior host read                  |
| `write_ack_counter`          | Service core L1             | Each worker acks once after writes (+ metadata, if enabled)   |
| `termination`                | Service core L1             | Clean shutdown of persistent loop                             |
| `bytes_sent` / `bytes_acked` | Socket FIFO                 | Transport-level backpressure                                  |


---

## Data path (one iteration)

```
Produce backing  →  (metadata fan-in)  →  worker acks  →  Sender streams  →  Host read  →  barrier
     (workers/host)      (master → svc core)   (write_ack++)     (FIFO)           (API)
```

---

## Cross-process design

- **Owner:** `export_descriptor(service_id)` → flatbuffer in `/dev/shm/tt_d2h_stream_service_<id>.bin`
- **Connector:** `connect(service_id)` — no `MeshDevice`; `read_from_tensor` + `barrier` only

---

## Key design decisions


| Decision                                 | Rationale                                                                             |
| ---------------------------------------- | ------------------------------------------------------------------------------------- |
| **Persistent sender kernel**             | Amortize program build/dispatch; steady streaming                                     |
| **write_ack counter (H2D mirror)**       | Sender waits for exactly `num_workers` producer acks before streaming                 |
| **Master only for metadata**             | Metadata is replicated identically across cores, so one core fans it in to the service core (host-ward); no master when metadata disabled |
| **Metadata after writes, before ack**    | Backing slices complete first; the master fans metadata in before its own ack, and the sender waits for all acks — so no inter-worker handshake is needed |
| **Compile-time master + metadata flags** | Hot path branches eliminated via `if constexpr`                                       |
| **Socket as transport layer**            | Reuse H2D socket infra (FIFO, pinned memory, cross-process descriptors)               |


---

## H2D symmetry


| H2D                                               | D2H                                                     |
| ------------------------------------------------- | ------------------------------------------------------- |
| `H2DStreamService`                                | `D2HStreamService`                                      |
| `forward_to_tensor`                               | `read_from_tensor`                                      |
| Persistent **receiver**                           | Persistent **sender**                                   |
| Host → device                                     | Device → host                                           |
| Workers **consume** after ready                   | Workers **produce** before ack                          |
| Service multicasts metadata **out** to workers before data_ready | Master fans metadata **in** to the service core after writes, before its ack — no peer release needed |


---

## One-line architecture

**A long-lived service-core sender waits for per-worker write acks, then streams a fixed backing DRAM tensor through per-coord D2H socket FIFOs; when metadata is enabled, one designated worker (the metadata master) fans replicated metadata in to the service core (host-ward) before its ack — no inter-worker handshake.**
