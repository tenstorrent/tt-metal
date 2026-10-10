# Pipeline-Stage Gate (Hybrid Page Pool, fine-grain slot eviction)

Status: **D2D entry gate implemented and tested (§8); H2D ingress and runner/engine wiring pending.**
Visual overview (diagrams, interactive stage timeline): [`stage_gate_overview.html`](stage_gate_overview.html).
Branch: `snijjar/hybrid-page-pool-stage-gate-support`

## 0. Decisions log

| # | Decision | Source |
|---|---|---|
| D1 | **One gate per execution-lane slot** (`num_gates` = number of slots). | Review 2026-10-09 (Q1) |
| D2 | **Every stage receives the same number of copy-ins per slot**, so one gate header stamped at stage 0 and forwarded unchanged is valid on all stages. | Review 2026-10-09 (Q2) |
| D3 | **Aggregation / fan-in / fan-out is owned by higher-level software** (engine/KVM). The socket sees exactly one logical "open" per (gate, chip, request) and does no cross-chip or cross-request aggregation. | Review 2026-10-09 (Q3) |
| D4 | **The KVM writes the gate.** Gating is an **optional, compile-time mode** of the socket, off by default; with it off, the kernel and host paths are identical to today. | Review 2026-10-09 (Q4) |
| D5 | **Gates are binary enable/disable, not counters/thresholds.** There is no clean atomic increment across writers (no PCIe atomics; a counter needs a single producer with a cached count that has to be passed around). Every gate write is a plain store of a constant. | Review 2026-10-09 |
| D6 | **A chunk can carry several slots.** The gate header is a list `{n, (gate_id, flags) × n}`, bounded by a compile-time `max_gates_per_transfer`; the receiver waits until all `n` gates are `OPEN`. | Review 2026-10-09 (Q10) |
| D7 | **The socket's receiver closes the gate**, entirely inside the socket. Model op code never sees gating. The gate closes when the flagged chunk *enters* the stage, not when the stage finishes it, so a request queued right behind it for the same slot cannot slip through. With `CLOSE_ON_TRANSIT` set, exactly one "burst" (that chunk) passes and the gate shuts behind it. | Review 2026-10-09 (Q11, Q12) |
| D8 | **Copy-in/copy-out happens once per request per stage, not per chunk.** Multi-chunk requests keep the gate open across chunks; only the engine knows which chunk is the last one for a slot, so that bit must travel from the engine to every stage's receiver (§3.5). | Review 2026-10-09 |
| D9 | **Strict in-order** release at the gate for this release of the design; head-of-line blocking is accepted. | Review 2026-10-09 (Q5) |
| D10 | **One gate check at stage entry** for milestone 1. This holds because each stage has exactly one entry socket (rank N−1 → N). Revisit if a stage ever gets several entry sockets. | Review 2026-10-09 (Q7) |
| D11 | **Stage 0 is gated too, in the H2D ingress.** If H2D turns out to be materially different, the fallback is engine-side: the engine is the pusher and can hold a chunk until its stage-0 copy-in is done. | Review 2026-10-09 (Q8) |
| D12 | **The device clears the gate**, specifically the socket's receiver side. Opening is the KVM's job; closing (clearing the bits) is never done by the engine/KVM. | Review 2026-10-09 (Q11) |
| D13 | **Gate info travels in-band in the per-chunk metadata (Option C)**, packaged as a generic chunk-metadata record. The socket and layer completion only *read* it; models forward it opaquely and never interpret gate fields. Multi-request-per-chunk is deferred: when it lands it only appends more slot entries, so readers stay read-only and models stay unaware. | Review 2026-10-09 (Q13) |
| D14 | **The request is pipelined through the model; the gate command is not.** The chunk and its metadata flow stage to stage through the sockets (D13). The KVM is responsible for getting each stage's gate-open command submitted to that stage's chips, **including chips on remote hosts**. The transport is the KVM's choice (an agent on the owning host doing a PCIe store, or a fabric write from another chip). The socket accepts any plain 32-bit store of OPEN that is ordered after the copy-in data. | Review 2026-10-09 |

## 1. Purpose

In the advanced hybrid page pool, an execution-lane slot is occupied on a pipeline stage only for the
window `[copy-in done, layer-completion]` plus copy-in/copy-out time around it. The **gate** is the
lower bound of that window: a stage must not start computing a chunk that uses slot `s` until the KVM
has finished copying that request's pages into slot `s` on that stage. Layer completion (v2 structured
protocol, branch `snijjar/fix-hol-blocking-in-prefill-layer-completion-issue-54632`) is the upper bound.

Putting the gate in the stage-ingress stream service (D2D receiver; H2D receiver on stage 0) gives every
pipeline model the gate without touching model graphs or traces.

## 2. Existing machinery this builds on (main @ 116142d300e)

| Piece | Location | Relevance |
|---|---|---|
| `D2DStreamService{Sender,Receiver}` | `ttnn/api/ttnn/tensor/d2d_stream_service.hpp`, `ttnn/core/tensor/d2d_stream_service.cpp` | Host handle that owns per-coord service cores + the persistent kernels. Gate config/addresses hang off this. |
| Persistent D2D receiver | `ttnn/core/tensor/kernels/persistent_d2d_receiver.cpp` | Per-coord loop: lease grant → wait `bytes_sent` (data landed) → mcast metadata to workers → `data_ready` inc → wait `consumed` → ack upstream → release lease. **The gate check goes between "data landed" and "mcast metadata / data_ready".** |
| Service-core scratch words | `allocate_service_core_words` (`d2d_stream_service.cpp:90`) | Same mechanism as `termination_addrs` / `link_grant_addrs`; gate words are allocated here. |
| Chunk metadata | `prefill_runner.py` `_d2d_send` / `_d2d_recv`; words `[slot_id, actual_start, actual_end, (provided_levels)]` | Already traverses every stage and is visible in the receiver service core's L1 (`receiver_socket.read_ptr`) before workers are released. Gate selector is read from here. |
| Sentinels | `SHUTDOWN_METADATA_WORD=-1`, `WARMUP_METADATA_WORD=-2` | Must bypass the gate. |
| Stage-0 ingress | `H2DStreamService` → `models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/persistent_h2d_writer.cpp` | Needs the same gate (stage 0 has no D2D receiver). |
| Lease | `d2d_lease.cpp`, `share_fabric_links=True` | Receiver is granted before data lands; the gate wait happens while granted (see §6.4). |

## 3. Device-side programming model

### 3.1 Gate state
- Each participating coord's **receiver service core** owns `num_gates` L1 words, one per
  execution-lane slot (D1): `gate_base_addr + slot * gate_stride`, `slot ∈ [0, num_gates)`.
- `num_gates` is a `D2DStreamConfig` field. `0` (default) compiles the gate out entirely: no L1
  allocated, no CT-arg-gated code, no metadata change (D4).
- Each word is a **binary gate** (D5): `GATE_CLOSED = 0`, `GATE_OPEN = 1`. Any other value is invalid
  (the kernel may assert in debug builds). It is initialised to `GATE_CLOSED` at service construction.
- Every write is a **plain 32-bit store** of a constant: no read-modify-write, no atomics, no writer-side
  cached count. That works from the KVM over NoC/fabric and from the host over PCIe, which has no
  atomic increment (BH PCIe tile).
- Gate state is per chip (per coord). The socket does not aggregate across chips; the KVM/engine writes
  each chip's gate once that chip (or whatever set the higher-level software decides) is ready (D3).

Who writes what:

| Transition | Writer | How |
|---|---|---|
| `CLOSED → OPEN` | KVM, on copy-in completion for (slot, stage, chip) | fenced/dependent inline write of `1` |
| `OPEN → CLOSED` | **Proposed:** the receiver kernel itself, when it releases the **last chunk** of the request occupying that slot (§3.2) | local L1 store of `0` |
| either | Host (tests, fallback, error recovery/abort) | `set_gate(coord, slot, open)` |

### 3.2 Per-transfer gate header (in chunk metadata)
Each transfer carries a gate header in its metadata blob (layout: Q6, multi-slot form: Q10):
- `gate_id` (`uint32`): the slot whose gate to wait on; `GATE_NONE = 0xFFFFFFFF` bypasses.
- `gate_flags` (`uint32`): bit 0 `CLOSE_ON_TRANSIT`, set by the engine on the **last chunk of the
  request** for that slot.

All chunks of a multi-chunk request wait on the same open gate. The last chunk closes it. The engine
stamps the header at stage 0 and it is forwarded unchanged; per D2 every stage sees the same copy-in
pattern, so the same header is valid on every stage.

Sentinel transfers (shutdown/warm-up) set `gate_id = GATE_NONE`.

**Why self-close instead of the engine/KVM writing CLOSED:** it is ordered for free. The receiver's
store of `0` happens before it signals `data_ready` for the last chunk, so it happens before that chunk's
compute, before that stage's layer completion, before the engine sees the slot free, and therefore
before the KVM can write `1` for the next request. An external close would need its own ordering
against an in-flight next-request open, plus an extra control message per request.

### 3.3 Receiver kernel FSM (per transfer)

```
          ┌──────────────────────────────────────────────────────────────────────┐
          v                                                                      │
 [WAIT_GRANT] --link_grant==1--> [WAIT_DATA] --bytes_sent advanced--> [READ_GATE_HDR]
  (LEASE mode only)                                                       │
                                     gate_id==GATE_NONE ┌─────────────────┤
                                                        │                 v
                                                        │          [WAIT_GATE] poll until
                                                        │                 │  gate[gate_id]==OPEN
                                                        │                 v  (invalidate_l1_cache)
                                                        │          [CLOSE_GATE] if CLOSE_ON_RELEASE:
                                                        │                 │  gate[gate_id] = CLOSED
                                                        v<────────────────┘
                                             [RELEASE_TO_WORKERS] (mcast metadata, inc data_ready)
                                                        v
                                             [WAIT_CONSUMED] --num_workers acks-->
                                             [ACK_UPSTREAM] (fabric atomic-inc bytes_acked)
                                                        v
                                             [RELEASE_LEASE] (link_grant=0) ─────────────────────┘

 Every WAIT_* state also exits to [TERMINATED] when termination_semaphore == 1.
```

Notes:
- The gate is checked **after** data has landed and **before** workers see `data_ready`, so a gated
  transfer costs one receive buffer (the receiver backing tensor) and nothing else. The upstream sender
  stalls naturally on `bytes_acked` (buffer depth 1). Back-pressure propagates without new protocol.
- `CLOSE_GATE` comes **before** `RELEASE_TO_WORKERS`, so the close is visible before any downstream
  consequence of the last chunk (see the ordering argument in §3.2).
- The gate is **in-order**: the socket is a FIFO, so a gated chunk blocks every later chunk on that
  stage even if their slots are ready (Q5).
- Optional debug/perf: per-gate `stall_cycles` / `times_opened` words for observability.

### 3.4 Gate FSM (per stage, per chip, per slot)

Device-visible state is only `CLOSED`/`OPEN`; the engine-side lane state is shown for context.

```
   device gate     engine lane state
   ───────────     ─────────────────
   CLOSED          FREE
     │   engine issues copy_in(slot, stage, completion_write = {gate addr, OPEN})
   CLOSED          COPYING_IN
     │   KVM: copy-in data visible on chip → inline write gate[slot] = OPEN
   OPEN            READY
     │   receiver releases chunk 0..k-2 of the request (gate stays OPEN)
   OPEN            IN_USE
     │   receiver releases last chunk (CLOSE_ON_RELEASE) → gate[slot] = CLOSED
   CLOSED          IN_USE (last chunk computing)
     │   layer completion for (request, stage)  [v2 message]
   CLOSED          DRAINING → engine issues copy_out → COPYING_OUT → KVM done
   CLOSED          FREE
```

**Invariants** (owned by higher-level software, D3):
- **I1.** The KVM writes `OPEN` to a slot's gate only while that gate is `CLOSED`, i.e. only after the
  previous request's last chunk on that stage has been released. The engine already guarantees this by
  issuing copy-in only after layer completion + copy-out (`while not s.slot_free`). If it is violated,
  the `OPEN` write is absorbed by the still-open gate and then lost to the self-close, and the next
  request hangs.
- **I2.** Exactly one chunk per request per slot carries `CLOSE_ON_TRANSIT`, and it is the last one.
- **I3.** Request abort/cancel: the host (or engine via KVM) must force the gate `CLOSED` if the last
  chunk will never be sent.

### 3.4a Chunk metadata today (for Q6)

The socket treats metadata as an opaque blob of `metadata_size_bytes`; it never parses it.

| Hop | Built by | Layout | Where it lands |
|---|---|---|---|
| Engine → stage 0 (H2D) | `prefill_producer.py:119` `struct.pack("<3I", ...)` | 12 B: `u32 slot_id, u32 actual_start, u32 actual_end` | H2D service → worker L1 → `metadata_msg` device tensor |
| Stage N → N+1 (D2D), eager | `_d2d_send` rebuilds it from host-decoded words (`prefill_runner.py:321-338`) | 12 B, or 16 B with MTP (`+ u32 provided_levels`) | outbound op stages it in sender service-core L1 (`sender_metadata_l1_addr`) → shipped as an L1-aligned span after the data → receiver FIFO base → multicast to every receiver worker's L1 → inbound op snapshots it to a DRAM tensor |
| Stage N → N+1, traced (DeepSeek family) | `TtPrefillRuntime` copies it into a persistent `[1,1,1,3]` tensor and forwards that (`tt_prefill_runtime.py:885`) | exactly 3 words | same as above |

**Endpoints per rank** (Kimi-K2.7 4-rank production config `pipeline_prefill_request_4rank_kimi.yaml`
+ manifest `kimi27.json`: `PREFILL_USE_TRACE=1`, `PREFILL_LAYER_ACK_D2H=1`):

| Rank | Entry | Exit | Layer acks |
|---|---|---|---|
| 0 | `H2DStreamService` (`persistent_h2d_writer.cpp`, mapper `[Shard(0), Replicate]`) | `D2DStreamServiceSender` | `D2HStreamService` (metadata-only) |
| 1 … N−2 | `D2DStreamServiceReceiver` | `D2DStreamServiceSender` | same |
| N−1 | `D2DStreamServiceReceiver` | none | same |

The same-chip claim (F2) covers a rank's D2D receiver and D2D sender. Rank 0's H2D entry is a different
service/kernel/mapper; whether its coords match the D2D sender's is Q14.

**The metadata tensor on the traced path (Kimi-K2.7, DeepSeek family).** It never round-trips through
the host:
1. `inbound_socket_service_sync` returns `metadata_msg`, a fresh `[1,1,1,3]` uint32 DRAM tensor per
   chunk.
2. `_metadata_from_msg` (`tt_prefill_runtime.py:716-734`) `ttnn.copy`s it into the persistent
   `_trace_metadata_msg` (fixed address, captured by the trace), then slices it into per-field scalar
   tensors that traced ops read on device (slot, start, end).
3. Inside the trace, every layer's `zero_pad_and_ack` ships `_trace_metadata_msg` verbatim through the
   D2H service (`kv_ack.py:115`): **this is how the slot id reaches layer completion today.**
4. After replay, the runner forwards `_trace_metadata_msg` as the outbound metadata
   (`prefill_runner.py:449-452`).

The runner also reads the record on the host (`_decode_metadata`) for logging and the eager path. On the
eager path, `_d2d_send` rebuilds it from those host-decoded words.

**Layer-completion slot id on the host side.** On main, `LayerAckService` reads each D2H record but
ignores its bytes (`layer_ack_service.cpp:~85`). Layer = `first_layer + k % local_layers` and request =
`k / local_layers` are inferred by counting. The v2 protocol (branch
`snijjar/fix-hol-blocking-in-prefill-layer-completion-issue-54632`) is what parses
`slot_id, pos_start, pos_end` out of the record. **Consequence for multi-slot chunks (D6):** the
record, the `[1,1,1,3]` persistent tensor and its scalar slices, the D2H `metadata_size_bytes`, and the
v2 `LayerCompletionMessageV2` (single `slot_id`) all have to grow to a slot list. That work is needed
regardless of how the gate header propagates.

Sentinels: all words `0xFFFFFFFF` (−1) = shutdown, `0xFFFFFFFE` (−2) = warm-up. The same record also
serves as the per-layer D2H ack record (`kv_ack.py:115`).

Proposed socket-owned header, **not** part of the user blob (only meaningful with `num_gates > 0`):

```
 offset  field
 0       u32 chunk_seq          engine-stamped, monotonic; desync detection
 4       u16 kind               DATA | LOCAL (runner-originated, e.g. warm-up) | SENTINEL
 6       u16 n                  number of gates this chunk needs (≤ max_gates_per_transfer)
 8       u32 close_mask         bit i = CLOSE_ON_TRANSIT for gate_ids[i]
 12      u16 gate_ids[n]        slot indices (padded to L1 alignment)
```

With Option A it travels as a trailer after the user span. With Option B it never goes on the D2D
wire: each stage pops it from its local schedule queue, which makes Q6 moot.

### 3.5 Propagating the gate header stage to stage (open: Q13)

The engine is the only source of truth for which chunk closes a gate (D8), so its per-chunk header
`{n, (gate_id, flags) × n}` has to reach every stage's receiver without model code handling it. Facts
from the code survey (main @ 116142d300e):

- **F1. One in, one out, in order.** No model touches the D2D socket; all pipelined models
  (DeepSeek-V3 family incl. Kimi-K2/K3, GLM-5, Mistral-Small-4, DeepSeek-V3.2; MiniMax-M3; GPT-OSS) go
  through the shared loop in `prefill_runner.py:500-540`: recv → `prefill_chunk` → `_d2d_send`. Real
  chunks are never merged, split, reordered or retried. MTP / DFlash / Kimi-K3 AttnRes extras ride inside
  the same transfer.
- **F2. Entry and exit share a chip.** Each rank builds its inbound receiver and outbound sender on the
  same `mesh_device` with the same `D2D_MAPPER_CONFIG`, so both take the same `topology.mesh_coords()`
  (`d2d_stream_service.cpp:259-268`). `claim_service_cores` picks one core per coord on
  `mesh->get_device(coord)`, so for coord c the entry and exit service cores are **different cores on the
  same chip**. Coords are wired 1:1 across ranks (`build_connections`), and every stage is one mesh on
  one host.
- **F3. Ingress and egress strictly alternate per chip.** `_lease_reclaim` orders: inbound release →
  outbound done → re-grant inbound, so a chip never has more than one chunk between its entry and exit.
- **F4. Today's user metadata cannot carry the header.** The eager path rebuilds the record from three
  host-decoded words (`_d2d_send`, extra words dropped); the traced DeepSeek path copies it into a
  persistent `[1,1,1,3]` tensor owned by `TtPrefillRuntime`. Using it would leak into model code.
- **F5. Runner-level exceptions to F1:**
  - Trace warm-up is sent locally (`_forward_send_warmup`, metadata −2) and dropped downstream.
    **On rank 0 the outbound runs one ahead of the inbound** (out#0 = warm-up, out#1 = chunk 0).
  - The shutdown sentinel is rebuilt, not forwarded (still 1 in → 1 out).
  - Rank 0 with MTP rewrites the 12 B record into 16 B.
- **F6. No device-side schedule queue exists today.** Reusable patterns: the H2D socket and the dispatch
  prefetch queue (single producer, plain stores, data → `sfence` → write-pointer store; back-pressure by
  reading back the consumer pointer), and the `link_grant` word (binary, one side writes 1 only after
  seeing 0, the kernel writes 0).
- **F7. Fabric inline writes are not flushed.** `NOC_UNICAST_INLINE_WRITE` at the receiving EDM has no
  flush option, so it is not ordered after earlier payload writes on the same channel. Only the flushed
  atomic-inc / fused write+atomic-inc variants are (`fabric_edm_packet_transmission.hpp:189-231`).
- **F8. No host atomics through a TLB.** UMD has `WindowFlags::Atomic`, but `TlbWindow::configure`
  asserts that only direction flags are supported, and tt_metal never uses it. This confirms D5.

#### Option A — entry hands the header to the exit on the same chip

```
 stage N, chip c
 ┌───────────────────────────────── chip c ─────────────────────────────────────┐
 │  entry svc core (receiver)                     exit svc core (sender)        │
 │  WAIT_DATA → read socket header →              … workers produce →           │
 │  WAIT_GATE → CLOSE_ON_TRANSIT →  ──NoC write──> handoff ring (depth 2)       │
 │  push header ────────────────────┘             pop header → write into the   │
 │  RELEASE_TO_WORKERS                            socket-owned trailer → ship ──┼──> stage N+1
 └──────────────────────────────────────────────────────────────────────────────┘
```

- **Wire format.** The socket header lives in a socket-owned trailer after the user metadata span. It is
  written by the sender service core and read by the receiver service core, and never multicast to
  workers. Models and the runner's metadata tensors never see it, so F4 doesn't matter.
- **Handoff.** A small ring in the exit core's L1 (depth 2; F3 means depth 1 suffices). The entry core
  writes entry → barrier → write pointer (single writer, plain stores); the exit core owns the read
  pointer. Same chip, so a local NoC write; no fabric involved.
- **Stage 0.** The engine already writes each chunk's H2D record. It adds the socket header to it; the
  H2D ingress does the gate and the handoff exactly like the D2D receiver. `prefill_producer.py`'s
  `_Slot.target_chunks / next_chunk` already knows the last chunk of each request (F5 of the survey).
- **Locally originated transfers (F5).** The outbound op gets a `local_origin` flag (runner-level only,
  set by `_forward_send_warmup`). The exit core then ships `{n=0, kind=LOCAL}` without popping, and the
  next receiver neither gates nor pushes it. Without this, rank 0 would attach chunk 0's header to the
  warm-up. The shutdown sentinel stays 1:1: it carries `GATE_NONE` and is handed off normally.
- **Desync detection.** The header carries a 32-bit `chunk_seq` stamped by the engine. The exit core
  checks it against what it pops, and the receiver checks monotonicity. A mismatch is a hard failure,
  not a silent wrong gate.
- **Cost.** One header per chunk, written once by the engine. No extra control messages. ~16 B + 8 B/gate
  per transfer. Trace-safe: no socket op is captured in a trace except the D2H acks, which are unaffected.
- **Limits.** Relies on F1/F2. A future stage that splits or merges chunks, or whose entry and exit
  coord sets differ, breaks it. That would show up as a `chunk_seq` mismatch, not silently.

#### Option B — per-chip command queue in DRAM, streamed in by the socket

The engine/KVM writes a **burst** of per-chunk commands (`{chunk_seq, n, gate_ids[n], close_mask}`) into
a DRAM ring on each D2D socket's local chip. The socket's entry service core streams the command
sequence in and pops one command per received transfer.

- **Producer.** The KVM already reaches every chip of every stage for copy-in (per request, stage and
  chip), so the command burst rides along with that traffic. No new transport and no MPI hop. Writes are
  per burst, not per chunk: one DRAM write of N commands plus one write-pointer publish.
- **Publish and ordering.** DRAM entries first, then a write-pointer store in the service core's L1
  (single writer, plain store). From the KVM over fabric, the publish must be flushed after the entry
  writes (flushed atomic-inc or fused write+inc) because inline writes are not flushed (F7). From the
  host: PCIe write, `sfence`, pointer store (the H2D-socket pattern).
- **Back-pressure.** The engine bounds outstanding commands by construction (it knows chunks in flight
  from layer completions). Alternatively, the socket publishes its read pointer and the producer reads it
  back (the dispatch prefetch-queue pattern).
- **Consumer.** On `WAIT_DATA` the entry core NoC-reads the next command from DRAM, or keeps a few
  prefetched in L1, and then runs `WAIT_GATE` / `CLOSE_GATE` from it. A missing command (ring empty)
  stalls the chunk, which is the right default.
- **Correlation.** Commands match transfers by order. Desync can be detected without a new wire field:
  the command also carries the chunk's `(slot ids, actual_start)`, and the socket compares them against
  the existing metadata record at a configured offset. That record already propagates on every path,
  including the traced one.
- **Locally originated transfers** (warm-up, rebuilt shutdown) must not pop a command. Either the
  engine knows the runner sends exactly one warm-up per stage at bring-up and doesn't schedule for it,
  or the sender marks the transfer `LOCAL`, which needs a one-word socket-owned field on the wire.
  Recognising the −1/−2 sentinel words in the metadata record is a third choice, but couples the socket
  to the runner's sentinel convention.
- **Strengths.** No gate data on the D2D wire, which makes Q6 moot. It doesn't depend on F1/F2:
  works if a stage's entry and exit coords differ, and allows per-stage schedules. Stage 0 (H2D) is the
  same mechanism. Close decisions stay where the knowledge lives (the engine).
- **Costs.** The engine/KVM must produce a per-stage, per-chip command stream that exactly matches the
  chunk sequence. It needs a DRAM ring allocation per socket chip and its address in the gate descriptor
  (§5). The command must be in DRAM before the chunk arrives, or the stage stalls.

#### Option C — gate fields inside the per-chunk metadata record (in-band)

The record that already reaches every stage (§3.4a) carries the gate fields. The socket parses them at a
configured offset from the landed copy in the receiver's FIFO L1, before releasing workers.

- **Untraced path today.** The *stage-to-stage forward* goes through the host: `_d2d_send` **rebuilds**
  the outbound record from the host-decoded `slot/start/end[/levels]` (`prefill_runner.py:321-338`), so
  extra words are dropped. Fix: forward the inbound `metadata_msg` verbatim instead of rebuilding.
  Layer completion is **not** affected by trace vs. eager. The host is in the loop only with the
  host-callback ack transport (`kv_ack.py`: `synchronize_device` + callback eager, or a trace split at
  each ack). With `PREFILL_LAYER_ACK_D2H=1` the ack is a device op on the same CQ in both eager and
  traced runs. Eager passes the fresh per-chunk `metadata_msg`; traced passes the persistent copy.
- **Traced path today.** Propagates entirely on device through the persistent `[1,1,1,3]` tensor; it
  must grow to the full record, with the runtime treating words it doesn't read as opaque.
- **Synergy with Q16.** Multi-slot chunks force the record to carry a slot list anyway (KV writes, layer
  completion). With per-slot gates (D1) **the gate ids are that slot list**; the only gate-specific
  addition is the close mask (plus an optional `chunk_seq`).
- **Strengths.** The header travels with the chunk, so there is no correlation/desync problem. No extra
  engine/KVM traffic or DRAM ring. Sentinels self-describe (`n = 0`). Works for any topology.
- **Costs.** Runner/runtime must forward the record verbatim (eager rebuild removed; traced tensor
  resized in `TtPrefillRuntime` and Gemma's runtime). The record layout becomes a shared contract between
  engine, socket and layer completion. The model carries bytes it doesn't interpret, but it doesn't
  implement gating.

**Race analysis: can a later chunk clobber the metadata before it is consumed? (Q17, resolved)**
- **R1. Receiver FIFO copy: safe.** The record lands at the receiver's socket-FIFO base (fixed
  address). The upstream sender cannot ship the next record there until it sees `bytes_acked`, which the
  receiver sends only after its workers consume the transfer. The gate check runs before release, so it
  always reads the current chunk's copy.
- **R2. Sender staging L1: safe.** The outbound op copies the record from the tensor into the sender
  service core's L1 *inside the op*, before `data_ready`. The host's `wait_for_fabric_links()` before the
  next dispatch keeps the next op from overwriting staging before the previous ship
  (`outbound_socket_service_sync_writer.cpp:16-29`).
- **R3. Traced persistent tensor: safe today (Q17, verified 2026-10-09).**
  - All ops are on CQ 0 with no explicit `sub_device_ids`.
  - `SubDeviceTraceController.replay()` ends with `ttnn.synchronize_device` (`sub_device_trace.py:159`),
    so every chunk-k reader inside the trace (D2H acks, scalar readers) has finished before chunk k+1's
    `ttnn.copy` is enqueued.
  - The outbound op k consumes the DRAM record inside its own kernel (read → barrier → write to service
    L1 → barrier → `data_ready`).
  - Chunk k+1's copy also follows the lease wait, the inbound op, and the blocking `to_torch` in
    `_decode_metadata` (a read stalls on all sub-devices).
  - Sub-device manager switches emit device-side waits on the old manager's completion counts, so they
    can't reorder work across the switch.
  - **Caveat:** safety rests mainly on the end-of-replay host sync and the blocking `to_torch`. If either
    is removed for performance, same-sub-device go-signal ordering still covers the outbound op, provided
    the manager is cleared after replay. Re-verify at that point.
- **R4. Eager path: no fixed address** (fresh tensor per chunk), so no clobber, as long as it is
  forwarded verbatim rather than rebuilt.
- **R5. D2H ack service core: latent hazard, pre-existing and independent of the gate (inferred; needs
  a stall experiment).**
  - The metadata-only ack op copies the record into a **single** L1 slot on the D2H service core and
    increments `write_ack` without waiting for the previous ack to drain
    (`outbound_socket_service_sync_writer.cpp:112-123`; slot allocated at `d2h_socket_service.cpp:524-533`).
  - The persistent writer ships that slot later, after `socket_reserve` on the 4 KB host FIFO
    (`persistent_d2h_writer.cpp:109-129`).
  - If the host `LayerAckService` stops draining, ack N can ship with ack N+1's record. That is harmless
    within a chunk, but wrong across a chunk boundary.
  - Worse, the reader waits for `(cur - last_write_ack) == num_workers` with exact equality
    (`persistent_d2h_reader.cpp:76`). With `num_workers == 1`, two acks landing while it is blocked make
    the difference 2, and it never matches again, so the service hangs.
  - Experiment: sleep the host ack drain thread for more than one chunk, then check record slot/start vs.
    expected layer/chunk and watch for a stuck service.
  - Fix candidates: `>=` with per-transfer credit accounting, plus back-pressure (or a ring) for the
    metadata slot.

#### Options considered and rejected
- **Burst-count gate** (KVM writes `k` = chunks to admit, receiver decrements and closes at 0; still only
  plain stores with alternating writers, like `link_grant`). It removes the close flag but the gate ids
  still have to propagate, so it doesn't remove the propagation problem. It also departs from D5's
  strictly binary gate. Noted only as a possible later simplification.

### 3.6 Generic chunk-metadata record (D13), draft

One fixed-size record per chunk (fixed size because the traced path pins it at a captured address).
All fields are u32 little-endian. Producers: the engine (stage 0, H2D). Forwarders: runner/runtime,
verbatim, opaque. Readers: socket gate (receiver service core, from its landed FIFO copy), traced ops
(slot/start/end), layer completion (D2H ack record → `LayerCompletionMessageV2`).

Draft A, legacy-compatible (milestone 1):
```
 w0  slot_id          \
 w1  actual_start      } unchanged legacy prefix; sentinels −1 / −2 unchanged
 w2  actual_end       /
 w3  provided_levels  (MTP; 0 otherwise)
 w4  gate_flags       bit0 CLOSE_ON_TRANSIT (for slot_id); bit31 GATE_BYPASS
 w5  chunk_seq        engine-stamped, monotonic (desync/debug)
```
Draft B, generic/versioned (the multi-request-ready form):
```
 w0  magic_version    'CM' | version
 w1  kind             DATA | SHUTDOWN | WARMUP
 w2  chunk_seq
 w3  n_entries        ≤ MAX_ENTRIES (compile-time; fixes the record size)
 w4+ entry[i] = { slot_id, pos_start, pos_end, gate_flags }   (4 words each)
```
The socket needs only `{offset of n_entries, entry stride, offset of slot_id and gate_flags within an
entry}` as compile-time args, so it stays layout-agnostic beyond that. Draft A is Draft B with
`n_entries` fixed at 1 and the entry at w0.

**Plumbing needed for D13:**
1. `_d2d_send` forwards the inbound record verbatim (no host rebuild).
2. The traced persistent record grows from `[1,1,1,3]` to the full record (`TtPrefillRuntime`, Gemma4
   runtime), and the scalar slices keep reading their words.
3. `METADATA_SIZE_BYTES` / `D2D_METADATA_SIZE_BYTES` and the D2H ack `metadata_size_bytes` grow to the
   record size. The v2 layer-ack parser reads the slot entry/entries.
4. The engine (`prefill_producer.py`) packs the record, including `CLOSE_ON_TRANSIT` on each request's
   last chunk.
5. The socket reads the gate fields at the configured offsets (§3.3 `READ_GATE_HDR`).

## 4. Writer semantics (KVM)

- **Delivery (D14).** The KVM ensures the open is submitted for every stage's chips, local or on a
  remote host. Candidate transports, any of which the socket must tolerate:
  - an agent on the owning host doing a PCIe store through a TLB window;
  - a fabric write from another chip to the descriptor's `(mesh_id, chip_id, service_core_noc,
    gate_base_addr)`;
  - a device-side agent on the same chip doing a NoC write.
- **Write.** A plain 32-bit store of `kStageGateOpen` to `gate_base_addr + 4·slot` on the receiver
  service core. Idempotent: a retried `OPEN` on an already-open gate is harmless (subject to I1). No
  KVM-side gate state (counts, epochs) is needed.
- **Ordering requirement.** The gate write must be ordered after all copy-in data for that chip is
  visible. Otherwise workers could read stale slot memory.
  - Host PCIe writer: strict-ordered TLB plus `sfence` between the copy-in data writes and the gate
    store (the H2D-socket data-then-counter pattern).
  - Fabric writer: a plain inline write is **not** flushed against earlier payload writes on the channel
    (F7). Barrier the copy-in writes first, or publish `OPEN` as a flushed atomic-inc of +1 (legal: per
    I1 the gate is `CLOSED` = 0) or a fused write+atomic-inc.
  - Same-chip agent: a NoC write barrier on the copy-in writes before the gate write.

A host write path (`receiver.set_gate(coord, slot, open)` / `read_gate`) is provided for tests and as a
stand-in until KVM supports dependent writes.

## 5. Host API / surfacing (implemented for D2D)

`ttnn/api/ttnn/tensor/d2d_stream_service.hpp`:
- Constants: `kStageGateClosed = 0`, `kStageGateOpen = 1`, `kStageGateFlagCloseOnTransit = 1u << 0`,
  `kStageGateFlagBypass = 1u << 31`.
- `D2DStreamConfig::stage_gate` (`D2DStageGateConfig{num_gates = 0, slot_id_offset_bytes,
  gate_flags_offset_bytes}`). `num_gates == 0` compiles the gate out. `num_gates > 0` requires
  `metadata_size_bytes > 0`, and both offsets must be 4-byte aligned and inside the metadata
  (`TT_FATAL` otherwise). A slot id ≥ `num_gates` (sentinels −1/−2) or the Bypass flag passes ungated.
- `D2DStreamServiceReceiver`:
  - `get_num_stage_gates()`
  - `get_stage_gate_descriptor(coord)` → `D2DStageGateDescriptor{mesh_id, chip_id,
    service_core_logical, service_core_noc (virtual), gate_base_addr, gate_stride_bytes (4), num_gates}`.
    This is the KVM's copy-in completion target: gate *i* is the u32 at `gate_base_addr + 4·i`.
  - `set_stage_gate(coord, gate, open)` / `is_stage_gate_open(coord, gate)`: host PCIe write/read,
    not CQ-ordered. For tests, bring-up before the KVM writes gates, and force-close on abort (I3).
- Python: `create_pair` / `create_receiver` kwargs `num_stage_gates`, `stage_gate_slot_offset_bytes`,
  `stage_gate_flags_offset_bytes`; `ttnn.D2DStageGateDescriptor`; `ttnn.STAGE_GATE_FLAG_CLOSE_ON_TRANSIT`,
  `ttnn.STAGE_GATE_FLAG_BYPASS`.
- Prefill runner (not yet wired): with the legacy-compatible record (§3.6 Draft A), pass
  `stage_gate_slot_offset_bytes = 0` and `stage_gate_flags_offset_bytes = 16`, and publish the
  descriptors to the engine at bring-up next to the layer-ack channel / migration table.

Kernel (`persistent_d2d_receiver.cpp`): the gate CT block sits after the backing-tensor accessor args
(`next_compile_time_args_offset()`). The check runs after "data landed" and before the metadata
multicast and `data_ready` (§3.3). It polls the termination word, so teardown while gated is clean.

## 6. Interactions / constraints

1. **Trace**: the gate lives in the persistent service kernel, outside the traced model graph. No
   trace recapture or sub-device change needed.
2. **Stage 0**: H2D ingress needs the identical gate (`persistent_h2d_writer.cpp`). Per the engine
   pseudo-code, the engine submits the chunk without waiting for copy-in; the gate holds it.
3. **Slot-reuse safety is the engine's job**: the gate stops *compute before copy-in*. It does not stop a
   copy-in from overwriting a slot still in use — the engine must wait for layer completion + copy-out
   (`while not s.slot_free`) before issuing the next copy-in to that slot.
4. **Lease**: in LEASE mode the receiver is granted before data lands (`_lease_reclaim` at loop top), so
   it holds the grant while gated. The stage's host thread is already blocked reading metadata at that
   point (`_d2d_recv` → `to_torch`), so this adds no new stall, but the gate must never depend on that
   host thread making progress. Option: open the credit-return fabric connection only after the gate
   passes (it is needed only for ACK_UPSTREAM).
5. **Aggregation**: none in the socket (D3). If the stage-level policy is "open on all chips only when
   every chip's copy-in is done", the KVM/engine performs that fan-in and then fans the write out to
   each chip's gate.
6. **Multi-layer stages**: a stage-entry gate requires copy-in of *all* layers of the stage before
   compute starts. Per-layer gating (overlapping copy-in of layer L+1 with compute of layer L) cannot be
   done in the socket; it needs an in-graph gate op (Q7).
7. **Sentinels**: shutdown / warm-up must carry `GATE_NONE`, otherwise teardown deadlocks.
8. **Termination**: `WAIT_GATE` must poll `termination_semaphore` like every other wait.
9. **CQ ordering (deadlock hazard)**: while a transfer is held, the stage's consumer op (and, in LEASE
   mode, the `wait_for_fabric_links()` spin kernel) spins on that stage's CQ, so everything enqueued
   behind it on the same CQ is stuck. The copy-in and the gate open for a held slot must therefore NOT
   be enqueued on that stage's CQ behind the consumer. Valid transports: host-direct PCIe writes, a
   kernel on another CQ / chip / host (tested by `StageGateRemoteFabricOpener`), or ops enqueued on the
   same CQ *ahead* of the consumer op.

## 7. Open questions

- **Q1–Q5, Q7, Q8, Q10–Q12** — resolved; see D1–D12 in §0.
- **Q13 — Header propagation.** Resolved: Option C (D13).
- **Q18 — Generic chunk-metadata layout.** Open. Working default: Draft A (legacy prefix + `gate_flags`
  at w4, i.e. offsets 0 / 16). The socket is layout-agnostic (two configured offsets), so switching to
  Draft B only changes those config values.
- **Q17 — Metadata clobber race.** Resolved: safe on every gating/forwarding path today (§3.5 R1–R4).
  New finding: R5, a latent D2H ack-service hazard (stale record / hang under host back-pressure), is
  pre-existing and should be tracked separately.
- **Q15 — Locally originated transfers under B.** Engine doesn't schedule for the runner's warm-up,
  a one-word `LOCAL` kind on the wire, or the socket recognises the −1/−2 sentinel words?
- **Q16 — Multi-slot metadata record.** Who owns growing the per-chunk record (`<3I` → slot list) and the
  traced `[1,1,1,3]` tensor and `LayerCompletionMessageV2`? It is needed for layer completion regardless
  of the gate.
- **Q14 — Stage-0 coords.** Resolved, verified in code (2026-10-09). On every supported config, rank 0's
  H2D entry and D2D exit cover **all sp×tp coords** of the 2D mesh, one service core per device each.
  - Both mappers are 2D and cover every mesh axis. Replicate yields every coord on its axis
    (`distributed_tensor.cpp:271-284, 325-347`).
  - A 2D shard producing fewer chunks than the mesh axis is a `TT_FATAL` (`distributed_tensor.cpp:296-300`),
    and so is an uneven split (`:127-133`), so coords can't be silently dropped.
  - 8x4 production: H2D `[8,1,640]` → `[1,1,640]` per chip; D2D `[1,planes,5120,hidden]` → 640 rows ×
    hidden/4 (or replicated width). 32 coords each. MTP, DFlash and `emb_tp_sharded=False` change shard
    shapes only.
  - The H2D metadata record is written to every coord ("Same bytes to every device",
    `h2d_socket_service.cpp:1382-1392`) and multicast to every worker on each chip
    (`persistent_h2d_writer.cpp:135-153`).
  - Could only diverge with a `mesh_shape_override`/offset on one mapper or a 1D mesh; neither exists.
- **Q6 — Metadata layout.** Current layout and proposed header are in §3.4a. Moot under Option B.
- **Q9 — Diagrams.** Source: Google Doc `1x7B8Xexa-9gj41lxthU3if_Tz6247_JEFaEBxxFnxqU`. Not readable
  from this host yet: the Drive connector lacks read scope, and the OAuth file named in AGENTS.md is
  missing. The design is still inferred from the text.

## 8. Implementation plan and status

1. [done] `D2DStreamConfig::stage_gate`; per-coord gate array allocated on the receiver service core,
   initialised CLOSED; CT args for the receiver kernel.
2. [done] Receiver kernel `READ_GATE_HDR` / `WAIT_GATE` / `CLOSE_GATE` (compiled out when
   `num_gates == 0`).
3. [done] Host API: descriptor, `set_stage_gate`, `is_stage_gate_open`; nanobind + `ttnn` exports.
4. [done, passing on a 2x2 Blackhole mesh, 2026-10-09] gtests in `tests/ttnn/unit_tests/gtests/tensor/test_d2d_stream_service.cpp`:
   `StageGateDisabledByDefault`, `StageGateRequiresMetadata`, `StageGateDescriptorSingleChipPair`,
   `StageGateSemanticsSingleChipPair`, `StageGateSemanticsRowPair`,
   `StageGatePerCoordIndependenceRowPair`, `StageGateRemoteFabricOpener`. Semantics covers: held while
   closed, release on open, CloseOnTransit closes, multi-transfer burst through one open, sentinel
   slot passes, Bypass passes, teardown while gated. A mutation run (gate wait forced to pass) fails
   `StageGateSemanticsSingleChipPair`, so the test catches a gate that never holds. No regressions:
   `D2DStreamServiceTest.*` and `*StreamPipeline*` show 55 passed and 3 skipped (they need more than 4 chips).
   Per-coord independence: opening gate 2 on one receiver coord releases only that coord, and
   CloseOnTransit closes only that coord's gate. Remote opener: a kernel on a third chip
   (`kernels/stage_gate_remote_opener.cpp`) opens the gate over fabric using only the descriptor,
   once as an inline write of OPEN and once as a flushed atomic +1. A mutation run (opener aimed at
   the neighbouring gate word) fails that test.
5. [todo] Same gate in the H2D ingress (stage 0, `persistent_h2d_writer.cpp`).
6. [todo] Prefill runner / engine: grow the record (§3.6), forward it verbatim on the eager path,
   resize the traced persistent record, stamp `CLOSE_ON_TRANSIT` in `prefill_producer.py`, publish
   descriptors.
