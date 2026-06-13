# pi0.5 `tt_bh_glx` — Traced Pipeline: Design Journey & Rationale

How the BH-Galaxy pi0.5 pipeline went from *sockets-on-1×1-meshes (untraceable)* to a
*fully-traced, on-device, multi-core-fabric* pipeline — and the trade-offs baked into
the final architecture.

- Correctness: matches torch `Pi0_5Model.sample_actions` at **PCC 0.9988**.
- Latency: **~82 ms / inference** (5 denoise steps), down from ~96 ms.

---

## 0. The hardware canvas: BH Galaxy as an (8×4) mesh

```
          col 0      col 1      col 2      col 3
        +----------+----------+----------+----------+
 row 0  | prefill  | prefill  | prefill  | denoise  |
 row 1  | prefill  | prefill  | prefill  | denoise  |
 row 2  | prefill  | prefill  | prefill  | denoise  |
 row 3  | prefill  | prefill  | prefill  | denoise  |
 row 4  | prefill  | prefill  | prefill  | denoise  |
 row 5  | prefill  | prefill  | prefill  | denoise  |
 row 6  | vision   | vision   | vision   | embed    |
 row 7  | scratch  |   --     |   --     |   --     |
        +----------+----------+----------+----------+

  vision  = (1,3) submesh @ (6,0)   prefill = (6,3) submesh @ (0,0)
  denoise = (6,1) submesh @ (0,3)   embed   = (1,1) @ (6,3)   scratch = (1,1) @ (7,0)
```

This placement is **not arbitrary** — it is the thing that makes the cross-stage
fabric transfers legal (see §4).

---

## 1. Where we started: sockets on 1×1 meshes — untraceable

The original pipeline carved **every chip as its own 1×1 child submesh** and moved
data with **fabric mesh sockets** (`send_direct_async` / `recv_direct_async`) under
**FABRIC_2D**.

```
   parent (8x4)
     |
     +-- 1x1 submesh (chip A)   --socket-->  1x1 submesh (chip B)  --socket--> ...
     +-- 1x1 submesh (chip C)
     +-- ... (one 1x1 submesh per chip, ops issued here)

   begin_trace_capture(parent)  --->  captures NOTHING (ops are on the children)
```

- Ran fine **eagerly**, but `capture_trace` **hung** in `end_trace_capture`.
- **Root cause:** a TTNN mesh trace only records ops issued on *that mesh's* command
  queue. All ops were on **1×1 child submeshes**, so the parent trace captured an
  **empty 64-byte epilogue** (`n_entries=1`). The ops actually ran *eagerly*.
- The hang: that empty epilogue's **full-mesh finish barrier** waits for a completion
  entry from **every** device in the root-mesh range; the per-1×1-submesh work never
  uniformly covered the mesh → deadlock (`mesh_trace.cpp` `populate_mesh_buffer`).
- `MeshTrace` assumes **homogeneous full-mesh workloads**. There was no repo precedent
  for `create_submesh` + `begin_trace_capture`.
- Ruled out: idle-row theories, moving the trace root to a (7,4) compute submesh, and a
  tt-metal "finish-range" patch (later confirmed unnecessary and removed).

**Conclusion:** to trace, ops must be issued on **one mesh that covers all the chips
they touch** — not on per-chip 1×1 submeshes.

---

## 2. The fix: per-stage SPMD + `point_to_point` inside, sockets/host between

Run **each stage as one single-mesh SPMD computation**, captured as **one trace per
stage**; hand off **between** stages outside the traces.

```
  vision trace        prefill trace            denoise trace
  (1x3 mesh)          (6x3 mesh)               (6x1 mesh)
  [embed+27 blk] --→  [18 VLM blk, snake] --→  [Euler loop, KV x-attn] --→ actions
       |  p2p in-mesh      |  p2p in-mesh           |  p2p in-mesh
       |                   |                        |
       +== host/socket ====+==== host/socket =======+   (cross-stage, OUTSIDE traces)
```

- **Single-mesh SPMD ops ARE captured** (non-empty trace with real per-chip commands)
  — vs the empty trace from 1×1-submesh ops. This is the whole reason it works.
- **Within a stage:** `ttnn.point_to_point` for chip→chip hand-off. Works and is
  trace-captured under **FABRIC_1D** (rows *and* columns). It **hangs under FABRIC_2D**
  → use **FABRIC_1D globally**.
- **Between stages:** one-shot per inference, *outside* the traces (host-bounce first,
  later on-device sockets).

### Why this is the *only* trace-compatible option

```
  op on full stage mesh (all chips)   -> trace records real work   ✅  (garbage on idle chips, §8)
  op on 1x1 submesh (active chip only)-> trace records nothing      ❌  empty trace
  op on partial device-range          -> full-mesh finish deadlock  ❌  original hang
```

The trace *forces* every stage op onto **all** chips of the stage mesh.

---

## 3. Sockets vs p2p: which goes where, and why

| Location | Mechanism | Why |
|---|---|---|
| **Inside a traced stage** | `point_to_point` | self-contained op, no cross-call state → **trace-safe** |
| **Between stages (eager)** | `send_direct_async` sockets | faster kernel + multi-core; but carries socket flow-control state |

- **`send_direct_async` cannot replace `point_to_point` inside a trace:** sockets keep
  credit/semaphore flow-control state that **trace replay does not reset** → the 2nd
  replay deadlocks (observed empirically — re-sending on the same socket hangs unless a
  fresh recv buffer is allocated, which a trace can't do).
- So: **multi-core `point_to_point` inside traces; multi-connection `send_direct_async`
  for the eager cross-stage hand-offs.**

---

## 4. The collinear layout: *where each layer lives* enables the sockets

FABRIC_1D sockets only forward to a **direct-neighbor** chip (multi-hop FATALs;
`fabric.cpp:153`) and require **collinear** endpoints (same row or column; cross-axis
creation FATALs). So we *placed* each stage's chips to make every hand-off an adjacent,
collinear pair.

### Prefill "snake" (boustrophedon) — keeps consecutive layers collinear

```
  layer index L lives on prefill chip (r,c),  c = (L%3 if r even else 2-(L%3)),  r = L//3

          col 0      col 1      col 2
  row 0    L0    ->   L1    ->   L2            (left to right)
                                  |
  row 1    L5    <-   L4    <-   L3            (right to left)
            |
  row 2    L6    ->   L7    ->   L8
                                  |
  row 3   L11    <-  L10    <-   L9
            |
  row 4   L12    ->  L13    ->  L14
                                  |
  row 5   L17    <-  L16    <-  L15

  every hop ((r,c)->(r,c+1) or (r,2)->(r+1,2)) is between ADJACENT collinear chips
```

### Cross-stage hand-offs are adjacent & collinear by construction

```
  vision tail (6,2)  --adjacent (col 2)-->  prefill (5,2)  --p2p (5,2)->(5,0)->(0,0)-->  prefill head
  prefill row r (r,2) --adjacent (row r)-->  denoise (r,3)                                (KV migration)
```

- prefill row `r` holds layers `3r, 3r+1, 3r+2` → exactly the 3 layers denoise chip `r`
  needs. The KV grouping **already matches** the denoise sharding (no remap needed).

---

## 5. On-device KV migration (the big cross-stage win)

Old host-bounce: **36 full-mesh `ConcatMeshToTensor` gathers + 6 reshards ≈ 265 ms**.

New on-device path (per local layer index `j ∈ {0,1,2}`):

```
  Phase 1  (in-mesh p2p-gather, on prefill mesh):
     for each row r:  move layer (3r+j) KV shard  (r, snake_col) --p2p--> (r, 2)   [in-place]
     (skip rows already on col 2: FROM==TO FATALs)

  Phase 2  (adjacent socket, prefill -> denoise):
     for each row r:  socket (r,2) --send_direct_async--> denoise (r,3) = past_k[j] shard

  prefill (6x3)                         denoise (6x1)
   (r,0)(r,1)(r,2)  --gather-> (r,2) --socket-> (r,3)=chip r holds layer 3r+j
```

- **265 ms → ~11 ms warm (24×)**, e2e PCC unchanged (**0.9988**).
- Gotchas: `point_to_point(input, FROM, TO)` — arg1 is the *source* (misleading C++
  name); `point_to_point` does **not** support bf8_b → transfer in **bf16**, typecast to
  bf8_b *after* the socket.

---

## 6. vision→prefix: the on-device experiment (and why host is the default)

We built a fully on-device vision→prefix path (gated `PI05_VISION_SOCKET=1`):

```
  on vblk mesh (SPMD; only chip (0,2)=(6,2) holds the real vision output):
     post_ln -> MultiModalProjectorTTNN -> ttnn.embedding(lang)*sqrt(W) -> concat -> prefix
  then:  socket (6,2)->prefill (5,2)  -->  p2p (5,2)->(5,0)->(0,0)  (prefill head)
```

- **Validated: PCC 0.9987** (vs 0.9988 host; tiny bf16-vs-float delta).
- **Why it's OFF by default:**
  - Pure vision→prefix transfer is only **~4.8 ms** (the earlier "13 ms" conflated the
    unavoidable 27-layer vision compute).
  - On-device single-shot was **~10 ms — *not* faster** than host (~4.8 ms): the tensor
    is tiny, so recv-buffer allocation + on-device projector/embed dominate; a host
    bounce of a small tensor is already cheap.
  - It needs the **~1 GB embed table replicated on the vision mesh**.
  - Socket **recv-buffer reuse** hangs (fine one-shot, awkward to loop).
- So the on-device path is correct and "fully on-device," but it's a **latency wash** —
  host-bounce stays the validated default.

---

## 7. Making fabric faster: multi-core `point_to_point` + multi-connection sockets

`point_to_point` was hardcoded **single-core, single-link** → flat **2.7 GB/s** and the
#1 kernel-time consumer (~30%).

```
  Adjacent BH-Galaxy chips have 2 fabric links (probed: send_direct_async FATALs at 3 conns).

  point_to_point  1 core/1 link  : 2.7 GB/s   (original)
  point_to_point  2 core/2 link  : 5.3 GB/s   (this change; TRACE-SAFE) ✅
  send_direct_async  1 connection : 8.3 GB/s   (faster kernel: direct write)
  send_direct_async  2 connections: 15.5 GB/s  (used for eager hand-offs)
```

- **C++ change** (`send_/receive_program_factory.cpp`): one worker core per available
  link via `get_forwarding_link_indices`, round-robined; `split_work_to_cores` already
  splits packets; sender/receiver coord-matched so the semaphore handshake needs **no
  kernel change**. Fixed the cached-program override to update *all* worker cores (was
  `core[0]` only — would break trace replay).
- **Multi-connection sockets:** KV-migration + on-device vision→prefix sockets bumped
  1 → 2 connections.
- Net: prefill snake p2p **29 → 16 ms**; prefill replay **45.8 → 32.5 ms**; total
  **~96 → ~82 ms**; PCC unchanged.

---

## 8. The "garbage compute" — power, not latency; and where it pays off

Because the trace forces **full-mesh SPMD ops**, each `block.forward` runs on **all 18
prefill chips at once**, but only the active snake chip has the correct activation —
**the other 17 compute on stale/garbage data that's discarded.**

```
  forward call k (SPMD on 18 chips, simultaneously):

     chip (snake[k])   : real layer-k compute        <- on the critical path
     other 17 chips    : garbage compute (discarded)  <- parallel, NOT on the path
        then  p2p:  chip snake[k] --> chip snake[k+1]
```

- Confirmed in the perf sheet: SDPA runs on **all 18 devices** (17/18 garbage).
- **Costs power, not latency** — the 18 chips run in parallel, so wall-time = one chip's
  compute. Eliminating it gives **0 ms** latency on batch-1.
- **Where it pays off — pipeline parallelism (throughput):** with many inferences in
  flight, the "idle" chips do useful work on *other* sequences:

```
  time -->
  chip0:  seqA L0 | seqB L0 | seqC L0 | ...
  chip1:           seqA L1 | seqB L1 | seqC L1 | ...
  chip2:                    seqA L2 | seqB L2 | ...
   ...   (steady state: all 18 chips busy on different sequences)
```

- **Eliminating it entirely** needs either partial-mesh ops (re-opens the trace
  deadlock) or **tensor-parallel single-mesh** (production path, ~45 ms, all chips
  useful per layer) — a different architecture.

---

## 9. Reading the perf sheet — pitfalls

`ops_perf_results.csv` rows are **`(op, device)`**, not logical ops. Three multipliers
inflate row counts: per-device (×18 for SPMD), per-layer (×18), passes (eager +
capture + replay ≈ ×3). E.g. SDPA = 990 ≈ 3 × 18 × 18.

| You want | Use this |
|---|---|
| op-mix %, d2d-vs-compute *share of work* | sum `DEVICE KERNEL DURATION` (ratios are valid) ✅ |
| total device work / power / utilization | sum across all rows ✅ |
| **per-inference latency** | **measured `execute_trace` wall-clock** — NOT the sum ❌ |

- **Summing the column ≠ latency:** raw sum **831 ms** vs real **32.6 ms** (~25×) — it
  adds parallel per-device work serially, includes all passes, garbage, and p2p *wait*.
- **One device + one pass still isn't latency** (12.7 ms vs 32.6 ms): the snake's
  critical path **spans all 18 chips** (layer-c on chip c → p2p → layer-(c+1) on chip
  c+1 …); one device sees only ~0.8 ms of the 146 ms of p2p.
- **`PointToPointOp` "kernel duration" includes fabric *wait*** — chained hops show
  10–26 ms of mostly-idle "duration"; the true transfer is ~0.8–1.6 ms.
- **Tooling:** full-e2e Tracy capture is ~27 GB (profiles all one-time block-building
  setup) and fills the disk — profile **per-stage** (~3 GB). The prefill repro doesn't
  bake the prod flags → run with `source _bench_runs/pi05_production.env` (else L1
  circular-buffer clash at prefix=1024).

---

## 10. Final per-inference latency (5 denoise steps, PCC 0.9988)

| Stage / hand-off | Before opt | After opt |
|---|---:|---:|
| vision trace replay | 8.7 ms | 8.0 ms |
| vision→prefix hand-off | ~4.8 ms | ~4.8 ms (host) |
| **prefill trace replay** | 45.8 ms | **32.5 ms** |
| KV migration | ~265 ms (host) | **~11 ms** (on-device) |
| denoise trace replay | 26.2 ms | 26.1 ms |
| **Traced compute (3 replays)** | 80.6 ms | **66.6 ms** |
| **Total / inference (warm)** | ~96 ms | **~82 ms** |

Remaining headroom is the **prefill matmul compute** (block-bound now that p2p is
~halved); reducing it further needs tensor-parallelism, not transfer optimization.

See also: `models/experimental/pi0_5/tests/perf/TRACED_E2E_PERF.md`.
