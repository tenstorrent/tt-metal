# Ring-fused `indexer_score`: porting ring-joint-SDPA-style AG/compute overlap

**Status:** investigation complete; **Steps A–E DONE and green** (correct). **Step E profiled (2026-07-08): fusion is a NET LOSS at ring-of-4 / this K size (~0.73–0.85×).** See §7 (Step E row) + §6.1. Next: decide — shift the AG/compute balance (bfp8_b K, larger per-chip work) or shelve fusion at this scale.

### Step E perf finding (2026-07-08) — measured, honest
Host-wall-clock, cold single dispatch, min over 20 iters, semaphores reset (not reallocated) per iter, one `synchronize` each (`test_indexer_score_lb_ring4_perf.py`):

| case | AG-only | score-only | UNFUSED (AG+score) | FUSED | ideal max(AG,score) | speedup |
|---|---|---|---|---|---|---|
| glm5-contiguous | 742 | 322 | 924 | 1182 | 742 | **0.78×** |
| glm5-block_cyclic | 744 | 295 | 910 | 1079 | 744 | **0.84×** |
| dsv32-contiguous | 753 | 484 | 1084 | 1489 | 753 | **0.73×** |
| dsv32-block_cyclic | 752 | 455 | 1062 | 1255 | 752 | **0.85×** |

(µs. Includes a fixed per-dispatch host overhead, so absolute µs are not device-pure — but the direction is robust.)

**Clincher (why this is trustworthy despite host overhead):** the fused op is ONE dispatch (less host overhead) vs the baseline's TWO, yet it is consistently *slower* → the device-side fused execution genuinely regresses; co-scheduling costs more than the overlap saves.

**Root cause:** (1) the **AG dominates** — AG-only (742–753µs) is *larger* than score-only (295–484µs), so the overlap ceiling is only `max(AG,score) ≈ AG`, a ~20% best case over sequential even with perfect overlap. (2) **Core-budget contention** wipes even that out: the fused op confines the AG workers to one reserved grid row AND shrinks the compute rectangle by that row, so BOTH the AG (cramped, less fabric parallelism than the standalone full-grid AG) and the compute (fewer cores) run slower, and the per-band gate serializes compute behind the AG-dominated critical path. This is exactly the "modest win / grid contention can negate it" caveat from §6.

**To make fusion win here you'd need the balance to flip (compute ≥ AG) and/or the AG to keep its parallelism when co-scheduled:** bfp8_b K halves the AG bytes (Step F), larger per-chip compute (longer sequences / more heads) raises score-only, and more AG links/workers help only if spare cores exist. At ring-of-4 with this bf16 K, none hold — so fusion is not worth it *at this scale* as built.

### Step B decision log (2026-07-08)
- **Architectural fork found:** the only Linear+fuse-capable AG (`ring_attention_all_gather_async_multi_core_with_workers_helper`, the one Step A proved byte-exact) is **`ProgramDescriptor`-only**, but `indexer_score`'s factory is **classic `Program`-model**. Co-scheduling needs both AG workers + compute cores in ONE program (same dispatch) to overlap, so they can't be separate programs.
- **Decision (user):** **migrate to the descriptor model + reuse the Step-A-proven `ring_attention` AG** (not the classic `strided_all_gather_async`, which would need a fresh layout-equivalence proof + a different signaler).
- **Low-regression strategy:** `program_factory_t` is already a `std::variant`. Add a SECOND, descriptor-based `IndexerScoreFusedProgramFactory` to the variant and leave the existing classic factory **byte-identical**; `select_program_factory` returns the fused one only when fusion attrs are present. All current (non-fused) usage is untouched.
- **Descriptor factory shape** (mirrors `ring_joint_sdpa`): implement `create_workload_descriptor(...)` → `WorkloadDescriptor` (one `ProgramDescriptor` per coord), wrapped by `MeshDeviceOperationAdapter::DescriptorMeshWorkloadAdapter`. The adapter auto-patches buffer addresses on cache hits; per-core scalar override (chunk_start/straddle on a hit) is deferred (tests dispatch cold → always a miss → correct).
- **Recipes extracted (verbatim-applicable):** co-schedule = push indexer kernels into `desc.kernels` first, build `RingSDPAFusedOpSignaler` (`init_all_gather` + inlined `init_fused_op` pushing 2 WORKER semaphores over the compute grid, MULTI), append the 6-arg block `{ring_size, ring_index, fwd_writes, bwd_writes, sem0, sem1}` to the reader, then call the AG helper with a `ccl_core_grid_offset` that keeps AG workers OFF the compute rectangle. Linear length-N, device r: `forward_writes_expected = r`, `backward_writes_expected = N-1-r`. Reader coarse barrier = `Semaphore(fwd_sem=sem[0]).wait_min(bwd_writes+1)` if `bwd_writes>0`, and `Semaphore(bwd_sem=sem[1]).wait_min(fwd_writes)` if `fwd_writes>0` (edge devices auto-skip). Step B host-seeds the local slab (defers device prologue copy to C/D).
**Oracle:** `tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score_lb_ring4.py`
(unfused ring-of-4 all-gather + `indexer_score_dsa`, contiguous + block-cyclic, heads {8,16}).
**Goal:** replace the production two-phase flow (`_gather_index_kbuf` → `_sp_all_gather` → `indexer_score_dsa`,
`models/demos/deepseek_v3_d_p/tt/mla/indexer.py:576-604`) with a single fused op that scores the local K
slab immediately and overlaps each remote slab's compute behind the ring all-gather — the way
`ring_joint_scaled_dot_product_attention` overlaps the KV all-gather with attention.

This document is the output of a 14-agent investigation (7 readers mapped both architectures, 3
independent designers proposed ports, 3 adversarial critics tore them down, 1 synthesizer converged).

---

## 1. Verdict

**Feasible: yes, with caveats.** And structurally *easier* than ring-joint SDPA.

The win is **real but modest**: it removes an op boundary and the `_gather_index_kbuf` intermediate copy,
and overlaps local + already-arrived remote compute behind fabric transport. It does **not** remove the
O(T) `[B,1,T,D]` gathered-buffer DRAM materialization, and the overlap is bounded by a serial last-hop
tail (see §6).

---

## 2. Why it's easier than ring-joint SDPA

| | ring-joint SDPA | `indexer_score` |
|---|---|---|
| Key-axis dependency | **Reduces** (online softmax: every K feeds every output row's running max/sum) | **None** — `score[b,0,s,t] = Σ_h relu(q[b,h,s,·]·k[b,t,·])·w[b,h,s]` sums over **heads** only; column `t` depends only on key `t` |
| Cross-iteration state | Online-softmax accumulator carried across all ring iters; no output final until ring completes | **None** — `cb_acc_strip` holds no cross-iteration state |
| Fusion shape | **Accumulate** | **Scatter** — ring step `i` scores the slab that arrived and writes *its* columns |

Because columns are independent, there is no accumulator to carry and no cross-iteration per-output state.

---

## 3. The key insight: invP is absorbed at read time → placement is by `ring_id`, not arrival

The block-cyclic permutation looked like the hard part (composing `invP` against ring arrival order). It isn't:

- The reader **already** gathers K in *logical* token order via `BC_KTILE = logical→chunked_physical`
  (`reader_indexer_score.cpp:63-70`), with `shard(L) = (L/cl_t) % sp`, `cl_t = block_cyclic_chunk_local/32`.
- The all-gather deposits source-shard `c` at the **fixed** physical band `[c·sll_t, (c+1)·sll_t)` keyed on
  `ring_id` (`tile_id_start = ring_id·slice`), `sll_t = T/sp/32`. That is **exactly** the physical tile
  `BC_KTILE(L)` already targets.
- **Placement is by `ring_id`, fully decoupled from arrival order.** Arrival only gates *when* a shard is
  resident, never *where* a column lands.

### Consequences
- **Compute (`compute_indexer_score.cpp`): byte-identical, no change.** Same natural band order; timing
  dictated by CB fill.
- **Writer (`writer_indexer_score.cpp`): byte-identical, no change.** `write_strip`
  (`offset_bytes = k_tile_start·frag_bytes`) already scatters each band's contiguous natural-order columns
  correctly, because K is delivered logically. The block-cyclic stride is *invisible* downstream.
- **Causal mask unchanged** — keyed on the logical column (`work_split.hpp` `causal_diag_tile` /
  `valid_prefix_tiles`).
- The port **cannot mis-place a column** regardless of ring order.

### Column mapping (resolved)
Ring-iteration `i` yields source-shard `c = ring_id`.
- **Contiguous** (`BC_KTILE = identity`): shard `c` owns the single contiguous column band
  `[c·sll_t, (c+1)·sll_t)`; a band's tiles fall in ≤2 adjacent shards.
- **Block-cyclic** (production): source-shard `c` owns the *strided* natural columns
  `{g·chunk_global + c·cl + o}` = `T/chunk_global` disjoint width-`cl` bands — but the writer never sees the
  stride. For a logical band `[b·KC, b·KC+KC)`, the reader computes the distinct shards `{shard(L)}` and
  waits each non-local one.

So the *only* remaining hard parts are **timing/wiring**, not math:
(a) sourcing the local shard (the AG omits it from the gathered buffer), and
(b) getting the 1×4 Linear-submesh wait thresholds right.

---

## 4. Architecture

One fused ttnn op (e.g. `ttnn::experimental::indexer_score_dsa_fused`, or a fused branch of the existing op).
Its program factory co-schedules the all-gather producer program and the indexer consumer program into one
workload sharing **exactly two direction semaphores**.

- **Producer:** drive `ring_attention_all_gather_async_multi_core_with_workers_helper` — the **only**
  Linear+fuse-capable AG. The default `all_gather_async` factory hard-FATALs on Linear+fuse
  (`all_gather_async_default_program_factory.cpp:317`). Populate `AllGatherFusedOpSignaler`,
  `cluster_axis = block_cyclic_sp_axis`, `Topology::Linear`.
- **Gathered buffer:** internally-allocated `[B,1,T,D]` DRAM scratch. Remote shard `c` lands at `c·sll_t`;
  **the local shard is NOT written there by the AG** (`writer.cpp:131-137`).
- **Consumer:** the indexer program, with the reader as the sole fused consumer.

### 4.1 Signaling (resolved)
- **REUSE the producer path verbatim:** `AllGatherFusedOpSignaler` + device `OpSignaler` (mcast +1 per
  delivered slab, MULTI mode) — both op-agnostic. MULTI is correct because every indexer core processes
  columns from every shard. The indexer factory creates two direction semaphores on the indexer worker grid
  (like `ring_fusion.cpp:52-54`), calls `init_fused_op` with the worker NOC list + two sem ids + MULTI.
- **DIVERGE on the consumer side:** do **not** instantiate `RingSDPAOpReceiver`/`RingIdSequencer` in the
  kernel, and do **not** loop `ring_size` times (indexer is band-major, not slab-major). Instead the **host
  replays `RingIdSequencer` once** (seeded with per-device 1×4-Linear `forward/backward_writes_expected`;
  edge devices 0 and 3 are one-directional) to emit a `shard→(dir,val)` delivery table passed as reader
  runtime args. Inherit verbatim from the replay: the crossed direction-index swap
  (`fused_op_receiver.hpp:28-31`), the asymmetric `forward = received+1` / `backward = received` thresholds,
  and the forward-writer `+1` pre-signal (`ring_id_sequencer.hpp:60-66`) — reuse the pre-signal and the `+1`
  together or drop both; a mismatch is an off-by-one hang.
- Because the consumer only **reads** monotone counters and never advances a shared sequencer, it **cannot
  desync on pruned/empty iters** — this sidesteps SDPA's "advance-every-iter-or-hang" gotcha entirely.
- **Compute and writer need no signaling** — gated purely by `cb_k` backpressure (the reader is the sole waiter).

### 4.2 Local-shard sourcing (staged — do NOT make dual-source the first bring-up dependency)
- **Stage 1 (prologue copy):** a prologue NoC copy of the local slab into its `[ring_index·sll_t, …)` region
  of the gathered buffer, leaving one unchanged full-T accessor and a byte-identical scoring path.
- **Stage 2 (dual-source):** a second `TensorAccessor` over the SP-sharded input cache; per tile, if
  `shard == ring_index` read local at page `(BC_KTILE(L) - ring_index·sll_t)·head_dim_tiles + d`, else read
  gathered at `BC_KTILE(L)·head_dim_tiles + d`. This is the block-cyclic exact-equality hinge. Preserve
  `k_batch_page_offset (= cache_batch_idx·Tt·Dt)` in whichever path is used —
  `read_k_chunk_streaming` currently omits it and must be fixed if used with an indexed cache.

---

## 5. Device kernel change list

The **only** device change is in the reader.

- **`reader_indexer_score.cpp`** — parse a new fused-ring RT block: 2 semaphore ids (load-order crossover
  preserved), `ring_size`, `ring_index`, `sll_tiles (= Tt/BC_SP)`, the per-shard `(dir,val)` delivery table
  (`2·ring_size` entries), and the gathered-buffer base addr.
- **`reader_indexer_score.cpp` `read_k_chunk` (255-282) / `read_k_chunk_streaming` (287-312)** — BEFORE the
  async reads for a band, compute the distinct SP-shards the band's `BC_KTILE` tiles land in (min/max shard,
  ≤2 adjacent under `KC ≤ cl_t`) and for each **non-local** shard call
  `Semaphore<>(sem_id[dir_c]).wait_min(val_c)`. **Guard the gate:** run it only for real bands
  (`band < num_bands`) and only in the sender/no-mcast K-read path — a phantom band (`stream_heads`) or the
  receiver-role mcast path must not wait, or it deadlocks.
  - ⚠️ **Wait on ALL distinct shards a band touches, NOT `b%sp`.** Under the oracle, `KC = 16/8` tiles while
    `cl_t = 20` tiles, so a band routinely straddles two adjacent shards; `compute:506-516` matmuls all `KC`
    columns unconditionally, so single-shard gating reads un-arrived tiles → garbage on visible columns →
    fails the exact `-inf`-map + PCC≥0.999 oracle. Waiting the farthest shard per direction covers all nearer
    ones (monotone counters), so ≤2 `wait_min` calls per band.
- **`reader_indexer_score.cpp`** — local-shard sourcing (staged, §4.2).
- **`compute_indexer_score.cpp`** — **NO CHANGE.** Preserve the diagonal L1-accumulate mask ordering
  (`stamp_mask_tile`) so a scored diagonal tile is masked before its column is written.
- **`writer_indexer_score.cpp`** — **NO CHANGE.**
- **Host validation (new):** `k_chunk_size ≤ block_cyclic_chunk_local` (guarantees ≤2 adjacent shards/band).
- **Compile gate:** all fused-ring code behind a `fused_ring` compile-time flag so the standalone indexer
  binary stays byte-identical.
- **Optional perf layer (Step E):** a host-provided per-core band-visit permutation (local bands first, then
  remote in ring-arrival order) fed **identically** to reader+compute+writer via the shared
  `WorkUnitSpan (group,band)` key; only inter-band ordering changes, never band identity. Requires guarding
  the `stream_heads` phantom-band q-mcast rendezvous count.

### Op interface
Takes the **SP-sharded LOCAL K cache** `[B,1,sll,D]` that `write_k`/`update_padded_kv_cache` leaves resident
(NOT a pre-gathered full-T tensor), plus all existing indexer attrs (`chunk_start_idx`, `cluster_axis`,
`cache_batch_idx`, `block_cyclic_sp_axis`, `block_cyclic_chunk_local`, `program_config`, `apply_relu`,
`num_groups`, `block_size`, …) AND the AG params (`num_links`, `topology=Linear`, ccl semaphore, `dim=2`,
`sub_device_id`, fwd/bwd coords; `sp` derived from the mesh extent on `cluster_axis`). `block_cyclic` is
implied/required when `sp>1`. Internally allocates the `[B,1,T,D]` gathered DRAM scratch, builds the signaler,
co-schedules AG + indexer in one workload. Output is unchanged full `[1,num_groups,Sq,T]` per device (do NOT
narrow to the local slab). In `indexer.py` it replaces the `_gather_index_kbuf` + `indexer_score_dsa` pair
(`indexer.py:576-604`); the downstream TP all-reduce + top-k (`indexer.py:616-628`) stay untouched.
Hashing: `sp/chunk_local/apply_relu/num_groups/block_size/synthesize_gate/gate_scale/cluster_axis` stay
hashed; `chunk_start_idx/kv_len` stay runtime; the `(dir,val)` thresholds depend only on ring
topology+geometry (fixed per program), so per-step `chunk_start` reuses one program.

---

## 6. Realistic performance

Honest and modest.

- ✅ Removes the separate op boundary and the `_gather_index_kbuf → to_memory_config` copy/intermediate
  (real latency + launch-overhead win).
- ✅ Overlaps local + already-arrived remote shards' compute behind fabric transport.
- ❌ Does **not** remove the O(T) `[B,1,T,D]` gathered-buffer DRAM materialization — the AG still writes the
  full buffer — and does **not** speed the scattered `BC_KTILE` DRAM reads. Fusion only starts scoring
  earlier; if scoring is DRAM-bandwidth-bound, the compute itself is not faster.
- ❌ **Serial tail:** under `Topology.Linear` (non-torus 1×4) the farthest shard reaches an edge device only
  after `sp-1 = 3` store-and-forward hops, and the device's full `[1,1,Sq,T]` row must complete before the
  downstream top-k. So the last-shard transport + its compute is exposed, plus the untouched TP all-reduce +
  top-k (`indexer.py:616-628`) remain serial.
- **Without** the Step-E arrival-order reorder, overlap is poor (head-of-line blocking: reader keeps natural
  band order, slabs arrive in distance order). **With** the reorder and IF compute-bound, expect to hide
  roughly `(sp-1)/sp` of AG latency minus the last-hop tail; if DRAM-bound, near-zero extra compute overlap
  and only the op-boundary/copy win.
- **Grid contention:** AG worker cores must be disjoint from the banded compute rectangle
  `(0,0)..(cols_used-1, rows_used-1)` on one Blackhole grid; carving cores shrinks parallel compute and can
  negate the overlap. **Measure, don't assume** (Step E).

---

## 7. Roadmap (each step validated against the oracle)

| Step | What | Validates |
|------|------|-----------|
| **0** | Run current two-op flow unmodified (contiguous + block_cyclic, heads {8,16}) | Golden `-inf`-map equality + PCC≥0.999 the fused op must not break |
| **A** ✅ | **Independent AG-equivalence spike (no fusion):** produce the gathered `[B,1,T,D]` via `ring_attention_all_gather_async` on the 1×4 Linear submesh, feed the **unmodified** `indexer_score`. Oracle both layouts, all 4 ranks. **Test: `test_indexer_score_lb_ring4_ag_equiv.py`.** | **PASSED (2026-07-08).** The single biggest assumption holds: `ring_attention_all_gather_async` reconstructs `k_host` **byte-exactly** (`comp_equal`, not PCC) on all 4 devices for both layouts (A1), and the unmodified `indexer_score_dsa` matches the oracle end-to-end for glm5+dsv32 × both layouts (A2). **Confirmed:** the AG omits the local slice (must be sourced separately — §4.2), needs a sub-device manager + 2 global semaphores + a pre-allocated full-T persistent buffer replicated along the SP axis, `num_links=1`. The port is unblocked. |
| **B** ✅ | Single fused program, coarse barrier + host-seeded local slab: `indexer_score_dsa_fused` co-schedules AG+indexer sharing two direction sems; reader waits for the FULL AG once, reads via the full-T accessor. **Test: `test_indexer_score_lb_ring4_fused.py`.** | **PASSED (2026-07-08).** All 4 cases green (glm5+dsv32 × contiguous+block_cyclic), no hang; classic path regression-clean. Validated: descriptor-model fused factory (added as a 2nd variant, classic factory untouched), `RingSDPAFusedOpSignaler`/`AllGatherFusedOpSignaler` wiring, 2 direction sems on the compute grid, 1×4 Linear per-device fwd/bwd thresholds + `+1` pre-signal, AG workers carved onto a reserved grid row (`ccl_core_grid_offset`), reader `FUSED_RING` coarse barrier. **Deferred to C/D as planned:** device-side local sourcing (Step B host-seeds the local slab); no overlap win yet (coarse barrier). |
| **C** ✅ | Per-band per-shard gated read: replaced the coarse barrier with a per-band `wait_min` on the distinct SP-shards each band's tiles land in. **Reader-only change** (runtime-compiled, no C++ rebuild); reuses `test_indexer_score_lb_ring4_fused.py`. | **PASSED (2026-07-08).** All 4 cases green, no hang. The reader REPLAYS `RingIdSequencer` **on-device** (from the fused block's ring_size/index/fwd/bwd) to build the shard→(dir,val) table — no new host args. `shard(L) = BC_KTILE(L)/sll_t`; gate walks the band's tiles, waits each run's non-local shard; gated only on the sender/no-mcast role (receivers get K via mcast). No `KC ≤ cl_t` host guard needed — the run-walk + monotone `wait_min` (harmless re-wait) is correct for any KC. |
| **D** ✅ | Dual-source local shard: reader reads its own shard from a second accessor over the SP-sharded `k_local` (local page `= (BC_KTILE(L) − ring_index·sll_t)·Dt`), remote shards from the gathered buffer; `k_batch_page_offset` preserved on the remote path. Factory passes `k_local` accessor CT args + address (slot 33). **Test seeds the gathered buffer with ZEROS** so a correct score proves device-side local sourcing. | **PASSED (2026-07-08).** All 4 cases green (0 failures), incl. both block_cyclic (the exact-equality hinge). `k_local`'s within-shard tile order matches the gathered band (both = `shard(k_host)`), so the local page math is exact. The Step-B host seed is gone — the op no longer needs the caller to pre-populate the local band. |
| **E** ✅ (reorder; profiling TODO) | Reorder the `(group,band)` walk local-first then remote by ring arrival: a host-computed per-core permutation (`RingIdSequencer` replay → `shard_order`; `band_readiness = max arrival-iter over the band's shards`; `stable_sort`) fed IDENTICALLY to reader/compute/writer via rt slots 34/10/11 (read directly per-iteration, no on-device array). Reader now reads q/w BEFORE the band loop (decouples the q-mcast rendezvous from the fabric gate). `stream_heads` disallowed on the fused path (`TT_FATAL HB==Hi`), so no phantom bands. | **REORDER PASSED (2026-07-08):** all 4 cases green, lockstep holds (no hang/desync), band identity preserved (output unperturbed). Perm is identical within a k-mcast column (device-deterministic `band_readiness`), so k-mcast stays in lockstep. **PROFILED (2026-07-08): net LOSS ~0.73–0.85×** — see §6.1 table + root cause. The AG dominates (742–753µs > score 295–484µs), so the overlap ceiling is ~20%, and core-budget contention (AG on a reserved row + shrunk compute rectangle) negates it. Fusion not worth it at ring-of-4 / this bf16 K as built. |
| **F** | Production dtype + config: bfp8_b K (unexercised by the bf16 oracle) at PCC≥0.999, block-pool (`block_size>0`), `cache_batch_idx` multi-user path; resolve the AG `input_batch_slice_idx` vs `cache_batch_idx` hashing conflict; wire into `indexer.py`. | Production readiness. |

**Strategic call:** Step A is a go/no-go gate requiring zero kernel changes. It either greenlights the port
or saves building on sand.

---

## 8. Top risks

1. **`ring_attention` AG layout equivalence UNVERIFIED.** It is KV-attention-shaped (multi-input,
   `tile_id_start = ring_id·Ht·Wt` for `dim!=3`); that it produces byte-identical dim=2 concat-by-`ring_id` to
   `_sp_all_gather` for a `[B,1,T,D]` block-cyclic input is asserted, not proven. → Step A.
2. **Local-shard sourcing under block-cyclic.** The AG omits the local slab; every core repeatedly needs it
   (period `sp·cl_t`). The dual-source offset assumes `write_k`/`update_padded_kv_cache` leaves the local
   slab in the within-shard order `BC_KTILE` expects; an off-by-shard silently mis-places LOCAL columns and
   fails **exact** `-inf` equality (not just PCC). → prologue-copy first (B/C), dual-source later (D).
3. **Band-straddle correctness.** `KC` (16/8) doesn't divide `cl_t` (20) in the oracle → most bands touch 2
   adjacent shards; single-shard gating is a correctness bug. Enforce `KC ≤ cl_t`, wait-on-all-distinct-shards.
   Respect padded/partial-tail bands and block-pool (`Tt % KC == 0`) divisibility.
4. **1×4 Linear edge-device thresholds + forward `+1` pre-signal.** `forward/backward_writes_expected` are
   asymmetric (devices 0 and 3 one-directional); a mismatch between the host replay seed and the AG's actual
   per-device increment pattern hangs. First-bring-up hang suspect; surfaced by Step B on all 4 ranks.
5. **Grid contention.** AG worker cores must be disjoint from the banded compute rectangle; carving cores can
   negate the overlap. Measure (Step E).
6. **Overlap overstatement / head-of-line blocking.** Real overlap needs the Step-E reorder, which then risks
   the `stream_heads` phantom-band q-mcast rendezvous hang; the bf16 oracle uses `head_group_size=0` and never
   exercises it.
7. **bfp8_b K (production) + block-pool + `cache_batch_idx` multi-user paths unverified** by the bf16 dense
   ring4 oracle. → Step F.

---

## 9. Open questions (resolve by reading more code or measuring before committing)

1. Does `ring_attention_all_gather_async_multi_core_with_workers_helper` produce a byte-identical gathered
   `[B,1,T,D]` (concat dim=2 by `ring_id`) to `_sp_all_gather`/`all_gather_async` for a single-input
   block-cyclic-sharded K on 1×4 Linear? (Step A — read `writer.cpp` `gather_dim` math and measure.)
2. What exact per-device `forward_writes_expected`/`backward_writes_expected` does the `ring_attention` AG
   compute for the 1×4 Linear submesh (edge devices 0/3), and does the forward writer emit the local `+1`
   pre-signal on every device? (Read `ring_attention_all_gather_writer.cpp ~328-340`; must match the host
   `RingIdSequencer` replay seed.)
3. Does `write_k`/`update_padded_kv_cache` leave the SP-sharded local cache in the within-shard order
   `BC_KTILE(L) - ring_index·sll_t` assumes, including trailing pad rows beyond `sll`? (The oracle uses an
   unpadded `_to_slab` tensor, so this is untested.)
4. How is `cache_batch_idx` (multi-user KV cache, hash-excluded today so one program serves all users)
   reconciled with the AG helper's build-time `input_batch_slice_idx`? Force it into the hash (regresses cache
   reuse) or gather all users (blows up the DRAM scratch)?
5. Does `gather_valid_Ht`/`kv_len<T` truncation ever leave a shard undelivered while a band's wait table still
   counts it (deadlock)? The wait table is derived from full-T geometry.
6. Can the AG worker cores be placed disjoint from the banded compute rectangle on the target Blackhole grid
   without shrinking `cols_used/rows_used` enough to erase the overlap? (Measure core budget.)
7. Is `head_group_size=0` (no `stream_heads`) safe for all production configs, or must the Step-E reorder
   handle the phantom-band q-mcast rendezvous?
8. Does `read_k_chunk_streaming`'s dropped `k_batch_page_offset` need fixing for the indexed-cache production
   path, or will the fused reader always use `read_k_chunk`?

---

## 10. Key file map

**Ring-joint-SDPA fusion infra (the pattern being ported):**
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_fusion.{hpp,cpp}` — `RingSDPAFusedOpSignaler`, `RingSDPAOpReceiver`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_id_sequencer.hpp` — ring iteration order
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/{ring_joint_reader,ring_joint_writer}.cpp`, `fused_op_receiver.hpp`

**All-gather producer (signals slab-ready):**
- `ttnn/cpp/ttnn/operations/ccl/ccl_op_fusion.{hpp,cpp}` — `AllGatherFusedOpSignaler`
- `ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_multi_core_with_workers_program_factory.{hpp,cpp}` — **the only Linear+fuse-capable AG**
- `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp:317` — hard-FATALs on Linear+fuse

**`indexer_score` op (the target):**
- `.../indexer_score/device/indexer_score_device_operation_types.hpp` — attrs, `BlockCyclicLayout` permutation `P`
- `.../indexer_score/device/indexer_score_program_factory.cpp`, `.../device/kernels/indexer_score_work_split.hpp`
- `.../indexer_score/device/kernels/{compute_indexer_score,reader_indexer_score,writer_indexer_score}.cpp`
- `.../indexer_score/indexer_score.hpp` — ttnn entry

**Model usage + oracle:**
- `models/demos/deepseek_v3_d_p/tt/mla/indexer.py` — `_gather_index_kbuf`, `_sp_all_gather`, `indexer_score_dsa`, PERF TODO
- `tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score_lb_ring4.py` — the oracle
