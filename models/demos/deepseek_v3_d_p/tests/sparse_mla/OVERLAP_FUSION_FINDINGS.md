# DeepSeek V3.2 Sparse MLA (DSA) — Overlap/Fusion Optimization Findings (Blackhole Galaxy)

Investigation into overlapping / fusing / restructuring the sparse MLA (DSA) chunked-prefill forward on
Blackhole **Galaxy, SP=8 × TP=4 (32 chips)**. Combines (a) measured per-op Tracy data across warm/cold/long,
(b) source-level verification of async CCL + fusion machinery, (c) a large multi-agent analysis with
adversarial verification, and (d) **real on-device experiments** validating the top levers.

> `[MEASURED]` = confirmed on the 32-chip BH Galaxy via `test_sparse_mla_perf.py`. Others are
> engineering estimates from verified per-op reasoning. Times are device-collapsed critical path
> (compute = max across chips, collectives = avg).

---

## 0. TL;DR — Final Top 5 (ranked by confidence × measured saving)

| Rank | Optimization | long saving | warm | cold | Status |
|---|---|---|---|---|---|
| **1** | **A2** — RS-only + distributed per-shard top-k (eliminate op68 AG, shrink top-k 4×) | **~27 ms probe / ~25.5 ms real** | ~2.5 ms | ~26 ms | **[MEASURED probe −27 ms]**; large effort (merge/values/sentinel) |
| **2** | **B2** — native AllGather (TILE) for KVPE gather, not composite AllBroadcast+Concat | **−10.7 ms** | **−1.2 ms** | **−12.8 ms** | **[MEASURED] ✅ ship now — 2-line patch** |
| **3** | **A1** — reshard indexer by QUERY-ROWS → delete the *whole* full-T logit all-reduce | ~35 ms | ~3 ms | ~33 ms | est.; supersedes A2 (also removes RS+tilize); medium effort + causal-geometry op change |
| **4** | **B1** — sub-device overlap: KVPE gather ∥ indexer branch *(user hypothesis)* | ~16 ms (~10 ms after B2) | ~1.8 ms | ~19 ms | est., large effort; composes with B2 |
| **5** | **fabric** — enable `num_links` 2→4 on BH Galaxy | ~15–18 ms | ~1.5 ms | ~17 ms | **[MEASURED linear scaling]**; blocked on BH fabric (infra, no model code) |

**Headline [MEASURED on the real 32-chip Galaxy — A2 + B2 stacked, both families]:**

| Config | Baseline | + B2 + A2 | Saving |
|---|---|---|---|
| warm | 16.48 ms | **12.57 ms** | −24% |
| cold | 168.7 ms | **134.73 ms** | −20% |
| long | 87.53 ms | **49.57 ms** | **−43%** |

A1 (if the causal-geometry change lands) removes the remaining RS head-sum (11.75 ms) + tilize (3.6 ms) →
long toward **~34 ms**; `num_links`→4 would then ~halve the residual fabric CCLs for another ~9 ms.

Notes: **A1 and A2 are mutually exclusive** (both attack the indexer logit-allreduce block — A1 is the bigger
ceiling, A2 is the measured/derisked path). **B1 and B2 compose** (B2 shrinks the gather; B1 hides the remainder).
**fp8 KVPE cache (B3) is blocked** in current ttnn (§5). Theoretical floor ~24 ms (TP-fabric engine capacity);
binding constraint is the ~55 ms serial indexer branch (§4) — which is exactly why A1/A2 are the top levers.

---

## 1. Measured baseline (Galaxy)

| Config | Cache | Total | Per-chunk |
|---|---|---|---|
| warm | 50k | **16.48 ms** | steady-state step |
| cold | 0→50k, 11 chunks | **168.7 ms** | ~15.3 ms/chunk (flat vs depth) → per-chunk savings ×11 |
| long | 512k | **87.53 ms** | cache-depth-bound ops dominate |

Long-run biggest ops: AllBroadcast (KVPE gather) 22.36 ms · Topk 17.10 · AllGather (5×) 13.29 ·
ReduceScatter (4×) 11.75 · IndexerScore 5.19 · SparseSDPA 4.84 (flat across depth — top-k gated at 2048).

**Core occupancy [MEASURED]:** CCLs are fabric/DRAM-BW-bound, NOT core-bound — AllBroadcast (22 ms) uses
**2/120 cores**, AllGather 18, ReduceScatter 34; compute ops (Topk, IndexerScore, Concat, SparseSDPA) use 120.

---

## 2. Structure: two independent heavy pipelines run serially

```
                ┌───── INDEXER branch (produces top-k indices) — ~55 ms, THE critical path ─────┐
 q-latent ─────▶│ op64 SP-AG idxK → op65 Score → op66 Tilize → op67 RS → op68 AG → op69 Untilize │──┐
 (op22-30)      │  (SP-fab 1.6)      (FPU 5.2)     (DRAM 3.5)   (TP-fab 11.4)(TP-fab 12.4)(DRAM 3.9)│  │
                │                                                          → op70 Topk (SFPU 17.1) │  ▼
                │    ┌──── KVPE-gather branch — ~25.7 ms ────┐                                       op92
                └───▶│ _kv_stem → op88 Copy → op89 AllBroadcast → op90 Concat │──────────────────▶ SparseSDPA
                     │ (0.8)      (0.4)        (SP-fab 22.3)     (DRAM 3.0)   │                    (needs BOTH)
                     └─────────────────────────────────────────────────────┘
```

- Branches are **data-independent** (indexer→indices; KVPE→prefix; join only at SparseSDPA, `mla.py:1195,1199`).
- `forward()` runs them **serially** today (indexer at `mla.py:1034` fully drains before the KVPE path at `:1044`).

---

## 3. How async CCLs work — and what gates overlap

`_async` collectives (`all_gather_async`, `reduce_scatter_minimal_async`, `all_broadcast`) are async only in
the **host-dispatch** sense: the call enqueues a fabric program and returns; device kernels self-synchronize
across chips via on-device global semaphores over the ethernet fabric (`all_gather_async_default_program_factory.cpp:691`).
But every CCL in `mla.py`/`indexer.py` runs on the **default sub-device, single CQ, no `subdevice_id`**, so it
holds the grid's Fast-Dispatch counter → the next op cannot launch until it drains. **⇒ zero overlap today;
a host reorder alone yields ~0.** The only working overlap mechanism is a **`SubDeviceManager` grid split**
(disjoint Tensix strips, one CQ), already proven in this model's MoE (`tt_moe.py:285-306`, `tt_shared_expert.py:452-511`).

---

## 4. Theoretical floor (long)

Per-engine busy time (long): **TP-fabric 24.4 ms** (op67 RS + op68 AG) · **SP-fabric 23.9 ms** (op89 + op64) ·
SFPU 17.1 ms (topk) · DRAM/NoC ~15 ms · matmul-FPU 11.2 ms. Serial sum ≈ 91.6 ms (device-0) ≈ 87.5 collapsed.

- **Absolute floor ≈ 24 ms** (max single-engine busy = TP-fabric) — only reachable with perfect cross-engine overlap.
- **But the DAG forbids it:** the indexer branch is a strict producer→consumer chain that hops SP-fab → FPU →
  DRAM → TP-fab → TP-fab → DRAM → SFPU (~55 ms). Engine-overlap within it buys nothing. **The ~55 ms serial
  indexer branch is the binding critical path** — which is why the biggest levers (A1/A2) attack it.

---

## 5. On-device experiments (this investigation)

| Exp | Change | Result |
|---|---|---|
| **B2** | native AllGather in TILE for KVPE gather (`_gather_kvpe_prefix`) | **[MEASURED] long 87.53→76.85 (−10.7), warm 16.48→15.29 (−1.2), cold 168.7→155.93 (−12.8). ✅** AllBroadcast(22.4)+Concat(3.0) removed; +native AG(10.7)+untilize(3.5)+tilize(0.4). Native AG ≈ **2× faster** than composite broadcast+concat. |
| **A2** | `rs_only=True` probe (`indexer.py:621`) — eliminate op68 AG, top-k per-shard on T/4 | **[MEASURED] long 87.53→60.55 (−27, −31%). ✅** Topk 17.10→**4.32** (~4× T/4 shrink), op68 AG(12.4) eliminated, untilize 4.09→1.07; RS unchanged (11.75). Timing probe (local indices); real A2 adds back ~0.8 ms candidate-AG + ~0.3 ms merge-topk → **~25.5 ms realistic**. |
| **B2 + A2** | both stacked (KVPE native-AG + indexer rs-only) | **[MEASURED] long 87.53→49.57 (−38, −43%). ✅** Confirms the two families are additive. Residual: AllGather 12.3, RS 11.75, IndexerScore 5.19, SparseSDPA 5.13, Untilize 4.55, Topk 4.32, Tilize 4.02. |
| **num_links** | force `ccl_num_links` 2→1 (long, +B2) | **[MEASURED]** AllGather 24.0→46.7 (1.94×), RS 11.7→23.0 (1.96×) → **fabric CCLs scale ~linearly with links**. ⇒ enabling 4 links would ~halve them (~−18 ms). |
| **top-k linearity** | warm vs long topk | **[MEASURED]** 1.888 ms (T=56320) vs 17.10 ms (T=517120) = 9.06×/9.18× → **linear** ⇒ validates A1/A2 top-k shrink ~4× at T/4. |
| **B3 fp8** | KVPE cache `bf16→fp8_e4m3` | **[MEASURED] BLOCKED:** `TT_FATAL: ttnn.copy only supports float, bfloat, int32, uint16 — got FP8_E4M3` (assert.hpp:104). Needs `ttnn.copy` extended to fp8 first. |

### The B2 patch (ready to ship, `models/demos/deepseek_v3_d_p/tt/mla/mla.py`, `_gather_kvpe_prefix` ~1416)
```python
        cache_i = ttnn.to_memory_config(kvpe_cache, ttnn.DRAM_MEMORY_CONFIG)  # ND_SHARDED → INTERLEAVED
        cache_i = ttnn.to_layout(cache_i, ttnn.TILE_LAYOUT)   # native AG path (composite lowering is RM-only)
        full = self._all_gather(cache_i, dim=2, cluster_axis=self.sp_axis)
        full = ttnn.to_layout(full, ttnn.ROW_MAJOR_LAYOUT)    # sparse_sdpa consumes RM
        if self.sp_factor > 1:
            ttnn.deallocate(cache_i)
        return full
```
Root cause: a ROW_MAJOR input forces the composite all-gather lowering (`composite_common.cpp:294-296`) into
`all_broadcast`+`concat`; the identical helper on the TILE index-K cache (op64) stays native. Follow-up: the
remaining +3.5 ms untilize is only to feed sparse_sdpa RM — storing the cache in TILE and/or a TILE-consuming
sparse_sdpa would recover it (larger change).

---

## 6. Top 5 — detail

### #1 — A1: reshard the indexer by QUERY ROWS, not heads (~35 ms long)
**Root cause:** `wq_b` is column-parallel, splitting the 64 index heads 16/chip across TP (`indexer.py:165-171,561-562`),
so each chip produces a *partial-head* logit → the full-T `[640×517120]` logit must be head-summed across TP
(op67 RS + op68 AG) before top-k. **Fix:** replicate `wq_b` and slice the 640 query rows to 160/chip instead;
each chip then head-sums all 64 heads locally (`compute_indexer_score.cpp:491-495`) and runs top-k on 160 rows —
the full-T partial-head all-reduce (op66-69, ~31 ms) **vanishes**, and top-k drops ~4× (op70 17.1→~4.3 ms).
Only a tiny `[640×2048]` index AllGather rejoins the rows. **Gated on** a `device_causal_geometry` change:
`indexer_score_program_factory.cpp:88,119-125` offsets query position by SP coord only; row-slicing corrupts the
causal mask and breaks `Sq==chunk_local`. op65 also drops to a per-column fallback (HB=16) whose regression is
the one unknown — **the experiment to run** is a standalone `indexer_score_dsa` micro-bench sweeping
16-head/640-row (today) vs 64-head/160-row (proposed) + top-k on `[160,517120]`.

### #2 — B2: native AllGather for KVPE gather — **[MEASURED −10.7 ms long]**
See §5. Confirmed, lowest effort, ready patch. Ship first.

### #3 — B1: sub-device overlap of KVPE gather ∥ indexer branch (user hypothesis, ~16 ms)
The two branches are data-independent and use **orthogonal fabric axes**: op89 KVPE gather is SP-axis, op67/op68
logit all-reduce are TP-axis (distinct ethernet planes). Carve a small CCL strip via `SubDeviceManager`
(MoE precedent `tt_moe.py:285-306`) and hoist the KVPE branch to run under the ~55 ms indexer branch.
**Confirmed cheap on cores** (AllBroadcast=2 cores; the "1-column CCL strip" framing is right — moving CCLs off
the compute grid barely dents the 120-core compute ops). **Two real caveats keep it plausible-not-strong:**
(1) op90 Concat has no `subdevice_id` path (`composite_common.cpp:400`) so it stays on the default grid and
exposed — **B2 fixes this by removing Concat entirely, so do B2 first**; (2) `all_broadcast` allocates fresh
outputs with no persistent-buffer hook (`all_broadcast_device_operation.cpp:57-63`) → the period-2 aliasing
hazard `tt_ccl.py:107-113` solved by pinning; needs kept-alive buffers. **Composes with B2:** after B2 the
gather is a ~14 ms native AG that B1 hides behind the indexer. **Experiment:** sub-device microbench under trace —
confirm window wall-clock → max(indexer ~41 ms, kvpe ~26 ms) not the sum, and assert 2-replay bit-exactness.

### #4 — Enable `num_links` 2→4 (fabric, ~15–18 ms) — **[MEASURED linear scaling]**
The fabric CCLs scale ~linearly with links (measured 1↔2). BH Galaxy currently exposes only 2 routing planes
(`tt_ccl.py:337` "possible increase to 4 when it's enabled"; `mla.py:339`). When 4 links land, op67 RS
11.7→~5.9 and op68 AG 12.4→~6.2 and the KVPE native AG (post-B2) ~halves — **~15-18 ms off long for zero model
code**. Blocked on BH fabric enablement (infra). Also worth testing: multi-worker-per-link for the 2-core
AllBroadcast (discriminate worker-bound vs plane-bound).

### #5 — A2: RS-only + distributed per-shard top-k (~24 ms; safer alternative to A1)
Keep the head-sum reduce-scatter but **drop op68's re-broadcast**: `rs_only=True` (flag exists `indexer.py:281`)
leaves each TP chip a fully-summed contiguous `[640×129280]` slab; run local top-2048+values, AllGather only the
`[640×8192]` candidates (~31 MB vs op68's ~662 MB), merge-top-k. **Exact** (global top-2048 ⊆ union of per-shard
top-2048), **validated by the measured top-k linearity**. op68 eliminated (12.4), op69 untilize→~1.0, op70
topk→~4.3+0.3. **Enabler:** `topk_large_indices` must emit VALUES (small independent change). Keeps the
intrinsic head-sum floor (RS+tilize ~14.8 ms) — which is exactly what A1 removes, hence A1 > A2 when feasible.

---

## 7. Rejected / blocked — do not chase

- **fp8 KVPE cache (B3):** [MEASURED] blocked — `ttnn.copy` rejects FP8_E4M3 (assert.hpp:104). Revisit if copy gains fp8.
- **Fused all-gather + indexer_score (ring-joint style):** FEASIBLE on Linear topology (not blocked by the ring
  assert — `all_gather_minimal_matmul_async` proves Linear AG-fusion via topology-agnostic `AllGatherFusedOpSignaler`)
  but **op64 is only 1.55 ms (1.8%)** → ~1.1 ms realistic for a whole new fused op. Low ROI; bundle only as a rider.
- **Full sparse fabric-gather (gather only the 2048 selected rows):** fabric is push-based (no read-by-page-id);
  needs a new distributed sparse collective and trades bulk BW for latency-bound scatter → ~0 or negative. Use B1.
- **TILE-consuming top-k to delete op69 untilize:** `topk_large_indices` TT_FATALs on non-ROW_MAJOR; tile-strided
  reconstruction is worse than the current multi-core untilize.
- **`_build_rope_tables` skip (op11/op12, 2.25 ms):** one-time construction *before* the signpost — off the
  measured critical path (~0 ms).
- **Stem matmul→RS and slice/concat fusions:** all <0.2 ms (cache-depth-independent) — cold-only compounders.

---

## 8. Key files
- `models/demos/deepseek_v3_d_p/tt/mla/mla.py` — forward, `_gather_kvpe_prefix` ~1404, `_all_gather` ~1289, `ccl_num_links` :339.
- `models/demos/deepseek_v3_d_p/tt/mla/indexer.py` — `_tp_rs_ag` ~267 (rs_only :281), score+topk ~590-628, wq_b :165-171,561-562.
- `models/demos/deepseek_v3_d_p/tt/moe/{tt_moe.py,tt_shared_expert.py}` — sub-device overlap precedent.
- `models/demos/deepseek_v3_d_p/tt/tt_ccl.py` — CCL handles, `get_num_links` :313-342 (BHGLX (2,2)).
- `ttnn/cpp/ttnn/operations/experimental/ccl/composite_common.cpp:294-296,390,400` — RM all-gather → broadcast+concat lowering.
- `ttnn/cpp/ttnn/operations/ccl/{all_broadcast,all_gather_async,reduce_scatter_minimal_async}/`, `.../ccl_op_fusion.hpp`, `.../ring_fusion.hpp`.
- `ttnn/cpp/ttnn/operations/experimental/topk_large_indices/` · `.../transformer/sdpa/device/.../sparse_sdpa_gather.hpp`.
- Test: `tests/sparse_mla/test_sparse_mla_perf.py` (`DS_PERF_SCENARIO`=warm|cold|long).
