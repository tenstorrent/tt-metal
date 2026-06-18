# DeepSeek-V3.2 Indexer — Change B (TP-shard heads) report

Date: 2026-06-18 · Hardware: Blackhole p150 (8×), grid 11×10 = 110 cores · Branch: `mvasilijevic/dsa_w_ops`

## What was done

1. **mcast INFO logging in the indexer op.** `indexer_score_program_factory.cpp` now emits an
   `info`-level line whenever a program is built, showing whether the grid-aligned multicast is
   applied to **q/w** (rows) and **k** (columns) or whether it fell back to per-core DRAM reads:
   ```
   indexer_score: q/w mcast OFF, k mcast OFF (grid_aligned=false, cores=110)
   ```
   It prints on the command line during the perf test (program-cache miss, i.e. first call).

2. **Model change B — TP-shard the indexer heads + all-reduce the partial logits**
   (`mla.py::_indexer_topk`). Previously the indexer ran **replicated at all 64 heads**, forcing
   the head-streamed `HB=16` config (see backlog 21). B distributes the head-sum across TP:
   - `_idx_wq_b` is now **column-parallel** (shard its `H_idx·D_idx` output across `tp_axis`), so each
     chip builds `H_idx/tp` indexer heads; `qr` is replicated so no reduce is needed.
   - `weights_proj` is **reduce-scattered on the head dim** (`_tp_rs_ag(rs_only=True)`) so each chip
     keeps the reduced `H_idx/tp` head-weights matching its q heads.
   - each chip runs `indexer_score` over its `H_idx/tp` heads → a **partial logit** (the head-sum is
     separable; `relu` is per-head and the `−inf` causal mask is head-independent, so it survives).
   - the partials are **all-reduced (SUM) over `tp_axis`** (`_tp_rs_ag` = reduce_scatter + all_gather)
     before top-k. The op emits ROW_MAJOR; the CCL reduce runs in TILE, so the logits are
     round-tripped RM→TILE→RM around the all-reduce.
   - **HB=0** (all `H_idx/tp` heads resident) is now used for `tp>1` (≤32 heads/chip fits L1, probed);
     `tp=1` keeps `HB=16` (all 64 heads on one chip).

   The replicated default (`ops.INDEXER_FULL_MODEL_CONFIG`, `QC=2/KC=8/HB=16`) is unchanged for
   callers that don't TP-shard; `ops.indexer_program_config(skv, head_group)` gained the `head_group` arg.

## What was found

### Indexer math utilization — target ~70%, achieved 75.2%

Measured via the perf harness (`test_mla_perf.py::test_mla_chunked_perf`, **sp4×tp2**, 5120-token
chunk @ 51200-token cache) using the op's own performance model (`PM IDEAL` / actual device kernel):

| | A (replicated 64h, HB=16) | B (TP-shard 32h/chip, HB=0) |
|---|---|---|
| indexer device time | **802.5 ms** | **9.86 ms** |
| indexer math util | **1.85 %** | **75.2 %** (PM-ideal 7.42 ms / 9.86 ms) |
| indexer share of layer | 95.7 % | 15.6 % |
| DSA layer critical path | **839 ms** | **≈ 63 ms** |

B is an **81× faster indexer** and a **~13× faster DSA layer**. The util jump (1.85 → 75.2 %) is the
whole point: at 32 heads/chip with `HB=0` all heads stay L1-resident, so the matmul is no longer
drowned in head-streaming q re-reads.

### The all-reduce after the indexer (the cost B adds)

The price of B is the partial-logit all-reduce. Per-op breakdown of that block (medians across the 8
chips), compared to the indexer:

| op | time | vs indexer |
|---|---|---|
| indexer_score | 9.86 ms | 1.00× |
| Tilize (RM→TILE) | 2.90 ms | |
| ReduceScatter | 6.80 ms | |
| AllGather | 6.65 ms | |
| Untilize (TILE→RM) | 3.12 ms | |
| **CCL only (RS+AG)** | **13.45 ms** | **1.36×** |
| **full reduce step** (incl. layout round-trip) | **19.47 ms** | **1.97×** |

**The all-reduce CCL alone (13.5 ms) is longer than the indexer compute (9.86 ms), and the full
reduce step is ~2× the indexer.** ~6 ms of that is pure Tilize/Untilize overhead, forced because
`indexer_score` emits ROW_MAJOR but the CCL reduce runs in TILE.

### Multicast status

At sp4×tp2 the indexer logged **`q/w mcast OFF, k mcast OFF (grid_aligned=false, cores=110)`** — the
dense work-unit deal did not land grid-aligned on the 110-core grid, so both inputs fall back to
per-core DRAM reads. The op still reaches 75 % util via the fallback path; grid-aligned multicast is
a further (unrealized) speedup opportunity.

### New bottleneck

With the indexer no longer dominating, **`topk_large_indices` is now the single largest op at
15.0 ms** (24 % of the layer), followed by the logits all-reduce and `sparse_sdpa` (8.7 ms).

## Validation

- **dev suite** (`-m dev`): **33 passed, 1 skipped** (expected: seq256/SP=8 below the kvpe ND-shard
  minimum). Includes `test_indexer_chunked_matches_single_shot[sp1xtp8]` — B's TP-shard + all-reduce
  at TP=8, chunked-vs-single-shot indexer selection matches (self-consistency only, no reference).
- **gate suite** (`-m gate`): first run **42 passed, 6 failed** — the 6 failures (all
  `test_indexer_device_vs_reference` at tp>1) were a **test-instrumentation artifact**, diagnosed and
  fixed below; after the fix all 6 pass (PCC 0.959–0.992, topk overlap ≥0.985). Everything else passed
  throughout: MLA-output-vs-reference at all meshes, seq4k DSA, chunked, determinism, and the same
  indexer test at tp=1.

### Initial gate run: 6 failures — diagnosed as a TEST artifact, B is correct

The first gate run failed `test_indexer_device_vs_reference` at tp>1 with PCC that *degraded with TP*
(tp1 0.96 / tp2 ~0.83 / tp4 ~0.62). Investigation:

1. **fp32 all-reduce flag → no effect (bit-identical PCC).** `reduce_scatter` already accumulates in
   fp32 internally, so the cross-device sum was never lossy. Precision hypothesis **refuted**.
2. **Op decomposition is exact.** Single-device check: summing 4×16-head partials vs the full 64-head
   op → **PCC 0.9999**. The head-sum split + the op at fewer heads are correct.
3. **Root cause: the test captured the wrong tensor.** It monkeypatched `ops.indexer_logits` and
   compared *its* output to the 64-head reference. Under A that return value *was* the full logits;
   under B it's the per-chip **partial** (`H_idx/tp` heads) — the head-sum is finished by the
   all-reduce *inside* `_indexer_topk`, **after** that call. So the captured tensor had only `1/tp` of
   the heads, which is exactly why PCC tracked `64 / (64/tp)`. (Top-k overlap and MLA-output, which use
   the full post-all-reduce logits, passed throughout.)

**Fix:** repoint the test capture to `top-k`'s input (the full post-all-reduce logits). **All 6 now
pass** at the tp=1 baseline:

| mesh | TP | heads/chip | indexer-vs-ref PCC (L0/L30/L60) | topk overlap |
|---|---|---|---|---|
| sp8×tp1 | 1 | 64 | 0.960 / 0.987 / 0.992 | ✅ |
| sp4×tp2 | 2 | 32 | 0.959 / 0.992 / 0.986 | 0.985–0.991 ✅ |
| sp2×tp4 | 4 | 16 | 0.959 / 0.992 / 0.987 | 0.985–0.991 ✅ |

So **B is numerically correct** — the full head-summed logits match the reference as well as tp=1.

An `INDEXER_LOGITS_ALLREDUCE_FP32` flag was added to test the precision hypothesis; since it was a
**confirmed no-op** (`reduce_scatter` already accumulates in fp32), it has been **removed**.

## Conclusions / next steps

1. **Perf goal met:** B brings the indexer to **75.2 % util** (802 → 9.86 ms; DSA layer 839 → ~63 ms).
2. **Correctness confirmed:** full gate green after the test-capture fix; dev green. B was always
   computing the right logits — the apparent regression was the test reading a pre-all-reduce partial.
3. **The logits all-reduce is the dominant indexer-block cost** (~19.5 ms ≈ 2× the indexer; ~6 ms is
   the RM↔TILE round-trip). It runs at **8.4 % of the 512 GB/s link** (42.9 GB/s) — overhead/impl-bound,
   not bandwidth-bound (12× headroom). Levers: an RM-native all-reduce (drop the ~6 ms tilize/untilize),
   or keep queries SP-sharded so the all-reduced tensor is `1/sp` the size.
4. **topk_large_indices (15 ms)** is the next op-level target now that the indexer is fast.
5. **Multicast is OFF (DRAM fallback)** at this geometry (`q/w OFF, k OFF, grid_aligned=false`) — the
   dense deal doesn't land grid-aligned on 110 cores. Flagged as a follow-up investigation (expected ON).
