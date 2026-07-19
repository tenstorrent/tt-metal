# QB4 ring-fused `indexer_score` — perf findings

Measurements from `test_ring_indexer_score_dsa_qb4_perf.py` on a **4-device Blackhole QuietBox**
(`FABRIC_1D_RING` + `Topology.Ring`, sp = 4). Compares the **fused** `ring_indexer_score_dsa` (SP
all-gather co-scheduled with the score) against the **separate** two-op baseline (standalone
`ring_attention_all_gather_async` → standalone `indexer_score_dsa`), and measures how much of the
all-gather is hidden behind scoring.

## Shape (deployed)

The deployed per-device indexer shape is the **TP=1 / SP=32 resharded "short" grid-fill** config
(`glm5_tp1` / `dsv32_tp1` in `test_indexer_score.py`):

| | GLM-5.1 | DeepSeek-V3.2 |
|---|---|---|
| index heads (resident, TP=1) | **32** | **64** |
| head_dim | 128 | 128 |
| Sq / device | 160 | 160 |
| q_chunk | 32 (QC=1 → 5 q-groups → grid-fill num_blocks=2) | 32 |
| k_chunk (prod, `short_config`) | 256 (KC=8) | 128 (KC=4) |
| T (all-gathered keys) | 56320 (1760 tiles), history 55680 + chunk 640 | same |
| chunk_local (`cl_t`) | 160 (5 tiles) | 160 (5 tiles) |

`head_group_size = 0` (all heads resident). At this shape the standalone indexer reproduces the
deployed device kernel duration and matmul utilization (below).

## Method

- **`math_util`** is read **from the Tracy report** (`--profile` → `ops_perf_results*.csv`), computed the
  same way as the official gate `test_indexer_score_math_util`:
  `math_util = mm_flops / (cores × duration_ns·1.35 × 2048_HiFi2)` at the bottleneck ring device.
- **Fused-vs-separate latencies** are trace-loop wall-clock with the **profiler OFF** (kernels compiled
  without `PROFILE_KERNEL` markers → real speed; the in-process `MID_RUN_DUMP` profiler inflates the
  kernel ~4×, so it is used only for the per-RISC breakdown, never for absolute µs).

## Grid parity (why a config knob was added)

The QB compute grid is 11×10 (110 cores). The fused op reserves **one column** for the AG workers →
100-core compute rectangle. To compare apples-to-apples, the standalone indexer is pinned to the same
100 cores via a new **benchmarking-only** field on `IndexerScoreProgramConfig`:

```python
IndexerScoreProgramConfig(..., max_core_grid_x=10)   # 0 = full device grid (default, no behavior change)
```

The classic factory clamps `grid_x`/`grid_y` to these caps; the fused factory ignores them. Default 0 →
byte-identical to before; hashed via struct reflection so capped/uncapped programs never collide.

## Results

### `math_util` from Tracy (full 110-core grid — deployed reference)

| model | Tracy `DEVICE KERNEL DURATION` | cores | **math_util** | official gate |
|---|--:|--:|--:|--:|
| GLM-5.1 (32h, KC8) | **0.324 ms** | 110 | **74.8 %** | 0.345 ms / 70.1 % |
| DeepSeek-V3.2 (64h, KC4) | **0.630 ms** | 110 | **76.9 %** | 0.636 ms / 76.1 % |

The QB4 standalone indexer matches the deployed per-device latency and utilization.

### Fused vs separate — `num_links = 2`, 10×10 grids (µs, profiler-off trace)

Two columns: **before** = the original striped `i % cols_used` band→column assignment; **after** = the
readiness-balanced assignment (commit *readiness-balanced band→column assignment*).

**GLM-5.1 (32 heads)**

| k_chunk | AG | indexer | **FUSED** before | hidden | **FUSED** after | hidden | speedup |
|---|--:|--:|--:|--:|--:|--:|--:|
| 256 (prod, KC8) | 215 | 355 | 552 | 8 % | **457** | **53 %** | 1.25× |
| **160 (aligned, KC5)** | 215 | 365 | 464 | 55 % | **396** | **86 %** | **1.47×** |

**DeepSeek-V3.2 (64 heads)**

| k_chunk | AG | indexer | **FUSED** before | hidden | **FUSED** after | hidden | speedup |
|---|--:|--:|--:|--:|--:|--:|--:|
| 128 (prod, KC4) | 215 | 694 | 883 | 12 % | **785** | **58 %** | 1.16× |
| **160 (aligned, KC5)** | 215 | 709 | 799 | 58 % | **719** | **95 %** | **1.29×** |

## Findings

1. **Readiness-balanced band→column assignment is the dominant AG-overlap lever.** Per-core Tracy shows the
   co-scheduled all-gather finishes on time (~220 µs, == standalone) and the compute starts immediately, yet
   before the fix one column stalled ~95 µs on a remote shard at its *very first band*. Cause: the striped
   `i % cols_used` assignment correlated SP shard with column — a block-cyclic band maps to shard
   `(first_tile / cl_t) % ring` and to column `band % cols_used`, so at 10 cols × ring 4 (gcd 2) each column
   saw only **2 of 4 shards** and *half the columns held no local band*. Those columns could not front-load
   local work; they gated on a remote shard immediately and exposed the entire first-slab arrival. Assigning
   bands round-robin *within each ring-arrival readiness level* balances local (readiness-0) work across all
   columns → **hidden 55→86 % (GLM), 58→95 % (DeepSeek)** at the aligned k_chunk; production k_chunk also
   jumps (8→53 %, 12→58 %).

2. **The fusion still pays off most with a fusion-aligned `k_chunk`.** `KC | cl_t` (cl_t = 5) keeps a band
   inside one SP shard so its readiness is a single arrival level. `k_chunk = 160` (KC 5) hides **86–95 %**
   vs **53–58 %** at the misaligned production k_chunk. **Recommendation: `k_chunk = 160` on the fused path.**

3. **Residual exposure is the whole-slab gating floor, set by the first-shard arrival (~120 µs at nl2).**
   The AG signals the fused reader once *per whole SP slab*; the compute cannot start a remote shard until
   its slab has fully arrived. At nl2 the nearest (1-hop) slab arrives at ~120 µs (the AG splits bandwidth
   across both ring directions, so a 3.6 MB slab lands at ~½ the 220 µs gather, not its ~36 µs
   bandwidth-only time). Per-device local work is C/4 = **91 µs (GLM) / 177 µs (DeepSeek)**:
   - **DeepSeek** (177 > 120) can fully cover the arrival → floor ≈ 0; measured residual 10 µs is minor
     load imbalance (95 % hidden).
   - **GLM** (91 < 120) runs out of local work 29 µs before the first slab → **~29 µs floor**; measured
     30 µs (86 % hidden) is essentially at the floor.
   Beating the GLM floor needs the AG to deliver the nearest shard *sooner* — sub-slab (finer-grained)
   signaling or delivering the 1-hop slab at full bandwidth first — a change to the shared ring-attention
   all-gather, not the indexer. `num_links` > 2 also shrinks the first-slab arrival (see the sweep below),
   but the deployed setting is fixed at nl2.

   **`num_links` sweep (aligned k_chunk, readiness-balanced, profiler-off trace):**

   | | nl1 (AG 413) | **nl2 (AG 215)** | nl3 (AG 170) | nl4 (AG 139) |
   |---|--:|--:|--:|--:|
   | GLM exposed / hidden | 169 µs / 59 % | **31 µs / 86 %** | 9 µs / 95 % | 8 µs / 94 % |
   | DeepSeek exposed / hidden | 48 µs / 89 % | **10 µs / 95 %** | 12 µs / 93 % | 12 µs / 92 % |

   GLM's exposure tracks the AG size (first-slab arrival) straight down to the ~8–10 µs floor at nl3+;
   DeepSeek is already at that floor at nl2 (its exposure is nl-independent load imbalance, not AG arrival).
   The **~8–10 µs floor** common to both is the residual band-load imbalance across columns plus the last
   whole-slab tail — the practical limit of the whole-slab-gated schedule.

4. **The standalone indexer is compute-bound at ~75–77 % matmul util** (both Q across rows and K down
   columns are multicast; the reader has slack). **AG-hiding grows with the compute/AG ratio** — DeepSeek's
   heavier 64-head compute buries more of the same-size AG than GLM.

## Files

- `test_ring_indexer_score_dsa_qb4_perf.py` — the harness:
  - `test_qb4_ring_vs_separate_true_latency` — fused vs separate, real µs (profiler off).
  - `test_qb4_indexer_true_latency` — standalone true latency + analytical FPU util.
  - `test_qb4_tracy_perf` — plain-dispatch target for `--profile`; `math_util` read from the Tracy CSV.
  - `test_qb4_indexer_bottleneck_sweep` — per-RISC breakdown + k_chunk sweep.
  - `test_qb4_ring_indexer_vs_separate` — in-process-profiler variant (per-program durations; inflated).
- `IndexerScoreProgramConfig.max_core_grid_x/_y` — the grid-cap knob (classic factory only).
