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

**GLM-5.1 (32 heads)**

| k_chunk | AG | indexer | separate (AG+idx) | **FUSED** | AG hidden | speedup |
|---|--:|--:|--:|--:|--:|--:|
| 256 (prod, KC8) | 215 | 355 | 570 | 552 | 8 % | 1.03× |
| **160 (aligned, KC5)** | 215 | 366 | 582 | **464** | **55 %** | **1.25×** |

**DeepSeek-V3.2 (64 heads)**

| k_chunk | AG | indexer | separate (AG+idx) | **FUSED** | AG hidden | speedup |
|---|--:|--:|--:|--:|--:|--:|
| 128 (prod, KC4) | 215 | 694 | 909 | 883 | 12 % | 1.03× |
| **160 (aligned, KC5)** | 215 | 709 | 924 | **799** | **58 %** | **1.16×** |

## Findings

1. **The fusion pays off only with a fusion-aligned `k_chunk`.** The overlap requires `KC | cl_t`
   (cl_t = 5 tiles here). Neither production k_chunk divides 5 (KC 8 / 4), so bands straddle SP-shard
   boundaries and back-load the ring-arrival tail → **8–12 % hidden, ~1.03×**. Setting `k_chunk = 160`
   (KC=5, the only divisor of 5 that is L1-safe) hides **55–58 %** → **1.16–1.25×**, for a ~2–3 %
   standalone slowdown. **Recommendation: both GLM and DeepSeek should use `k_chunk = 160` on the fused
   path.**

2. **The AG scales with T; `num_links = 2` is the right setting.** At the deployed T = 56320 each device
   gathers ~10.8 MB; AG ≈ 410 µs at nl1 → **≈ 215 µs at nl2** (~2× fabric BW). The smaller AG is much
   easier to bury under the compute — this test fixes `num_links = 2`.

3. **The standalone indexer is compute-bound at ~75–77 % matmul util.** It is *not* reader/DRAM-bound:
   both Q (across rows) and K (down columns) are multicast (verified `k_mcast_on = q_mcast_on = 1`), so
   inputs are read once and reused; the reader thread has slack. The residual to 100 % is the SFPU
   (`relu` + per-head weighted sum) sharing the math pipeline and the output-matrix write, not a memory
   bottleneck.

4. **AG-hiding grows with the compute/AG ratio.** DeepSeek (heavier 64-head compute) hides more of the
   AG than GLM at the same AG size.

## Files

- `test_ring_indexer_score_dsa_qb4_perf.py` — the harness:
  - `test_qb4_ring_vs_separate_true_latency` — fused vs separate, real µs (profiler off).
  - `test_qb4_indexer_true_latency` — standalone true latency + analytical FPU util.
  - `test_qb4_tracy_perf` — plain-dispatch target for `--profile`; `math_util` read from the Tracy CSV.
  - `test_qb4_indexer_bottleneck_sweep` — per-RISC breakdown + k_chunk sweep.
  - `test_qb4_ring_indexer_vs_separate` — in-process-profiler variant (per-program durations; inflated).
- `IndexerScoreProgramConfig.max_core_grid_x/_y` — the grid-cap knob (classic factory only).
