# Ring Joint SDPA — C1 at **d_v=512**, full 4-way matrix (latent/non-latent × DM on/off)

Branch: `skrstic/ring_joint_sdpa_optional_latent_v_fix`
Date: 2026-06-03
Test: `test_ring_joint_attention_create_chunked_perf_table`, **50K+5K last chunk**, perf-only.
Hardware: P150_X8, 110 SDPA cores.

**Config C1:** sp8 · per-device seq 640 · q_chunk 32 · chunk_size 5120 · **d_v = 512** (wide value-latent).
Latent V = V rematerialized on-device from K's first 512 cols. Non-latent V = a separate **512-wide** V
tensor with its own ring all-gather. DM-off = bulk NoC primitives physically commented out of the
kernels (no macros). OOM-skip enabled. Cells = **Duration ms (Math Util %)**.

Raw data: `sweep_runs/results_C1_dv512.tsv`.

## Results

| k_chunk | Latent DM-on | Latent DM-off | NonLatent DM-on | NonLatent DM-off | DM overhead (lat) | DM overhead (non-lat) | Latent speedup (DM-on) |
|--------:|:------------:|:-------------:|:---------------:|:----------------:|:-----------------:|:---------------------:|:----------------------:|
| 256 | 6.951 (56.7%) | 6.395 (61.6%) | 19.398 (20.3%) | 14.710 (26.8%) | 8.0% | 24.2% | **64.2%** |
| 384 | 7.020 (56.1%) | 6.037 (65.2%) | 19.278 (20.4%) | 14.663 (26.9%) | 14.0% | 23.9% | **63.6%** |
| 512 | 6.777 (58.1%) | 5.794 (68.0%) | 18.848 (20.9%) | 14.650 (26.9%) | 14.5% | 22.3% | **64.0%** |
| 640 | 6.536 (60.3%) | 5.760 (68.4%) | OOM | OOM | 11.9% | — | — |
| 768 | OOM | OOM | OOM | OOM | — | — | — |

Best latent DM-on: **k640 = 6.536 ms (60.3%)**. Best non-latent DM-on: k512 = 18.848 ms (20.9%).

## Conclusions

### 1. At d_v=512 the latent path is ~2.8–2.9× faster than separate-V
Latent ≈ 6.5–7.0 ms at ~57–60% util; non-latent ≈ 18.8–19.4 ms at only ~20% util — a **~64% latency
reduction** from latent. This is the same mechanism as d_v=128 but hugely amplified: separate-V must
all-gather a **512-wide** V tensor across the 8-device ring *on top of* the 576-wide K gather, whereas
latent rematerializes those 512 cols locally from the already-present K for free. Quadrupling the V
width quadruples the cost of the thing latent avoids, so the gap explodes (it was ~30% at d_v=128/q32,
now ~64%).

### 2. Non-latent is catastrophically DM/sync-bound — even its compute ceiling stays slow
Non-latent util is stuck at ~20% (DM-on) and only ~27% (DM-off): the device spends most of its time on
the V all-gather, not matmul. Crucially, **DM-off only recovers ~24%** (19.4 → 14.7 ms) — far less than
the 2.8× gap — because commenting out the NoC *payload* leaves the separate V gather's
**ring-synchronization structure intact** (the per-device chain semaphores are kept by design). With q32
there isn't enough compute to hide that extra round-trip choreography, so even the "compute ceiling" of
non-latent (~14.65 ms) is ~2.5× the latent compute ceiling (~5.79 ms). Latent skips the V gather
entirely — payload *and* sync — so it's both faster and near compute-bound (~68% util at DM-off).

### 3. Latent also has more L1 headroom
Latent reaches **k640** (6.536 ms) before OOM; non-latent OOMs at **k640** (max k512) because the
separate 512-wide V tensor + its all-gather buffer consume extra L1. So latent is faster *and* tiles to
a larger, higher-util k_chunk.

### 4. DM overhead, latent: modest and growing with k
Latent DM overhead is 8% (k256) → ~14% (k384/512) → 12% (k640): with V free, the residual data movement
is the K stream + ring handoff, a small and well-overlapped fraction (util ~57–60% on, ~65–68% off).

**Bottom line:** widening V to 512 makes the latent-vs-separate choice decisive — separate-V is a
non-starter for prefill at d_v=512 (~3× slower, OOMs earlier). If d_v=512 is required, **use latent V**;
the best C1 point is k640 latent (6.536 ms, 60.3% util). (Recall from the d_v sweep that d_v=512 latent
is itself ~17% slower than d_v=128 latent for C1 — 6.536 vs 5.829 ms — so d_v=128 remains preferable
for prefill where the model allows it.)

## Reproduce
```
bash sweep_driver_C1_dv512.sh    # -> sweep_runs/results_C1_dv512.tsv  (latent/non-latent × DM on/off, OOM-skip)
```
TEMP scaffolding — revert before merge.
