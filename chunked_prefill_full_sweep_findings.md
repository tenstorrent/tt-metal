# Ring Joint SDPA — Full Extended Chunked-Prefill 50K+5K Sweep (latent d_v=512 vs non-latent d_v=128)

Branch: `skrstic/ring_joint_sdpa_optional_latent_v_fix`
Date: 2026-06-03
Test: `test_ring_joint_attention_create_chunked_perf_table`, **50K+5K last chunk**, perf-only.
Hardware: P150_X8, 110 SDPA cores.

## Matrix (320 cases)

8 configs (C1–C8) × {latent, non-latent} × {DM-on, DM-off} × **10 k-chunks**, with the **realistic d_v
pairing**:
- **latent → d_v = 512** (wide value-latent; V rematerialized on-device from K's first 512 cols).
- **non-latent → d_v = 128** (standard separate V tensor with its own ring all-gather).

k pool (sorted): **224, 256, 384, 448, 480, 512, 640, 672, 768, 800**.
DM-off = bulk NoC primitives physically commented out of the kernels (no macros). Perf-only
(`CHUNKED_SKIP_PCC=1`), last chunk only. **OOM-skip**: once a (dm, latent, config) OOMs at some k, all
larger k are recorded `OOM` without running. Cells = **Duration ms (Math Util %)**. Raw:
`sweep_runs/results_full.tsv` (164 OK, 30 measured-OOM, 126 OOM-skipped).

Configs (per-device seq = chunk_size/sp; nhq=16/ring, d_q=d_k=576, Q bf16, KV bf8):

| | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|--|--|--|--|--|--|--|--|--|
| sp | 8 | 4 | 4 | 8 | 8 | 8 | 8 | 8 |
| per-dev seq | 640 | 1280 | 1248 | 640 | 640 | 640 | 640 | 640 |
| q_chunk | 32 | 64 | 96 | 128 | 64 | 96 | 160 | 192 |
| chunk_size | 5120 | 5120 | 4992 | 5120 | 5120 | 5120 | 5120 | 5120 |

> Note: latent and non-latent here are **different computations** (d_v 512 vs 128 → different output
> width and PV cost), per the requested pairing. "lat speedup" = `(nonlat_on − latent_on)/nonlat_on`;
> negative means the non-latent d_v=128 path is faster.

---

## Best per config (DM-on)

| config | shape | Latent-512 best | NonLat-128 best | winner | latent OOM≤ | nonlat OOM≤ |
|--|--|--|--|--|--|--|
| C1 | sp8·q32  | 6.532 (k640, 60.3%) | 8.156 (k640, 31.2%) | **latent (20%)** | k640 | k800 |
| C2 | sp4·q64  | 10.724 (k448, 73.5%) | 8.278 (k672, 61.6%) | non-latent (23%) | k480 | k672 |
| C3 | sp4·q96  | 11.119 (k256, 67.4%) | 7.235 (k512, 67.0%) | non-latent (35%) | k384 | k512 |
| C4 | sp8·q128 | 7.514 (k256, 52.4%) | **5.390 (k384, 47.3%)** | non-latent (28%) | k256 | k512 |
| C5 | sp8·q64  | 7.129 (k448, 55.2%) | 5.716 (k448, 44.6%) | non-latent (20%) | k480 | k672 |
| C6 | sp8·q96  | 11.358 (k256, 34.7%) | 7.147 (k512, 35.7%) | non-latent (37%) | k384 | k512 |
| C7 | sp8·q160 | 9.525 (k256, 41.4%) | 6.311 (k512, 40.4%) | non-latent (34%) | k256 | k512 |
| C8 | sp8·q192 | **all OOM** | 7.285 (k384, 35.0%) | non-latent (only) | none | k480 |

**Global best (realistic pairing): C4 non-latent d_v=128, k384 = 5.390 ms.**

---

## Full per-config tables

### C1 — sp8·q32 (chunk_size=5120)
| k | Latent-512 DM-on | Latent-512 DM-off | NonLat-128 DM-on | NonLat-128 DM-off | lat speedup (on) |
|--:|:--:|:--:|:--:|:--:|:--:|
| 224 | 7.041 (55.9%) | 6.578 (59.9%) | 8.968 (28.4%) | 5.171 (49.3%) | 21.5% |
| 256 | 6.953 (56.7%) | 6.396 (61.6%) | 8.823 (28.9%) | 5.060 (50.4%) | 21.2% |
| 384 | 7.029 (56.0%) | 6.037 (65.2%) | 8.667 (29.4%) | 4.786 (53.3%) | 18.9% |
| 448 | 6.799 (57.9%) | 5.884 (66.9%) | 8.417 (30.3%) | 4.754 (53.6%) | 19.2% |
| 480 | 6.837 (57.6%) | 5.988 (65.8%) | 8.428 (30.2%) | 4.776 (53.4%) | 18.9% |
| 512 | 6.762 (58.2%) | 5.794 (68.0%) | 8.379 (30.4%) | 4.742 (53.7%) | 19.3% |
| 640 | 6.532 (60.3%) | 5.760 (68.4%) | 8.156 (31.2%) | 4.741 (53.8%) | 19.9% |
| 672 | OOM | OOM | 8.531 (29.9%) | 4.746 (53.7%) | — |
| 768 | OOM | OOM | 8.831 (28.9%) | 4.734 (53.8%) | — |
| 800 | OOM | OOM | 8.282 (30.8%) | 4.740 (53.8%) | — |

### C2 — sp4·q64 (chunk_size=5120)
| k | Latent-512 DM-on | Latent-512 DM-off | NonLat-128 DM-on | NonLat-128 DM-off | lat speedup (on) |
|--:|:--:|:--:|:--:|:--:|:--:|
| 224 | 11.898 (66.2%) | 11.793 (66.8%) | 8.874 (57.4%) | 8.069 (63.2%) | -34.1% |
| 256 | 12.405 (63.5%) | 12.260 (64.3%) | 8.749 (58.3%) | 8.468 (60.2%) | -41.8% |
| 384 | 11.940 (66.0%) | 11.790 (66.8%) | 8.557 (59.6%) | 8.224 (62.0%) | -39.5% |
| 448 | 10.724 (73.5%) | 10.579 (74.5%) | 8.498 (60.0%) | 7.376 (69.1%) | -26.2% |
| 480 | 12.120 (65.0%) | 11.886 (66.3%) | 8.676 (58.7%) | 8.344 (61.1%) | -39.7% |
| 512 | OOM | OOM | 8.487 (60.1%) | 8.085 (63.0%) | — |
| 640 | OOM | OOM | 8.334 (61.2%) | 7.978 (63.9%) | — |
| 672 | OOM | OOM | 8.278 (61.6%) | 7.275 (70.1%) | — |
| 768 | OOM | OOM | OOM | OOM | — |
| 800 | OOM | OOM | OOM | OOM | — |

### C3 — sp4·q96 (chunk_size=4992)
| k | Latent-512 DM-on | Latent-512 DM-off | NonLat-128 DM-on | NonLat-128 DM-off | lat speedup (on) |
|--:|:--:|:--:|:--:|:--:|:--:|
| 224 | 11.439 (65.5%) | 11.265 (66.5%) | 7.902 (61.3%) | 7.641 (63.4%) | -44.8% |
| 256 | 11.119 (67.4%) | 10.999 (68.1%) | 7.752 (62.5%) | 7.513 (64.5%) | -43.4% |
| 384 | 11.599 (64.6%) | 11.496 (65.1%) | 8.283 (58.5%) | 8.037 (60.3%) | -40.0% |
| 448 | OOM | OOM | 7.313 (66.3%) | 7.062 (68.6%) | — |
| 480 | OOM | OOM | 7.408 (65.4%) | 7.169 (67.6%) | — |
| 512 | OOM | OOM | 7.235 (67.0%) | 7.026 (69.0%) | — |
| 640–800 | OOM | OOM | OOM | OOM | — |

### C4 — sp8·q128 (chunk_size=5120)
| k | Latent-512 DM-on | Latent-512 DM-off | NonLat-128 DM-on | NonLat-128 DM-off | lat speedup (on) |
|--:|:--:|:--:|:--:|:--:|:--:|
| 224 | 7.857 (50.1%) | 7.681 (51.3%) | 5.704 (44.7%) | 5.486 (46.5%) | -37.7% |
| 256 | 7.514 (52.4%) | 7.332 (53.7%) | 5.451 (46.8%) | 5.217 (48.9%) | -37.8% |
| 384 | OOM | OOM | 5.390 (47.3%) | 5.075 (50.2%) | — |
| 448 | OOM | OOM | 6.094 (41.8%) | 5.704 (44.7%) | — |
| 480 | OOM | OOM | 5.510 (46.3%) | 5.142 (49.6%) | — |
| 512 | OOM | OOM | 5.427 (47.0%) | 4.998 (51.0%) | — |
| 640–800 | OOM | OOM | OOM | OOM | — |

### C5 — sp8·q64 (chunk_size=5120)
| k | Latent-512 DM-on | Latent-512 DM-off | NonLat-128 DM-on | NonLat-128 DM-off | lat speedup (on) |
|--:|:--:|:--:|:--:|:--:|:--:|
| 224 | 8.065 (48.8%) | 7.934 (49.6%) | 6.238 (40.9%) | 5.730 (44.5%) | -29.3% |
| 256 | 8.433 (46.7%) | 8.253 (47.7%) | 6.299 (40.5%) | 6.048 (42.1%) | -33.9% |
| 384 | 8.164 (48.2%) | 7.906 (49.8%) | 6.122 (41.6%) | 5.842 (43.6%) | -33.4% |
| 448 | 7.129 (55.2%) | 7.073 (55.7%) | 5.716 (44.6%) | 5.171 (49.3%) | -24.7% |
| 480 | 8.037 (49.0%) | 7.961 (49.5%) | 6.183 (41.2%) | 5.946 (42.9%) | -30.0% |
| 512 | OOM | OOM | 5.982 (42.6%) | 5.736 (44.4%) | — |
| 640 | OOM | OOM | 5.882 (43.3%) | 5.623 (45.3%) | — |
| 672 | OOM | OOM | 5.790 (44.0%) | 5.139 (49.6%) | — |
| 768–800 | OOM | OOM | OOM | OOM | — |

### C6 — sp8·q96 (chunk_size=5120)
| k | Latent-512 DM-on | Latent-512 DM-off | NonLat-128 DM-on | NonLat-128 DM-off | lat speedup (on) |
|--:|:--:|:--:|:--:|:--:|:--:|
| 224 | 11.631 (33.9%) | 11.590 (34.0%) | 7.902 (32.3%) | 7.754 (32.9%) | -47.2% |
| 256 | 11.358 (34.7%) | 11.312 (34.8%) | 7.717 (33.0%) | 7.581 (33.6%) | -47.2% |
| 384 | 11.976 (32.9%) | 11.837 (33.3%) | 8.458 (30.1%) | 8.216 (31.0%) | -41.6% |
| 448 | OOM | OOM | 7.208 (35.4%) | 7.055 (36.1%) | — |
| 480 | OOM | OOM | 7.344 (34.7%) | 7.177 (35.5%) | — |
| 512 | OOM | OOM | 7.147 (35.7%) | 6.973 (36.5%) | — |
| 640–800 | OOM | OOM | OOM | OOM | — |

### C7 — sp8·q160 (chunk_size=5120)
| k | Latent-512 DM-on | Latent-512 DM-off | NonLat-128 DM-on | NonLat-128 DM-off | lat speedup (on) |
|--:|:--:|:--:|:--:|:--:|:--:|
| 224 | 9.722 (40.5%) | 9.577 (41.1%) | 6.737 (37.8%) | 6.555 (38.9%) | -44.3% |
| 256 | 9.525 (41.4%) | 9.355 (42.1%) | 6.618 (38.5%) | 6.422 (39.7%) | -43.9% |
| 384 | OOM | OOM | 6.345 (40.2%) | 6.105 (41.7%) | — |
| 448 | OOM | OOM | 6.327 (40.3%) | 6.033 (42.2%) | — |
| 480 | OOM | OOM | 6.414 (39.7%) | 6.120 (41.6%) | — |
| 512 | OOM | OOM | 6.311 (40.4%) | 5.959 (42.8%) | — |
| 640–800 | OOM | OOM | OOM | OOM | — |

### C8 — sp8·q192 (chunk_size=5120)
| k | Latent-512 DM-on | Latent-512 DM-off | NonLat-128 DM-on | NonLat-128 DM-off | lat speedup (on) |
|--:|:--:|:--:|:--:|:--:|:--:|
| 224 | OOM | OOM | 7.795 (32.7%) | 7.606 (33.5%) | — |
| 256 | OOM | OOM | 7.427 (34.3%) | 7.232 (35.2%) | — |
| 384 | OOM | OOM | 7.285 (35.0%) | 7.045 (36.2%) | — |
| 448 | OOM | OOM | 7.310 (34.9%) | 6.992 (36.5%) | — |
| 480 | OOM | OOM | 7.381 (34.5%) | 7.096 (35.9%) | — |
| 512–800 | OOM | OOM | OOM | OOM | — |

---

## Conclusions

### 1. With the realistic pairing, non-latent d_v=128 wins everywhere except the extreme DM-bound C1
For 7 of 8 configs the **standard non-latent d_v=128 path is 20–37% faster** than latent d_v=512, because
the wide-V (512) PV matmul costs more than the 128-wide V all-gather it replaces. The lone exception is
**C1 (sp8·q32)**: there the tiny 32-row Q chunk makes the 8-device V-128 all-gather a dominant,
unhideable cost (non-latent util only ~29–31%), so eliminating it via latent (V from K) wins by ~20%
*despite* the 4× PV — latent C1 best 6.532 ms vs non-latent 8.156 ms.

So the latent-512 trade only pays when you are **so DM-bound that removing the V all-gather beats
quadrupling PV** — which here means only the smallest-q, highest-ring-count shape.

### 2. Global best is non-latent d_v=128, C4 (q128) k384 = 5.390 ms
The standard path at the q128 sweet spot remains the fastest 50K+5K prefill point overall. Among sp8
shapes the q-chunk ordering for non-latent d_v=128 is the same as the earlier d_v=128 sweep (q128 best,
q96/q192 worst due to padding waste).

### 3. The finer k grid exposes 448/672 as strong points
The added k values matter:
- **k448** is a local optimum for several configs: C2 latent (10.724, its best + highest util 73.5%),
  C5 latent (7.129, its best), C2/C5 non-latent DM-off dips.
- **k672** is the non-latent best for C2 (8.278) and C5 (5.790), and gives util spikes (C2 DM-off 70.1%).
- The coarse grid (256/384/512/640) would have missed these; 448 and 672 tile the per-device slabs
  (1248/1280/640) more favorably for the MM subblocking than the powers-of-two-ish neighbors.
- k384 remains anomalously poor on the 4992-slab configs (C3) and several sp8 shapes.

### 4. Latent d_v=512 collapses the L1 ceiling; non-latent d_v=128 tiles much further
Latent-512 OOMs 2–4 k-steps earlier than non-latent-128 in every config (e.g. C4 k256 vs k512, C1 k640
vs k800), and **C8 (q192) latent-512 OOMs at every k** — its 512-wide V/out CBs never fit. Non-latent
d_v=128 keeps small CBs, so it reaches higher k_chunk (and thus higher MM util) before OOM.

### 5. DM overhead splits sharply by which path/d_v
- **Non-latent d_v=128** stays the most DM-exposed where Q is tiny: C1 DM overhead is still huge
  (8.2→4.7 ms off, ~43%); for larger-q sp8 and sp4 it's only a few %.
- **Latent d_v=512** is near compute-bound everywhere (DM overhead typically 1–15%): the wide PV gives
  plenty of compute to hide the K stream, and there is no V gather at all.

**Bottom line:** keep **non-latent d_v=128** for prefill in general (it's faster and tiles further);
reach for **latent d_v=512** only in the pathologically DM-bound corner (sp8, very small q_chunk like
C1), where avoiding the V all-gather outweighs the wider PV.

## Reproduce
```
bash sweep_driver_full.sh   # -> sweep_runs/results_full.tsv  (latent d_v=512 / non-latent d_v=128, DM on/off, 10 k, OOM-skip)
```
TEMP scaffolding (env hooks incl. CHUNKED_D_V + extended k_chunk_sizes, dm_toggle.py, sweep_driver*.sh)
— revert before merge.
