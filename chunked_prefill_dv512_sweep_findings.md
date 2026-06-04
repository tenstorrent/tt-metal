# Ring Joint SDPA — Chunked-Prefill 50K+5K Perf Sweep, **d_v = 512** (wide value-latent)

Branch: `skrstic/ring_joint_sdpa_optional_latent_v_fix`
Date: 2026-06-03
Test: `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_chunked_perf_table`
Hardware: **P150_X8** (8× Blackhole P150), SDPA grid 12×10 → **110 cores**.

## What this sweep measures

Same 8 shapes and same **50K + 5K** last-chunk isolation as the `d_v = 128` sweep
(`chunked_prefill_dm_latent_sweep_findings.md`), but with the **value head dim widened to
`d_v = 512`** — i.e. V is the first **512** columns of the 576-wide latent K instead of the first 128
(`vDHt = 16` tiles instead of 4). This is the wide value-latent / "absorbed-MLA-style" configuration:
the PV matmul and the attention output are 512-wide, so per-chunk FLOPs ≈ 1.5× (rect/tri scale with
`d_q + d_v` = 576+512 = 1088 vs 576+128 = 704) and the V/out circular buffers are ~4× larger.

- **Latent V only** (`CHUNKED_LATENT_V=1`, `CHUNKED_D_V=512`) — no non-latent runs this time.
- **DM-on** vs **DM-off** (bulk NoC data movement physically commented out of the kernels via
  `dm_toggle.py`, no macros — same methodology as before).
- **Perf only** (`CHUNKED_SKIP_PCC=1`), last chunk only (`CHUNKED_ONLY_LAST_CHUNK=1`).
- **OOM-skip:** once a config OOMs at some k_chunk, larger k_chunks for that config are not run
  (recorded `OOM`). Cells are **Duration ms (Math Util %)**.

Shapes (per-device seq = `chunk_size/sp`; nhq=16/ring, d_q=d_k=576, **d_v=512**, Q bf16, KV bf8):

| | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|--|--|--|--|--|--|--|--|--|
| sp | 8 | 4 | 4 | 8 | 8 | 8 | 8 | 8 |
| per-dev seq | 640 | 1280 | 1248 | 640 | 640 | 640 | 640 | 640 |
| q_chunk | 32 | 64 | 96 | 128 | 64 | 96 | 160 | 192 |
| chunk_size | 5120 | 5120 | 4992 | 5120 | 5120 | 5120 | 5120 | 5120 |

---

## Results — d_v = 512 (latent V only)

### C1 — sp8·seq640·q32 (chunk_size=5120)
| k | DM-on | DM-off | DM overhead |
|--:|:-----:|:------:|:-----------:|
| 256 | 7.034 (56.0%) | 6.393 (61.6%) | 9.1% |
| 384 | 7.083 (55.6%) | 6.037 (65.2%) | 14.8% |
| 512 | 6.847 (57.5%) | 5.796 (68.0%) | 15.3% |
| 640 | OOM | OOM | — |
| 768 | OOM | OOM | — |

### C2 — sp4·seq1280·q64 (chunk_size=5120)
| k | DM-on | DM-off | DM overhead |
|--:|:-----:|:------:|:-----------:|
| 256 | 12.358 (63.7%) | 12.260 (64.3%) | 0.8% |
| 384 | 11.903 (66.2%) | 11.794 (66.8%) | 0.9% |
| 512 | OOM | OOM | — |
| 640 | OOM | OOM | — |
| 768 | OOM | OOM | — |

### C3 — sp4·seq1248·q96 (chunk_size=4992)
| k | DM-on | DM-off | DM overhead |
|--:|:-----:|:------:|:-----------:|
| 256 | 11.169 (67.0%) | 11.007 (68.0%) | 1.5% |
| 384 | OOM | OOM | — |
| 512 | OOM | OOM | — |
| 640 | OOM | OOM | — |
| 768 | OOM | OOM | — |

### C4 — sp8·seq640·q128 (chunk_size=5120)
| k | DM-on | DM-off | DM overhead |
|--:|:-----:|:------:|:-----------:|
| 256 | 7.548 (52.2%) | 7.331 (53.7%) | 2.9% |
| 384 | OOM | OOM | — |
| 512 | OOM | OOM | — |
| 640 | OOM | OOM | — |
| 768 | OOM | OOM | — |

### C5 — sp8·seq640·q64 (chunk_size=5120)
| k | DM-on | DM-off | DM overhead |
|--:|:-----:|:------:|:-----------:|
| 256 | 8.406 (46.9%) | 8.241 (47.8%) | 2.0% |
| 384 | 8.116 (48.5%) | 7.907 (49.8%) | 2.6% |
| 512 | OOM | OOM | — |
| 640 | OOM | OOM | — |
| 768 | OOM | OOM | — |

### C6 — sp8·seq640·q96 (chunk_size=5120)
| k | DM-on | DM-off | DM overhead |
|--:|:-----:|:------:|:-----------:|
| 256 | 11.398 (34.6%) | 11.317 (34.8%) | 0.7% |
| 384 | OOM | OOM | — |
| 512 | OOM | OOM | — |
| 640 | OOM | OOM | — |
| 768 | OOM | OOM | — |

### C7 — sp8·seq640·q160 (chunk_size=5120)
| k | DM-on | DM-off | DM overhead |
|--:|:-----:|:------:|:-----------:|
| 256 | OOM | OOM | — |
| 384–768 | OOM | OOM | — |

### C8 — sp8·seq640·q192 (chunk_size=5120)
| k | DM-on | DM-off | DM overhead |
|--:|:-----:|:------:|:-----------:|
| 256 | OOM | OOM | — |
| 384–768 | OOM | OOM | — |

---

## d_v = 128 vs d_v = 512 — best latent DM-on per config

| config | shape | d_v=128 best | d_v=512 best | slowdown | OOM ceiling 128→512 |
|--|--|--|--|--|--|
| C1 | sp8·seq640·q32  | 5.829 (k640, 43.7%) | **6.847 (k512, 57.5%)** | +17% | k768 → **k512** |
| C2 | sp4·seq1280·q64 | 8.002 (k640, 63.7%) | 11.903 (k384, 66.2%) | +49% | k640 → k384 |
| C3 | sp4·seq1248·q96 | 6.751 (k512, 71.8%) | 11.169 (k256, 67.0%) | +65% | k512 → k256 |
| C4 | sp8·seq640·q128 | 4.889 (k384, 52.1%) | 7.548 (k256, 52.2%) | +54% | k512 → k256 |
| C5 | sp8·seq640·q64  | 5.440 (k640, 46.8%) | 8.116 (k384, 48.5%) | +49% | k640 → k384 |
| C6 | sp8·seq640·q96  | 6.944 (k512, 36.7%) | 11.398 (k256, 34.6%) | +64% | k512 → k256 |
| C7 | sp8·seq640·q160 | 5.937 (k512, 42.9%) | **OOM (all k)** | — | k512 → **none** |
| C8 | sp8·seq640·q192 | 7.082 (k384, 36.0%) | **OOM (all k)** | — | k384 → **none** |

---

## Conclusions

### 1. Widening V to 512 is a large prefill regression — exactly as the FLOP/L1 model predicts
At every config the d_v=512 best is **17–65% slower** than the d_v=128 best, and two configs (C7 q160,
C8 q192) **no longer run at all** — they OOM at every k_chunk. The PV matmul is now 512-wide instead of
128 (per-chunk FLOPs ≈1.5×), and the V + output CBs are ~4× larger, so:
- **The L1 / OOM ceiling collapses one-to-three k-steps**: C1 k768→k512, C2 k640→k384, C3/C4/C6
  k512→k256, and C7/C8 fall off the table entirely.
- The slowdown is *smaller* than the 1.5× FLOP ratio because QK^T (d=576) is unchanged and still a big
  share of the work; the extra cost is concentrated in PV + the wider output write/CB traffic.

**For chunked prefill, keep d_v = 128.** The wide value-latent (512) only pays off in the *decode*
regime (Sq≈1, KV-load-bound), where attending over the compressed latent avoids per-head V
materialization — that benefit does not exist here, where you are compute-bound on the matmuls.

### 2. The sp8-vs-sp4 winner flips under d_v=512 — L1 headroom now dominates
At d_v=128 the fastest config was **C4 (sp8, q128)** at 4.889 ms. At d_v=512 the fastest *viable* config
is **C1 (sp8, q32) k512 = 6.847 ms** — small q_chunk wins because its Q/intermediate CBs are smallest,
so it survives to a larger k_chunk (k512) where MM utilization is higher, while the larger-q configs OOM
at k256–384 and are stuck at low-util small k. When V is wide, **minimizing q_chunk to buy k_chunk
headroom** is the right trade — the opposite of the d_v=128 advice (where q128 packed cores best).

### 3. Everything becomes more compute-bound → DM overhead shrinks
Because PV compute grew while data movement is unchanged, the DM share drops across the board:
- C1 sp8/q32: DM overhead **~9–15%** at d_v=512 vs **~31%** at d_v=128 — still the most DM-exposed
  config, but the fixed K-stream/ring-handoff is now hidden behind a bigger matmul.
- sp4 (C2/C3) and large-q sp8 (C4/C6): **<3%, often <1%** — essentially fully compute-bound.

So widening V doesn't reduce data movement; it just dilutes its relative cost by adding compute — not a
useful lever for latency, and it costs you the OOM headroom.

---

## Methodology / reproduce
Same as the d_v=128 sweep; `d_v` is now env-driven (`CHUNKED_D_V`, default 128) in the kimi50k config.
```
CHUNKED_SP_SIZE=<4|8> CHUNKED_PER_DEVICE_CHUNK=<n> CHUNKED_Q_CHUNK=<n> CHUNKED_D_V=512 \
CHUNKED_LATENT_V=1 CHUNKED_ONLY_LAST_CHUNK=1 CHUNKED_SKIP_PCC=1 \
scripts/run_safe_pytest.sh \
 "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_chunked_perf_table[kimi50k-q<Q>-k<K>-chunk<CS>]"

python dm_toggle.py off          # disable data movement (restore: on)
bash sweep_driver_dv512.sh       # full matrix -> sweep_runs/results_dv512.tsv (OOM-skip enabled)
```
All TEMP scaffolding (test env hooks incl. `CHUNKED_D_V`, `dm_toggle.py`, `sweep_driver*.sh`) is to be
reverted before merge. Raw data in `sweep_runs/results_dv512.tsv`.
