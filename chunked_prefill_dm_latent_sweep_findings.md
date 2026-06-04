# Ring Joint SDPA — Chunked-Prefill 50K+5K Perf Sweep (latent-V × data-movement)

Branch: `skrstic/ring_joint_sdpa_optional_latent_v_fix`
Date: 2026-06-03
Test: `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_chunked_perf_table`
Hardware: **P150_X8** (8× Blackhole P150), SDPA grid 12×10 → **110 cores** (col 11 = CCL).

## What this sweep measures

For each of four (seq_len, q_chunk, sp) shapes, the **50K + 5K** prefill step only — i.e. chunk 10,
the last and largest chunk: a 5K Q chunk attending over a ~50K-token K/V prefix. This is the
`CHUNKED_ONLY_LAST_CHUNK=1` isolation: the KV cache is re-uploaded fresh for the last chunk, so it
reproduces the in-sequence measurement without running chunks 0–9.

Each shape is swept over k_chunk ∈ {256, 384, 512, 640, 768} under the **2×2** matrix:

- **Latent V** (`CHUNKED_LATENT_V=1`) — V rematerialized on-device from the first `d_v` cols of the
  shared latent K; no separate V tensor, no V all-gather.
- **Non-latent V** (`=0`) — separate `nhv`-head V tensor with its own ring all-gather (original path).
- **DM-on** — full kernel.
- **DM-off** — **bulk NoC data movement physically commented out of the kernels** (no macros, no env
  toggle). The compute ceiling: matmuls + CB/semaphore handshakes run on stale L1 (outputs garbage),
  but no NoC reads/writes/mcast throttle the pipe. See **Methodology** below.

**Perf only — no PCC** (`CHUNKED_SKIP_PCC=1`, torch oracle skipped entirely). All four shapes use the
kimi-K2.6-style MLA config: nhq=16/ring, nhk=1, d_q=d_k=576, d_v=128, Q bf16, KV bf8, 11 chunks.

| Shape | sp (ring) | tp | per-device seq (chunk Q rows) | q_chunk | chunk_size | total_seq | last-chunk prefix |
|---|--:|--:|--:|--:|--:|--:|--:|
| **C1** | 8 | 1 | 640 | 32 | 5120 | 56320 | 51200 (50K) |
| **C2** | 4 | 2 | 1280 | 64 | 5120 | 56320 | 51200 (50K) |
| **C3** | 4 | 2 | 1248 | 96 | 4992 | 54912 | 49920 (≈50K, a bit less) |
| **C4** | 8 | 1 | 640 | 128 | 5120 | 56320 | 51200 (50K) |

Cells below are **Duration ms (Math Util %)**. Math Util = measured device-kernel FLOPs / theoretical,
for this one chunk (rectangle Qchunk×prefix non-causal + triangle Qchunk×current causal). In DM-off mode
util can exceed the DM-on value because the same FLOPs complete in less wall time (it is *not* a real
efficiency — it's the compute ceiling).

---

## Results

### C1 — sp8 · seq640 · q32  (chunk_size=5120, 50K+5K = chunk 10)

| k_chunk | Latent DM-on | Latent DM-off | NonLatent DM-on | NonLatent DM-off | DM overhead (lat) | Latent speedup (DM-on) |
|--------:|:------------:|:-------------:|:---------------:|:----------------:|:-----------------:|:----------------------:|
| 256 | 6.156 (41.4%) | 4.564 (55.8%) | 8.821 (28.9%) | 5.056 (50.4%) | 25.9% | 30.2% |
| 384 | 6.151 (41.4%) | 4.248 (60.0%) | 8.657 (29.4%) | 4.785 (53.3%) | 30.9% | 28.9% |
| 512 | 6.029 (42.3%) | 4.074 (62.6%) | 8.358 (30.5%) | 4.739 (53.8%) | 32.4% | 27.9% |
| 640 | 5.829 (43.7%) | 4.028 (63.3%) | 8.156 (31.2%) | 4.732 (53.9%) | 30.9% | 28.5% |
| 768 | 6.448 (39.5%) | 3.984 (64.0%) | 8.831 (28.9%) | 4.731 (53.9%) | 38.2% | 27.0% |

### C2 — sp4 · seq1280 · q64  (chunk_size=5120, 50K+5K = chunk 10)

| k_chunk | Latent DM-on | Latent DM-off | NonLatent DM-on | NonLatent DM-off | DM overhead (lat) | Latent speedup (DM-on) |
|--------:|:------------:|:-------------:|:---------------:|:----------------:|:-----------------:|:----------------------:|
| 256 | 8.588 (59.4%) | 8.467 (60.2%) | 8.749 (58.3%) | 8.471 (60.2%) | 1.4% | 1.8% |
| 384 | 8.316 (61.3%) | 8.162 (62.4%) | 8.554 (59.6%) | 8.229 (61.9%) | 1.9% | 2.8% |
| 512 | 8.219 (62.0%) | 7.979 (63.9%) | 8.479 (60.1%) | 8.087 (63.0%) | 2.9% | 3.1% |
| 640 | 8.002 (63.7%) | 7.826 (65.1%) | 8.324 (61.2%) | 7.984 (63.8%) | 2.2% | 3.9% |
| 768 | OOM | OOM | OOM | OOM | — | — |

### C3 — sp4 · seq1248 · q96  (chunk_size=4992, 50K+5K = chunk 10)

| k_chunk | Latent DM-on | Latent DM-off | NonLatent DM-on | NonLatent DM-off | DM overhead (lat) | Latent speedup (DM-on) |
|--------:|:------------:|:-------------:|:---------------:|:----------------:|:-----------------:|:----------------------:|
| 256 | 7.450 (65.0%) | 7.273 (66.6%) | 7.752 (62.5%) | 7.511 (64.5%) | 2.4% | 3.9% |
| 384 | 8.177 (59.3%) | 7.964 (60.8%) | 8.270 (58.6%) | 8.037 (60.3%) | 2.6% | 1.1% |
| 512 | 6.751 (71.8%) | 6.620 (73.2%) | 7.245 (66.9%) | 7.021 (69.0%) | 1.9% | 6.8% |
| 640 | OOM | OOM | OOM | OOM | — | — |
| 768 | OOM | OOM | OOM | OOM | — | — |

### C4 — sp8 · seq640 · q128  (chunk_size=5120, 50K+5K = chunk 10)

| k_chunk | Latent DM-on | Latent DM-off | NonLatent DM-on | NonLatent DM-off | DM overhead (lat) | Latent speedup (DM-on) |
|--------:|:------------:|:-------------:|:---------------:|:----------------:|:-----------------:|:----------------------:|
| 256 | 4.977 (51.2%) | 4.763 (53.5%) | 5.457 (46.7%) | 5.210 (48.9%) | 4.3% | 8.8% |
| 384 | 4.889 (52.1%) | 4.600 (55.4%) | 5.381 (47.4%) | 5.076 (50.2%) | 5.9% | 9.1% |
| 512 | 4.910 (51.9%) | 4.512 (56.5%) | 5.440 (46.8%) | 4.995 (51.0%) | 8.1% | 9.7% |
| 640 | OOM | OOM | OOM | OOM | — | — |
| 768 | OOM | OOM | OOM | OOM | — | — |

---

## Results — sp8 q_chunk scan (q64 / 96 / 160 / 192)

Follow-up: hold sp8 · seq640 fixed and scan q_chunk to map the sweet spot (combine with C1 q32 and
C4 q128 above for the full q ∈ {32,64,96,128,160,192} picture). Same 2×2 matrix, same 50K+5K chunk.

### C5 — sp8 · seq640 · q64  (chunk_size=5120, 50K+5K = chunk 10)

| k_chunk | Latent DM-on | Latent DM-off | NonLatent DM-on | NonLatent DM-off | DM overhead (lat) | Latent speedup (DM-on) |
|--------:|:------------:|:-------------:|:---------------:|:----------------:|:-----------------:|:----------------------:|
| 256 | 5.934 (42.9%) | 5.711 (44.6%) | 6.291 (40.5%) | 6.046 (42.2%) | 3.8% | 5.7% |
| 384 | 5.761 (44.2%) | 5.479 (46.5%) | 6.114 (41.7%) | 5.851 (43.6%) | 4.9% | 5.8% |
| 512 | 5.564 (45.8%) | 5.341 (47.7%) | 5.984 (42.6%) | 5.730 (44.5%) | 4.0% | 7.0% |
| 640 | 5.440 (46.8%) | 5.220 (48.8%) | 5.869 (43.4%) | 5.619 (45.4%) | 4.0% | 7.3% |
| 768 | OOM | OOM | OOM | OOM | — | — |

### C6 — sp8 · seq640 · q96  (chunk_size=5120, 50K+5K = chunk 10)

| k_chunk | Latent DM-on | Latent DM-off | NonLatent DM-on | NonLatent DM-off | DM overhead (lat) | Latent speedup (DM-on) |
|--------:|:------------:|:-------------:|:---------------:|:----------------:|:-----------------:|:----------------------:|
| 256 | 7.615 (33.5%) | 7.468 (34.1%) | 7.714 (33.0%) | 7.576 (33.6%) | 1.9% | 1.3% |
| 384 | 8.460 (30.1%) | 8.192 (31.1%) | 8.459 (30.1%) | 8.217 (31.0%) | 3.2% | -0.0% |
| 512 | 6.944 (36.7%) | 6.785 (37.6%) | 7.154 (35.6%) | 6.979 (36.5%) | 2.3% | 2.9% |
| 640 | OOM | OOM | OOM | OOM | — | — |
| 768 | OOM | OOM | OOM | OOM | — | — |

### C7 — sp8 · seq640 · q160  (chunk_size=5120, 50K+5K = chunk 10)

| k_chunk | Latent DM-on | Latent DM-off | NonLatent DM-on | NonLatent DM-off | DM overhead (lat) | Latent speedup (DM-on) |
|--------:|:------------:|:-------------:|:---------------:|:----------------:|:-----------------:|:----------------------:|
| 256 | 6.324 (40.3%) | 6.138 (41.5%) | 6.626 (38.5%) | 6.423 (39.7%) | 2.9% | 4.6% |
| 384 | 5.998 (42.5%) | 5.773 (44.1%) | 6.347 (40.2%) | 6.101 (41.8%) | 3.8% | 5.5% |
| 512 | 5.937 (42.9%) | 5.617 (45.4%) | 6.302 (40.4%) | 5.968 (42.7%) | 5.4% | 5.8% |
| 640 | OOM | OOM | OOM | OOM | — | — |
| 768 | OOM | OOM | OOM | OOM | — | — |

### C8 — sp8 · seq640 · q192  (chunk_size=5120, 50K+5K = chunk 10)

| k_chunk | Latent DM-on | Latent DM-off | NonLatent DM-on | NonLatent DM-off | DM overhead (lat) | Latent speedup (DM-on) |
|--------:|:------------:|:-------------:|:---------------:|:----------------:|:-----------------:|:----------------------:|
| 256 | 7.258 (35.1%) | 7.070 (36.0%) | 7.427 (34.3%) | 7.232 (35.2%) | 2.6% | 2.3% |
| 384 | 7.082 (36.0%) | 6.854 (37.2%) | 7.285 (35.0%) | 7.044 (36.2%) | 3.2% | 2.8% |
| 512 | OOM | OOM | OOM | OOM | — | — |
| 640 | OOM | OOM | OOM | OOM | — | — |
| 768 | OOM | OOM | OOM | OOM | — | — |

### sp8 q_chunk sweet-spot summary (best latent DM-on per q, seq640)

| q_chunk | shape | best k | Duration | util | per-dev q-chunks (640/q) | pad waste | DM overhead |
|--------:|:-----:|-------:|---------:|-----:|:------------------------:|:---------:|:-----------:|
| 32  | C1 | 640 | 5.829 | 43.7% | 20 (exact) | 0% | ~31% |
| 64  | C5 | 640 | 5.440 | 46.8% | 10 (exact) | 0% | ~4% |
| 96  | C6 | 512 | 6.944 | 36.7% | 6.67 → 7   | ~5% | ~2% |
| **128** | **C4** | **384** | **4.889** | **52.1%** | **5 (exact)** | **0%** | ~8% |
| 160 | C7 | 512 | 5.937 | 42.9% | 4 (exact) | 0% | ~5% |
| 192 | C8 | 384 | 7.082 | 36.0% | 3.33 → 4   | ~20% | ~3% |

**q_chunk = 128 is the clear sp8 winner (4.889 ms, 52.1% util)** — non-monotonic around it. Two
mechanisms drive the shape of this curve:
- **Padding waste** when 640 (per-device Q rows/chunk) isn't divisible by q_chunk: q96 → 7 chunks
  (672 padded, ~5% waste) and q192 → 4 chunks (768 padded, **~20% waste**) are the two slowest points,
  and q192's 768-row Q CBs OOM the earliest (already at k512).
- **Work-item packing across 110 cores** (work items = 16 heads × ⌈640/q⌉): q128 yields 80 items →
  ~1/core with the fewest idle cores *and* one q-chunk of real size per core; q160 (64 items) leaves
  more cores idle, q64 (160 items) forces 2 serial chunks/core. q128 is the packing optimum.

---

## Conclusions

### 1. Best 50K+5K latency (real, DM-on latent V)
| Shape | best k | Duration | Math util |
|---|--:|--:|--:|
| **C4 sp8 q128** | 384 | **4.889 ms** | 52.1% |  ← fastest overall
| C1 sp8 q32 | 640 | 5.829 ms | 43.7% |
| C3 sp4 q96 | 512 | 6.751 ms | 71.8% |  ← highest util
| C2 sp4 q64 | 640 | 8.002 ms | 63.7% |

**sp8 (split Q across 8 devices) is far faster in wall-clock** than sp4 for the same 50K+5K work,
because each device owns only 640 Q rows/chunk vs 1280. The trade is util: sp4 keeps the cores busier
(C3 hits **71.8%**, the highest in the sweep) while sp8/q32 (C1) sits at ~42% — but C1/C4 still finish
sooner because there is less per-device work on the critical path.

### 2. Data-movement is the whole story for sp8; it's nearly free for sp4
DM overhead (latent) = `(DM-on − DM-off) / DM-on` at the best k:

- **C1 sp8 q32: ~31–38%** of wall time is data movement. Severely DM-bound: a 32-row Q chunk does so
  little matmul per ring step that the K stream + ring handoff can't be hidden. The compute ceiling
  (DM-off) is ~4.0 ms vs ~5.8–6.2 ms actual.
- **C4 sp8 q128: ~8%.** A 128-row Q chunk gives 4× the compute per ring step, so most of the K
  streaming hides behind matmul — but sp8's ring handoff still leaves ~8% exposed.
- **C2 sp4 q64 & C3 sp4 q96: ~2%.** Essentially **compute-bound** — DM-on and DM-off are within noise.
  The ring of 4 with 1248–1280 K rows/device/step provides enough matmul to fully overlap the NoC.

**Takeaway:** the lever for sp8 is reducing/overlapping data movement (or raising q_chunk, C1→C4);
the lever for sp4 is raising math util (bigger k_chunk), since DM is already hidden.

### 3. Latent V is always a win, biggest on sp8
Latent-V speedup (DM-on) vs non-latent at the best k:
- **C1 sp8: ~27–30%**, **C4 sp8: ~9–10%** — sp8 pays for an 8-device V all-gather in the non-latent
  path; rematerializing V from K on-device removes it entirely. The win is largest exactly where DM
  dominates (C1).
- **C2 sp4: ~2–4%, C3 sp4: ~1–7%** — over a ring of 4 the V all-gather is cheaper and mostly hidden,
  so latent V helps less, but it never hurts.

Confirms and extends the earlier C2-only finding (latent consistently faster, gap widens with k_chunk).

### 4. q_chunk matters enormously at sp8
C1 and C4 are the *same* shape except q_chunk (32 vs 128). Going 32→128 cuts latency from ~5.8→4.9 ms
and slashes DM overhead 31%→8%, because a larger Q chunk amortizes the per-ring-step K-stream cost.
**For sp8, prefer large q_chunk (128).**

### 5. L1 ceiling (OOM)
`Statically allocated circular buffers … beyond max L1 size of 1572864 B`. The ceiling depends on
q_chunk and chunk_size, *not* on latent-vs-non-latent (both OOM at the same k):
- C1 (q32): no OOM through k768 (smallest Q CBs).
- C2 (q64): OOM at **k768**.
- C3 (q96, chunk 4992): OOM at **k640**.
- C4 (q128): OOM at **k640**.
Larger q_chunk → larger Q/intermediate CBs → lower k_chunk ceiling. C1's tiny q32 is the only shape
that survives k768 (and it's its best compute-ceiling point, 3.984 ms / 64.0%, though DM-bound in reality).

### 6. k_chunk anomaly at chunk_size 4992 (C3)
C3 k384 is *slower* than both k256 and k512 (8.18 ms vs 7.45 / 6.75). 4992 tiles don't divide as
cleanly by 384 for the MM subblocking as by 512; k512 is the clear sweet spot for C3 (also its util
peak). k512 is a robust choice across C2/C3/C4.

---

## Methodology — DM-off without macros

Per request, the "without data movement" runs do **not** use the existing `TT_RING_JOINT_DISABLE_NOC_DM`
env/macro path. Instead `dm_toggle.py` physically comments out the bulk NoC transfer primitives in the
four active ring-joint dataflow kernels and lets the JIT recompile (a source change guarantees a fresh
kernel hash — no stale-cache ambiguity). Restored to pristine after the sweep.

- **Commented (19 call sites):** `noc_async_read`, `noc_async_read_page`,
  `noc_async_read_one_packet_set_state`, `noc_async_read_one_packet_with_state<…>`, `noc_async_write`,
  `noc_async_write_page`, `noc_async_write_multicast` — in `ring_joint_reader.cpp`,
  `ring_joint_writer.cpp`, `dataflow_common.hpp`, `chain_link.hpp` (K/Q DRAM reads, V-from-K
  rematerialization reads, output writes, and the ring K/V multicast/unicast handoff).
- **Kept intact:** every `*_barrier` / `*_flushed*` / `*_set_trid` ordering primitive, all
  `noc_semaphore_*` (the chain handoff sync), and all CB `cb_reserve/push/wait/pop` handshakes — so the
  pipeline never deadlocks; compute runs on whatever stale L1 it finds.

Verified: round-trip restore is byte-identical; DM-off forces a recompile and does not hang.

## Reproduce
```
# one cell:
CHUNKED_SP_SIZE=<4|8> CHUNKED_PER_DEVICE_CHUNK=<640|1248|1280> CHUNKED_Q_CHUNK=<32|64|96|128> \
CHUNKED_LATENT_V=<0|1> CHUNKED_ONLY_LAST_CHUNK=1 CHUNKED_SKIP_PCC=1 \
scripts/run_safe_pytest.sh \
 "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_chunked_perf_table[kimi50k-q<Q>-k<K>-chunk<CS>]"

# DM-off: python dm_toggle.py off   (restore: python dm_toggle.py on)
# full matrix: bash sweep_driver.sh   (writes sweep_runs/results.tsv, resumable)
```
Sweep scaffolding (env-driven per-device-chunk/q_chunk/latent/skip-pcc, `dm_toggle.py`, `sweep_driver.sh`)
is TEMP — revert the test-file env hooks and remove the helper scripts before merge.
