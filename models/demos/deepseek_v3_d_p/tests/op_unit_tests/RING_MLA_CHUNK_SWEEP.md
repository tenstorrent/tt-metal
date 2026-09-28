# Kimi-K2.7 ring MLA: chunk-size sweep on the 8x4 torus

Goal: the smallest per-device ISL (global chunk / SP=8) at which the dense chunked-prefill
`ttnn.transformer.ring_mla` stays compute bound with the same core count and roughly flat math
utilization. Baseline: `main` @ `db596ae144d`, Blackhole Galaxy, FABRIC_2D_TORUS_XY.

## Setup (matches `mla.py::_chunked_attn`)

- Q `[1, 16, chunk/8, 576]` bf16 per device (64 heads / TP 4); KVPE cache bf8, V = first 512 cols.
- KV prefix held at ~51200 tokens (rounded to a whole number of chunks) + the current chunk.
- Fused KV all-gather on SP axis 0, `num_links=2`, CCL column at x=11; SDPA grid 11x10 = 110 cores.
- HiFi2, `packer_l1_acc=True`, `exp_approx_mode=False`, scalar `kv_actual_isl` path.
- Util = matmul FLOPs (rectangle + causal half, `effective_kv = prefix + chunk/2`) /
  (110 cores x 2048 FLOP/cycle x 1.35 GHz x duration); same formula as the nightly
  `test_ring_mla_chunked_perf_check`, and matches tracy's `PM IDEAL` / `PM FPU UTIL`.

## Tooling

| piece | path |
|---|---|
| traced test (warm-up, capture, N replays) | `test_ring_mla_chunk_sweep.py` |
| tracy summary (kernel time, core count, PM IDEAL) | `summarize_ring_mla_tracy.py` |
| compute-only switch | `RING_SDPA_COMPUTE_ONLY=1` (host env; adds a kernel define) |

```
# realtime profiler, one window per replay (fast; appends generated/ring_mla_chunk_sweep/rt_results.jsonl)
pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_mla_chunk_sweep.py -k "chunk5120-q32-k640"
# tracy
python -m tracy -r -p -v -m pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_mla_chunk_sweep.py -k "chunk5120-q32-k640"
python models/demos/deepseek_v3_d_p/tests/op_unit_tests/summarize_ring_mla_tracy.py
# compute only
RING_SDPA_COMPUTE_ONLY=1 pytest ...test_ring_mla_chunk_sweep.py -k "chunk5120-q32-k640"
```

### Compute-only mode

`RING_SDPA_COMPUTE_ONLY=1` removes every data movement while keeping all CB
reserve/push/wait/pop handshakes, so compute runs on stale L1:

- `fetch_block` (Q/K/V/prev-out DRAM reads) and the output/stat writes in
  `write_block*` / `issue_stats_column_*` are no-ops (`dataflow_common.hpp`, `ring_joint_writer.cpp`).
- K/V store-and-forward chains (receive/forward) and the L1->L1 latent-V materialization are skipped
  (`ring_joint_reader.cpp`); every core acts as its own source.
- The reader no longer waits for fused all-gather signals (`fused_op_receiver.hpp`), and the four
  all-gather worker kernels return at entry.
- Kept: the rotated-Q remainder handoff semaphore (a compute dependency, not a transfer) and the
  on-core mask / scalar CB generation.

## Step 1 — tooling check at 5k (640 / device)

| source | median us | util % | notes |
|---|---|---|---|
| realtime profiler, 10 replays | 5877 | 67.0 | spread 5875-5885 |
| tracy, 10 replays | 5867 | 67.1 (PM FPU UTIL) | PM IDEAL 3939 us; CORE COUNT 114 = 110 SDPA + 4 AG workers; device skew ~50 us |
| nightly perf check expectation | — | 68.5 | 4x8 1D ring, `packer_l1_acc=False`, fresh KV |

5k q/k sweet spot (realtime, us / util):

| q \ k | 128 | 256 | 320 | 512 | 640 | 1280 |
|---|---|---|---|---|---|---|
| 32 | 7719 / 51.0 | 6646 / 59.3 | 6294 / 62.6 | 6021 / 65.4 | **5880 / 67.0** | L1 overflow |
| 64 | 7523 / 52.4 | 6643 / 59.3 | 6197 / 63.6 | L1 overflow | L1 overflow | L1 overflow |

The current `MLA_SDPA_CONFIG[640]` pick (q32/k640) is the best. The naive occupancy
`units / (110 * ceil(units / 110))` (0.73 for q64) does not predict perf: the Q-remainder rotation
spreads leftover chunks, so it is logged only as a reference.

## Step 2 — compute-only at 5k

| mode | median us | util % |
|---|---|---|
| full | 5877 | 67.0 |
| compute only (3 runs, no hang) | 5784 | 68.1 |

Data movement costs 1.6%: at 5k the op is compute bound; the gap to ideal is inside compute.

## Step 3 — 1k / 2k / 3k / 4k

Full sweep: q ∈ {32, 64, 128} x k ∈ {128, 256, 320, 512, 640, 1024, 1280}; every k ≥ 1024 overflows L1.
Realtime profiler, median of 10 replays, us / util %. Full mode first, compute-only in brackets.

**1k (128 / device, 64 work units < 110 cores)**

| q \ k | 128 | 256 | 320 | 512 | 640 |
|---|---|---|---|---|---|
| 32 | **2473 / 30.6** [2348 / 32.3] | 2477 / 30.6 [2014 / 37.6] | 2631 / 28.8 [1930 / 39.2] | 2832 / 26.8 [1823 / 41.6] | 3010 / 25.2 [1803 / 42.0] |
| 64 | 4481 / 16.9 | 4008 / 18.9 | 3790 / 20.0 | 3855 / 19.7 | L1 |
| 128 | 7881 / 9.6 | 7013 / 10.8 | 7559 / 10.0 | L1 | L1 |

**2k (256 / device)**

| q \ k | 128 | 256 | 320 | 512 | 640 |
|---|---|---|---|---|---|
| 32 | 3154 / 48.5 | 2714 / 56.4 | 2553 / 60.0 | **2530 / 60.5** [2307 / 66.3] | 2534 / 60.4 [2293 / 66.7] |
| 64 | 4640 / 33.0 | 4135 / 37.0 | 3941 / 38.8 | 4042 / 37.9 | L1 |
| 128 | 8085 / 18.9 | 7148 / 21.4 | 7738 / 19.8 | L1 | L1 |

**3k (384 / device, prefix 52224)**

| q \ k | 128 | 256 | 320 | 512 | 640 |
|---|---|---|---|---|---|
| 32 | 4759 / 49.7 | 4057 / 58.3 | 3970 / 59.5 | 3861 / 61.2 | **3815 / 61.9** [3567 / 66.3] |
| 64 | 4859 / 48.6 | 4329 / 54.6 | 4235 / 55.8 | 4358 / 54.2 | L1 |
| 128 | 8429 / 28.0 | 7445 / 31.7 | 8205 / 28.8 | L1 | L1 |

**4k (512 / device, prefix 49152)**

| q \ k | 128 | 256 | 320 | 512 | 640 |
|---|---|---|---|---|---|
| 32 | 6101 / 49.2 | 5236 / 57.3 | 4984 / 60.2 | **4727 / 63.5** [4612 / 65.1] | 4859 / 61.8 [4588 / 65.4] |
| 64 | 5915 / 50.7 | 5193 / 57.8 | 4874 / 61.6 | L1 | L1 |
| 128 | 8134 / 36.9 | 7241 / 41.5 | 7840 / 38.3 | L1 | L1 |

### Best config per chunk

| chunk | / device | best q/k | full us | compute-only us | DM cost | util full | util compute-only | tracy PM FPU | time vs 5k | ideal vs 5k |
|---|---|---|---|---|---|---|---|---|---|---|
| 5k | 640 | 32 / 640 | 5877 | 5784 | +1.6% | 67.0 | 68.1 | 67.1 | 1.00 | 1.00 |
| 4k | 512 | 32 / 512 | 4727 | 4612 | +2.5% | 63.5 | 65.1 | 63.8 | 0.80 | 0.76 |
| 3k | 384 | 32 / 640 | 3815 | 3567 | +7.0% | 61.9 | 66.3 | 62.2 | 0.65 | 0.60 |
| 2k | 256 | 32 / 512 | 2530 | 2307 | +9.7% | 60.5 | 66.3 | 60.8 | 0.43 | 0.39 |
| 1k | 128 | 32 / 128 | 2473 | 2348 | +5.3% | 30.6 | 32.3 | 30.7 | 0.42 | 0.19 |

(`DM cost` = full / compute-only - 1 at the same q/k. `ideal vs 5k` = tracy `PM IDEAL` ratio: 3001, 2363,
1530, 758 vs 3939 us.)

Tracy `CORE COUNT` is 114 (110 SDPA + 4 AG workers) for every chunk: it counts *launched* cores, not
cores with work, so it cannot show the idle cores at 1k.

### Findings

- **Compute stays efficient down to 2k.** Compute-only util holds at 65-68% from 5k to 2k; the loss
  is data movement, which grows from 1.6% at 5k to ~10% at 2k (less compute per K/V chunk to
  hide the DRAM reads and the all-gather behind). Full util: 67.0 → 63.5 → 61.9 → 60.5.
- **1k falls off a cliff.** 16 heads x 128/32 = 64 Q work units cannot fill 110 cores, so even
  compute-only reaches only 42% (best case, k640). In full mode the exposed DM makes large K chunks
  worse (k640: 3010 us vs 1803 compute-only, +67%), so the best full config is k128, and 1k takes
  about as long as 2k (2473 vs 2530 us) for half the work.
- **q32 wins everywhere.** q64/q128 lose from 2k down; q128 is 3-4x slower (no multi-Q rotation
  benefit and larger per-core serial work). k ≥ 1024 always overflows L1.
- **Floor candidate: 2k global (256 / device)** at q32/k512, -6.5 util points vs 5k. 3k/4k sit in
  between. Below 2k, keeping 110 cores busy would need a K-split (flash-decoding style) or more
  heads per device.

Candidate `MLA_SDPA_CONFIG` entries (not added yet): 512 → q32/k512, 384 → q32/k640, 256 → q32/k512,
128 → q32/k128. Today these lengths fall back to q32/k32.
