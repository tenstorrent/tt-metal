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

## Step 4 — KV-prefix sweep per chunk size (best q/k each)

`test_ring_mla_prefix_sweep`. Prefix targets 0, 2k, 4k, 8k, 16k, 32k, 50k, 64k, 100k, 128k, 192k and 256k,
each rounded to a whole number of chunks. Realtime profiler, median of 10 replays. The ideal time counts
the rectangle plus the causal half; util is measured against all 110 SDPA cores.

Core columns: the SDPA grid is 11x10 (110 cores) plus 4 fused-AG workers (114 launched; tracy
`CORE COUNT`). Active cores = min(110, 16 heads x chunk_local / q), from the op's work split
(`ring_joint_sdpa_program_factory.cpp:1464`). This is derived, not measured, and the prefix does
not change it.

An empty prefix needs one spare chunk of cache capacity, because ring_mla requires Q.seq < K.seq. The
model's cache is always larger than the first chunk.

### 1k chunk (128 / device, q32/k128)

| KV prefix | grid | SDPA cores | + AG | Q units | active cores | ideal us | full us | util % | compute-only us | util % | DM cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0k | 11x10 | 110 | 4 (114) | 64 | 64 | 8 | 194 | 3.9 | 53 | 14.1 | +265% |
| 2k | 11x10 | 110 | 4 (114) | 64 | 64 | 38 | 260 | 14.4 | 144 | 26.0 | +80% |
| 4k | 11x10 | 110 | 4 (114) | 64 | 64 | 68 | 332 | 20.4 | 237 | 28.5 | +40% |
| 8k | 11x10 | 110 | 4 (114) | 64 | 64 | 128 | 507 | 25.2 | 420 | 30.4 | +21% |
| 16k | 11x10 | 110 | 4 (114) | 64 | 64 | 248 | 881 | 28.1 | 788 | 31.4 | +12% |
| 32k | 11x10 | 110 | 4 (114) | 64 | 64 | 488 | 1631 | 29.9 | 1522 | 32.0 | +7% |
| 50k | 11x10 | 110 | 4 (114) | 64 | 64 | 758 | 2473 | 30.6 | 2348 | 32.3 | +5% |
| 64k | 11x10 | 110 | 4 (114) | 64 | 64 | 968 | 3128 | 30.9 | 2991 | 32.4 | +5% |
| 100k | 11x10 | 110 | 4 (114) | 64 | 64 | 1508 | 4813 | 31.3 | 4645 | 32.5 | +4% |
| 128k | 11x10 | 110 | 4 (114) | 64 | 64 | 1928 | 6123 | 31.5 | 5932 | 32.5 | +3% |
| 192k | 11x10 | 110 | 4 (114) | 64 | 64 | 2888 | 9132 | 31.6 | 8875 | 32.5 | +3% |
| 256k | 11x10 | 110 | 4 (114) | 64 | 64 | 3849 | 12113 | 31.8 | 11815 | 32.6 | +3% |

### 2k chunk (256 / device, q32/k512)

| KV prefix | grid | SDPA cores | + AG | Q units | active cores | ideal us | full us | util % | compute-only us | util % | DM cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0k | 11x10 | 110 | 4 (114) | 128 | 110 | 30 | 419 | 7.2 | 191 | 15.7 | +120% |
| 2k | 11x10 | 110 | 4 (114) | 128 | 110 | 90 | 420 | 21.4 | 187 | 48.1 | +125% |
| 4k | 11x10 | 110 | 4 (114) | 128 | 110 | 150 | 536 | 28.0 | 292 | 51.5 | +84% |
| 8k | 11x10 | 110 | 4 (114) | 128 | 110 | 270 | 698 | 38.7 | 468 | 57.7 | +49% |
| 16k | 11x10 | 110 | 4 (114) | 128 | 110 | 510 | 1059 | 48.2 | 822 | 62.0 | +29% |
| 32k | 11x10 | 110 | 4 (114) | 128 | 110 | 990 | 1656 | 59.8 | 1532 | 64.7 | +8% |
| 50k | 11x10 | 110 | 4 (114) | 128 | 110 | 1530 | 2528 | 60.5 | 2307 | 66.3 | +10% |
| 64k | 11x10 | 110 | 4 (114) | 128 | 110 | 1951 | 3168 | 61.6 | 2949 | 66.2 | +7% |
| 100k | 11x10 | 110 | 4 (114) | 128 | 110 | 3031 | 4754 | 63.8 | 4537 | 66.8 | +5% |
| 128k | 11x10 | 110 | 4 (114) | 128 | 110 | 3871 | 5921 | 65.4 | 5782 | 67.0 | +2% |
| 192k | 11x10 | 110 | 4 (114) | 128 | 110 | 5792 | 8894 | 65.1 | 8608 | 67.3 | +3% |
| 256k | 11x10 | 110 | 4 (114) | 128 | 110 | 7713 | 11736 | 65.7 | 11439 | 67.4 | +3% |

### 3k chunk (384 / device, q32/k640)

| KV prefix | grid | SDPA cores | + AG | Q units | active cores | ideal us | full us | util % | compute-only us | util % | DM cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0k | 11x10 | 110 | 4 (114) | 192 | 110 | 68 | 620 | 10.9 | 350 | 19.3 | +77% |
| 3k | 11x10 | 110 | 4 (114) | 192 | 110 | 203 | 947 | 21.4 | 473 | 42.9 | +100% |
| 9k | 11x10 | 110 | 4 (114) | 192 | 110 | 473 | 1161 | 40.7 | 834 | 56.7 | +39% |
| 15k | 11x10 | 110 | 4 (114) | 192 | 110 | 743 | 1461 | 50.9 | 1215 | 61.1 | +20% |
| 33k | 11x10 | 110 | 4 (114) | 192 | 110 | 1553 | 2737 | 56.8 | 2412 | 64.4 | +13% |
| 51k | 11x10 | 110 | 4 (114) | 192 | 110 | 2363 | 3815 | 61.9 | 3567 | 66.3 | +7% |
| 63k | 11x10 | 110 | 4 (114) | 192 | 110 | 2903 | 4680 | 62.0 | 4377 | 66.3 | +7% |
| 99k | 11x10 | 110 | 4 (114) | 192 | 110 | 4524 | 6875 | 65.8 | 6730 | 67.2 | +2% |
| 129k | 11x10 | 110 | 4 (114) | 192 | 110 | 5874 | 8862 | 66.3 | 8700 | 67.5 | +2% |
| 192k | 11x10 | 110 | 4 (114) | 192 | 110 | 8710 | 12967 | 67.2 | 12780 | 68.2 | +1% |
| 255k | 11x10 | 110 | 4 (114) | 192 | 110 | 11546 | 17245 | 67.0 | 16955 | 68.1 | +2% |

### 4k chunk (512 / device, q32/k512)

| KV prefix | grid | SDPA cores | + AG | Q units | active cores | ideal us | full us | util % | compute-only us | util % | DM cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0k | 11x10 | 110 | 4 (114) | 256 | 110 | 120 | 774 | 15.5 | 376 | 31.9 | +106% |
| 4k | 11x10 | 110 | 4 (114) | 256 | 110 | 360 | 1022 | 35.2 | 730 | 49.3 | +40% |
| 8k | 11x10 | 110 | 4 (114) | 256 | 110 | 600 | 1309 | 45.8 | 1081 | 55.5 | +21% |
| 16k | 11x10 | 110 | 4 (114) | 256 | 110 | 1080 | 1978 | 54.6 | 1788 | 60.4 | +11% |
| 32k | 11x10 | 110 | 4 (114) | 256 | 110 | 2041 | 3297 | 61.9 | 3200 | 63.8 | +3% |
| 48k | 11x10 | 110 | 4 (114) | 256 | 110 | 3001 | 4727 | 63.5 | 4612 | 65.1 | +2% |
| 64k | 11x10 | 110 | 4 (114) | 256 | 110 | 3961 | 6221 | 63.7 | 6033 | 65.7 | +3% |
| 100k | 11x10 | 110 | 4 (114) | 256 | 110 | 6122 | 9420 | 65.0 | 9216 | 66.4 | +2% |
| 128k | 11x10 | 110 | 4 (114) | 256 | 110 | 7803 | 11833 | 65.9 | 11692 | 66.7 | +1% |
| 192k | 11x10 | 110 | 4 (114) | 256 | 110 | 11644 | 17543 | 66.4 | 17350 | 67.1 | +1% |
| 256k | 11x10 | 110 | 4 (114) | 256 | 110 | 15485 | 23312 | 66.4 | 23008 | 67.3 | +1% |

### 5k chunk (640 / device, q32/k640)

| KV prefix | grid | SDPA cores | + AG | Q units | active cores | ideal us | full us | util % | compute-only us | util % | DM cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0k | 11x10 | 110 | 4 (114) | 320 | 110 | 188 | 1018 | 18.4 | 554 | 33.9 | +84% |
| 5k | 11x10 | 110 | 4 (114) | 320 | 110 | 563 | 1400 | 40.2 | 1079 | 52.2 | +30% |
| 10k | 11x10 | 110 | 4 (114) | 320 | 110 | 938 | 1848 | 50.8 | 1600 | 58.6 | +15% |
| 15k | 11x10 | 110 | 4 (114) | 320 | 110 | 1313 | 2266 | 57.9 | 2123 | 61.8 | +7% |
| 30k | 11x10 | 110 | 4 (114) | 320 | 110 | 2438 | 3784 | 64.4 | 3691 | 66.1 | +3% |
| 50k | 11x10 | 110 | 4 (114) | 320 | 110 | 3939 | 5881 | 67.0 | 5784 | 68.1 | +2% |
| 65k | 11x10 | 110 | 4 (114) | 320 | 110 | 5064 | 7451 | 68.0 | 7350 | 68.9 | +1% |
| 100k | 11x10 | 110 | 4 (114) | 320 | 110 | 7690 | 11144 | 69.0 | 11017 | 69.8 | +1% |
| 130k | 11x10 | 110 | 4 (114) | 320 | 110 | 9941 | 14298 | 69.5 | 14156 | 70.2 | +1% |
| 190k | 11x10 | 110 | 4 (114) | 320 | 110 | 14442 | 20614 | 70.1 | 20433 | 70.7 | +1% |
| 255k | 11x10 | 110 | 4 (114) | 320 | 110 | 19319 | 27458 | 70.4 | 27234 | 70.9 | +1% |

### Cross-chunk summary (full-mode util %)

| chunk | active cores | 0k | ~16k | ~32k | ~50k | ~100k | ~256k |
|---|---|---|---|---|---|---|---|
| 1k | 64 | 3.9 | 28.1 | 29.9 | 30.6 | 31.3 | 31.8 |
| 2k | 110 | 7.2 | 48.2 | 59.8 | 60.5 | 63.8 | 65.7 |
| 3k | 110 | 10.9 | 50.9 | 56.8 | 61.9 | 65.8 | 67.0 |
| 4k | 110 | 15.5 | 54.6 | 61.9 | 63.5 | 65.0 | 66.4 |
| 5k | 110 | 18.4 | 57.9 | 64.4 | 67.0 | 69.0 | 70.4 |

### Findings

- **Every chunk ≥ 2k converges to the same ceiling.** At ≥ 128k prefix, full util is 65-67% and the
  DM cost is 1-3%, matching compute-only (67-68%). Larger chunks get there sooner: DM cost falls
  below 5% at ~32k for 4k chunks, ~100k for 3k chunks, and ~100-128k for 2k chunks.
- **Short prefixes are latency bound at every size.** At 0k prefix the per-call floor is 194 / 419 /
  620 / 774 us (1k / 2k / 3k / 4k) against 8-120 us of ideal work. Compute-only is also far from ideal
  there (14-32% util): each call has a fixed per-ring-iteration and per-Q-chunk cost.
- **1k is capped by occupancy, not DM.** With 64 of 110 cores active, compute-only saturates at 32.5%
  (≈ 64/110 x the 2k-plus compute-only ceiling), and full mode tracks it (+3-5% DM from 50k up).
  At any prefix, 1k is never cheaper than half of 2k: at 256k it takes 12113 us, vs 11736 us for 2k
  at twice the work.
- **2k is the smallest chunk that keeps all 110 cores busy.** At a production-length prefix (≥ 50k)
  it is within ~2-6 util points of 4k/5k.
