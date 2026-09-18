# MiniMax-H3 t2va on Wormhole Galaxy (4x8, 32 chips) — perf sweep

Re-measured 2026-09-17 on `tt-metal` @ `eab3dfbd599` (Wormhole bringup + the fused MM/RS
gate fix). Supersedes the 2026-09-16 run at `3a016b74847`, which stalled at 13/18.
Mesh param `MESH_4X8_RING_WH` (`4x8nl4`), TP=4 axis 0 / SP=8 axis 1, Ring, 4 links.
50 scheduler steps => 49 forwards. `RUN_VBENCH=0` (CLIP still gated).

Raw logs are **not** committed (too large to be useful in-tree); they were kept at
`~/h3_wormhole_results/*.log.gz` on the run host, with `parse.py` there to regenerate
these tables from any of them. Current run: `sweep_fixed.log.gz`.

## Command

```bash
TT_DIT_CACHE_DIR=~/tt_dit_cache MINIMAX_H3_DIT_FSDP=1 RUN_VBENCH=0 \
  python -m pytest models/tt_dit/tests/models/minimax_h3/test_pipeline_minimax_h3.py \
  -k "4x8nl4" -q
```

Drop `RUN_VBENCH=0` for the VBench gate (verified working, see below).
Drop `MINIMAX_H3_DIT_FSDP=1` for the unsharded baseline.

## Headline

DiT FSDP is the fix for the memory limits. Without it only 5 s fits; with it 10 s and 15 s run.

| | DiT alloc/bank | free/bank | largest contig |
|---|---|---|---|
| FSDP off | 799.5 MiB | 221.7 MiB | 221.7 MiB |
| FSDP on  | **101.8 MiB** | **919.4 MiB** | **917.7 MiB** |

7.85x reduction (SP=8 sharding), costs 5-11% denoise time, **bit-identical output**
(CLIP equal to 2 dp on all six 5 s cases).

Sweep outcome: **18/18 passed** (3 h 14 m, zero failures). The previous run reached 13/18
before an intermittent device hang blocked the rest; that hang was root-caused to the fused
MM/RS gate (open issue 2) and the 5 blocked points now all pass.

Removing the accidental fused ff2 path also made every case **2.0-4.0% faster (mean 3.0%)**.
Per-forward, old -> new: 5 s 2839 -> 2754 (21:9), 1477 -> 1417 (1:1); 10 s 6588 -> 6422 (16:9);
15 s 12700 -> 12447 (21:9). The six cases that never completed before are new measurements.

## Timings — FSDP ON (seconds unless noted)

### 5 s / 124 frames

| aspect | canvas | MPix | cold | warm | enc | denoise | vae | audio | ms/fwd | realtime | CLIP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 21:9 | 1536x672 | 1.03 | 160.9 | 146.6 | 3.5 | 134.9 | 5.8 | 2.4 | 2754 | 28.4x | 35.95 |
| 16:9 | 1344x768 | 1.03 | 161.3 | 147.6 | 3.4 | 135.3 | 5.1 | 3.7 | 2761 | 28.6x | 37.42 |
| 9:16 | 768x1344 | 1.03 | 161.0 | 146.7 | 3.4 | 134.4 | 5.7 | 3.1 | 2744 | 28.4x | 37.02 |
| 4:3 | 1024x768 | 0.79 | 120.4 | 104.9 | 3.4 | 94.0 | 3.7 | 3.8 | 1919 | 20.3x | 37.26 |
| 3:4 | 768x1024 | 0.79 | 118.7 | 105.3 | 3.5 | 94.2 | 4.0 | 3.5 | 1922 | 20.4x | 36.54 |
| 1:1 | 768x768 | 0.59 | 92.7 | 79.3 | 3.5 | 69.4 | 3.0 | 3.4 | 1417 | 15.3x | 36.32 |

### 10 s / 243 frames

| aspect | canvas | cold | warm | enc | denoise | vae | audio | ms/fwd | realtime | CLIP |
|---|---|---|---|---|---|---|---|---|---|---|
| 21:9 | 1536x672 | 349.0 | 332.8 | 3.5 | 314.2 | 11.6 | 3.5 | 6413 | 32.9x | 33.84 |
| 16:9 | 1344x768 | 344.6 | 333.4 | 3.5 | 314.7 | 11.0 | 4.4 | 6422 | 32.9x | 36.92 |
| 9:16 | 768x1344 | 344.7 | 332.8 | 3.5 | 315.0 | 10.6 | 3.8 | 6428 | 32.9x | 36.40 |
| 4:3 | 1024x768 | 255.7 | 240.5 | 3.5 | 226.3 | 7.5 | 3.2 | 4618 | 23.8x | 37.09 |
| 3:4 | 768x1024 | 252.6 | 241.3 | 3.5 | 226.5 | 7.4 | 3.8 | 4623 | 23.8x | 36.89 |
| 1:1 | 768x768 | 173.8 | 159.2 | 3.5 | 146.4 | 5.8 | 3.6 | 2988 | 15.7x | 37.54 |

### 15 s / 362 frames

| aspect | canvas | cold | warm | enc | denoise | vae | audio | ms/fwd | realtime | CLIP |
|---|---|---|---|---|---|---|---|---|---|---|
| 21:9 | 1536x672 | 648.5 | 633.2 | 3.4 | 609.9 | 16.0 | 3.9 | 12447 | 42.0x | 35.27 |
| 16:9 | 1344x768 | 644.3 | 633.1 | 3.5 | 609.3 | 15.8 | 4.6 | 12435 | 42.0x | 36.31 |
| 9:16 | 768x1344 | 645.7 | 633.8 | 3.5 | 610.7 | 15.2 | 4.3 | 12464 | 42.0x | 35.50 |
| 4:3 | 1024x768 | 457.3 | 434.6 | 3.3 | 416.0 | 10.9 | 4.3 | 8491 | 28.8x | 36.01 |
| 3:4 | 768x1024 | 447.6 | 434.9 | 3.3 | 416.6 | 11.2 | 3.8 | 8501 | 28.8x | 36.40 |
| 1:1 | 768x768 | 281.8 | 266.6 | 3.5 | 251.2 | 8.5 | 3.6 | 5126 | 17.7x | 38.26 |

## Timings — FSDP OFF (baseline, for the 5 s comparison)

| aspect | canvas | cold | warm | denoise | ms/fwd | realtime |
|---|---|---|---|---|---|---|
| 21:9 | 1536x672 | 155.1 | 145.1 | 132.3 | 2699 | 28.1x |
| 16:9 | 1344x768 | 156.2 | 143.3 | 132.0 | 2695 | 27.7x |
| 9:16 | 768x1344 | 157.1 | 144.6 | 131.8 | 2690 | 28.0x |
| 4:3  | 1024x768 | 136.1 | 101.0 |  91.1 | 1858 | 19.6x |
| 3:4  | 768x1024 | 113.5 | 102.0 |  91.0 | 1856 | 19.8x |
| 1:1  |  768x768 | 108.8 |  76.0 |  65.3 | 1333 | 14.7x |

10 s without FSDP: 21:9 / 16:9 / 9:16 OOMed in **warmup**; 4:3 (cold 456.5) /
1:1 (191.5) / 3:4 (257.5) completed one generation then OOMed reloading the DiT for
the **timed** pass. All six 15 s OOMed in warmup.

## Scaling notes

- Denoise is 87-96% of warm time and tracks **pixel area**, not aspect: the three
  1.03 MPix 5 s canvases agree within 0.9 s regardless of orientation (134.4/134.9/135.3),
  and the same holds at 15 s (609.3/609.9/610.7).
- ms/forward vs area at 5 s: 1417 (0.59) / 1920 (0.79) / 2753 (1.03) — near-linear.
- Duration scales **superlinearly**, now measurable across all three durations:
  16:9 denoise 135.3 (5 s) -> 314.7 (10 s) -> 609.3 (15 s); 21:9 134.9 -> 314.2 -> 609.9.
  That is 2.33x for the first 1.96x in frames and 1.94x for the next 1.49x — the
  superlinearity is real but milder than the earlier two-point estimate suggested.
- Realtime factor improves with duration (28.4x at 5 s -> 42.0x at 15 s for 1.03 MPix):
  the fixed ~3.5 s encoder and the cold-start step amortise over more frames.
- Encoder is a flat ~3.5 s everywhere (same prompt, cached weights).
- The 4:3/10s audio-decode outlier of 192.6 s in the non-FSDP run dropped to 3.1 s
  with FSDP — it was memory pressure, not the conv1d MAC fallback.

## Per-op device breakdown — one transformer block, 15 s / 16:9 (Tracy)

Measured 2026-09-17 on `bc1d99d05f6`: Wormhole `GALAXY_RING` rows, the re-keyed 15 s ff1/ff2
blockings, and `_packed_sizes` producing the length the pipeline runs. One block, warm iteration
between `start`/`stop` signposts, both FSDP settings.

```bash
scripts/run_safe_pytest.sh --profile \
  "'models/tt_dit/tests/models/minimax_h3/test_performance_minimax_h3.py::test_minimax_h3_transformer_block_perf[wormhole_b0-sp_sim1-15s_768p-4x8sp1tp0nl4_ring_is_fsdp1]'" \
  -s --timeout 3600
python models/tt_dit/tests/models/minimax_h3/tools/project_block_perf.py fsdp1=<csv> fsdp0=<csv>
```

Two quoting/config traps, both load-bearing:
- `tools/tracy/__main__.py:368` joins argv with spaces and re-execs under `shell=True`, so a
  `-k "a and b"` is word-split and never reaches pytest. Pass a node id (space-free), and wrap it
  in embedded single quotes so the re-shell cannot glob the `[...]`.
- `pytest.ini` sets `timeout = 300`, too short for the 15 s shape. Override with `--timeout 3600`.

Shape, as the pipeline packs it and as the test logs it: 1344x768, 362 frames -> 107 latent frames
x 24x42 patches = 107856 video rows + 603 audio latents x 2 channels = 1206 audio rows + 39 text
tokens = seq_len 109101, padded to `sp_factor * TILE` = 256 -> 109312, **13664 rows/device** at
SP=8. Block input shard 13664 x 1344 bf16 = 36.7 MB/device.

| op | calls | fsdp1 ms | fsdp0 ms | delta |
|---|---|---|---|---|
| **RingJointSDPADeviceOperation** | 1 | **174.56** | **174.57** | -0.01 |
| AllGatherMinimalMatmulAsyncOp | 3 | 32.25 | 31.61 | +0.64 |
| EmbeddingsDeviceOperation | 6 | 10.41 | 10.35 | +0.06 |
| MinimalMatmulDeviceOperation | 2 | 8.54 | 8.55 | -0.02 |
| AllBroadcastDeviceOperation | 1 | 3.99 | — | +3.99 |
| AllGatherAsyncDeviceOperation | 4 | 3.96 | — | +3.96 |
| DitFusedDistributedRmsnorm | 4 | 2.99 | 2.64 | +0.35 |
| ReduceScatterMinimalAsync | 1 | 2.78 | 2.78 | 0.00 |
| UntilizeWithUnpadding | 21 | 1.76 | 0.07 | +1.70 |
| ConcatDeviceOperation | 1 | 1.51 | — | +1.51 |
| TilizeWithValPadding | 8 | 1.35 | 0.06 | +1.30 |
| *(remaining 8 ops)* | | 2.81 | 2.78 | +0.04 |
| **device only** | | **246.92** | **233.41** | **+13.51 (5.8%)** |
| **device + op gap** | | **250.80** | **238.46** | +12.34 |
| SDPA share of block | | 70.7% | 74.8% | |

Projected over 50 layers: **12.35 s/step** device-only, 12.54 s with op gaps (617 / 627 s per
50-step video). The 16:9/15 s row above measures the same configuration end-to-end at
**12435 ms/forward**, which lands inside that bracket exactly as `project_block_perf.py` intends:
`device only` is the underestimate (no dispatch gaps) at 99.3% of the forward, `device + op gap`
the overestimate at 100.8%. The 50-layer block stack is therefore the whole forward to within the
bracket's width; the refiner, input projections, `norm_out` and output heads fit in the remainder.

### Findings

1. **Ring SDPA is 71% of the block** at 48.3% FPU utilization (`PM FPU UTIL (%)`, 47.5-48.7 across
   all 32 devices). Nothing else is close: the three fused all-gather+matmuls are 13%, the six adaLN
   embedding gathers 4%, the two plain matmuls 3.5%.
2. **FSDP costs 5.8%** (13.51 ms/block) — inside the 5-11% the pipeline sweep saw. Three ops exist
   only with FSDP on: AllBroadcast 3.99 + AllGatherAsync 3.96 + Concat 1.51 = 9.46 ms. The rest is
   layout conversion (+3.00, next item), a slightly larger fused AG+matmul (+0.64) and RMSNorm (+0.35).
3. **Layout conversions grow 24x under FSDP** — tilize + untilize go 0.13 -> 3.12 ms, 22% of the
   FSDP cost spent on format round-trips rather than communication. Cheapest apparent win.
4. **ff2 runs the intended Wormhole path**: `ReduceScatterMinimalAsyncDeviceOperation` is present
   and there is no fused `Matmul_RS` row.

### SDPA chunk sizes: already optimal at 15 s

`test_ring_joint_attention_create_perf_table[minimax_h3_15s_768p]` at 13664 rows/device, sweeping
`q in {192, 256} x k in {512, 640, 768, 1024}` (run plain — it self-shells `run_device_profiler`,
so wrapping it in `--profile` would nest profilers):

| rank | q_chunk | k_chunk | duration | iters/core | pad waste | slot waste | FPU util | math util |
|---|---|---|---|---|---|---|---|---|
| 1 | **256** | **512** | **175.269 ms** | 2592 | 2.3% | 0.0% | 47.3-47.7% | 35.1% |
| 2 | 192 | 640 | 184.930 ms | 2816 | 4.1% | 0.0% | 44.9-46.1% | 33.2% |
| 3 | 192 | 512 | 185.212 ms | 3456 | 2.3% | 0.0% | 44.8-45.7% | 33.2% |
| — | 192 | 768 / 1024 | L1 infeasible | | | | | |
| — | 256 | 640 / 768 / 1024 | L1 infeasible | | | | | |

`(256, 512)` is what `measured_sdpa_chunk_sizes[13664]` ships, and it wins by 5.5%. The five
infeasible points fail with `Statically allocated circular buffers on core range [0-0 - 6-8] grow to
N B which is beyond max L1 size of 1499136 B` — Wormhole's 1.5 MB/core, on the 7x9 = 63 compute grid
— at N = 1,602,880 (192/768), 1,639,744 (256/640), 1,836,352 (256/768), 1,963,328 (192/1024) and
2,229,568 (256/1024), each matching the L1 envelope calibrated under *Sweeps run* below.

Slot waste is zero at both feasible q: 13664 rows give 54 Q chunks at q=256 (54 x 14 heads = 756 =
12 x 63) and 72 at q=192 (1008 = 16 x 63), so the ranking is decided by per-core efficiency, not
scheduling — and larger q wins, FPU utilization 47.5% against 45.5%. The harness reports
"63 compute + 9 CCL = 72 total cores" and measures SDPA at 175.269 ms against 174.56 ms in-block,
0.4% apart.

So chunk-size tuning at 15 s is exhausted: 3 feasible points, 5 ruled out by L1, and the shipped
config is the best of them. The ~48% FPU / 35% math utilization is **inherent to the ring joint SDPA
kernel at this shape**. At 71% of the block it is the only thing worth attacking, but the work is in
the kernel. Note the contrast with 5 s, where `q=320` wastes 16.7% of the 63 slots and chunk
tuning *does* have headroom.

Caveat: `CORE COUNT` for `RingJointSDPADeviceOperation` reads 71, not 63, because the profiler
counts the fused CCL workers — `ccl_core_grid_offset=(7, 0)` with `use_column_major_ccl=True`
(`attention_minimax_h3.py:572-573`) places them in the reserved last column.

## Roofline — speed of light on this part

Back-of-envelope, derived 2026-09-17 from the shapes above and the part constants the repo already
uses (`tests/nightly/sdpa_perf_utils.py`: Wormhole 1.0 GHz, 2048 FLOP/cycle/core at HiFi2, i.e.
4096 at LoFi halved). Every linear and the ring SDPA run at HiFi2 with bf16 weights, so HiFi2 is the
right peak. Galaxy WH exposes 8x9 = 72 compute cores per chip:

    72 cores x 2048 FLOP/cycle x 1.0 GHz = 147.5 TFLOPS / chip  ->  4.72 PFLOPS / 32-chip Galaxy

Per token per layer, with hidden 5376, inner 7168 (56 x 128), SwiGLU ffn 14336 and full joint
attention over the packed sequence `S`:

    dense     = 2 x (5376x21504 + 7168x5376 + 5376x28672 + 14336x5376) = 0.771 GFLOP  (385 M params/layer)
    attention = 4 x S x 7168                                            = 28.7 kFLOP x S

| duration | padded S | attention share | FLOP / forward | FLOP / device / layer |
|---|---|---|---|---|
| 5 s | 37888 | 58% | 3.52 PFLOP | 2.20 TFLOP |
| 10 s | 73472 | 73% | 10.57 PFLOP | 6.61 TFLOP |
| 15 s | 109312 | 80% | 21.34 PFLOP | 13.34 TFLOP |

Dividing by the Galaxy peak gives the floor per forward; the measured column is the FSDP-on 16:9
row of each duration's table above. "RT" is the realtime factor (denoise seconds per video
second, lower is better), over 49 forwards.

| duration | roofline ms/fwd | measured ms/fwd | achieved FPU util | headroom | denoise @100% | @70% | @50% | measured |
|---|---|---|---|---|---|---|---|---|
| 5 s | **746** | 2761 | 27% | 3.7x | 37 s (7.1x RT) | 52 s (10x) | 73 s (14x) | 135 s (26x) |
| 10 s | **2240** | 6422 | 35% | 2.9x | 110 s (10.8x) | 157 s (15.5x) | 220 s (22x) | 315 s (31x) |
| 15 s | **4523** | 12435 | 36% | 2.75x | 222 s (14.7x) | 317 s (21x) | 443 s (29x) | 609 s (40x) |

The 70% tier is the realistic ceiling: `tech_reports/GEMM_FLOPS/GEMM_FLOPS.md` has well-tuned
Wormhole matmuls at 80-93% of HiFi2 peak, and a full block also carries norms, embeddings, layout
conversions and collectives that do no FLOPs.

### Cross-check against the Tracy block breakdown (15 s, per device, per layer)

Same arithmetic per op, on the grid each op actually runs on:

| op | FLOP | min at HiFi2 peak on its grid | measured | util |
|---|---|---|---|---|
| RingJointSDPA (63 cores) | 10.71 T | 83.0 ms | 174.6 ms | **48%** — Tracy's `PM FPU UTIL` reads 47.5-48.7% |
| ff1 AGMM (8x8) | 1.05 T | 8.0 ms | 15.7 ms | 51% |
| qkv AGMM (8x8) | 0.79 T | 6.0 ms | 10.4 ms | 58% |
| to_out AGMM (8x8) | 0.26 T | 2.0 ms | 4.3 ms | 46% |
| ff2 matmul (8x9) | 0.53 T | 3.6 ms | 6.8 ms | 53% |
| **block** | **13.34 T** | **90 ms** (72 cores) | **247 ms** | **36%** |

The derived SDPA utilization lands on the profiler's FPU-utilization counter exactly, which
validates both the FLOP count and the 2048 FLOP/cycle/core constant.

### Nothing but the FPU binds

Per device per layer at 15 s, against 4 links x 12.5 GB/s = 50 GB/s of ring ingress
(`tech_reports/EthernetMultichip/BasicEthernetGuide.md`):

| traffic | bytes | time at 50 GB/s | overlaps |
|---|---|---|---|
| KV ring all-gather (14 heads x 7/8 x S x 128 x 2 x bf16) | 686 MB | ~14 ms | 175 ms of SDPA |
| three AGMM activation gathers | ~370 MB | ~7 ms | 30 ms of matmul |
| FSDP weight gather (7/8 x 193 MB bf16) | ~170 MB | ~3.4 ms | (measured FSDP cost 13.5 ms is mostly the extra ops + layout, not the bytes) |
| DRAM: 193 MB weights + ~1-2 GB activation passes at 288 GB/s | | < 10 ms | everything |

Even at 100% FPU the collectives sit under compute by 5-10x. The floor is the matrix engine.

### What this says about the tuning below

1. Speed of light at HiFi2 is ~2.75x today's 15 s forward and ~3.7x the 5 s one; a realistic
   70%-util target is ~6.5 s/fwd at 15 s (317 s, 21x RT) and ~1.1 s/fwd at 5 s (52 s, 10x RT).
2. Ring SDPA is 71-80% of the FLOPs at 48% util. Chunk tuning is exhausted (below), so the
   remaining ~2x on that op is kernel work — `use_exp_ring_sdpa` (experiment 7) is the one
   untried lever.
3. 5 s is occupancy-limited, not FLOP-limited: 27% util against 36% at 15 s, from SDPA slot
   waste (q=320 idles 16.7% of the 63 slots) and small per-device M in the matmuls.
4. Going below HiFi2 (bfp8 / LoFi operands) doubles the ceiling again, to ~2.3 s/fwd at 15 s,
   but that is a quality decision rather than a tuning one.

## Optimization target — 15 s / 768P / 16:9

Tuning work is scoped to this one configuration. Baseline is `c825d089e31` (the per-op breakdown
above), measured at `fsdp1`.

| | warm total | denoise | ms/fwd | realtime | CLIP | block device-only |
|---|---|---|---|---|---|---|
| baseline, other host (`c825d089e31` tables) | 633.1 s | 609.3 s | 12435 | 42.0x | 36.31 | 246.31 ms |
| baseline, **this host**, tuned entries disabled | 612.9 s | 590.9 s | 12058.3 | 40.6x | — | **TODO** |
| **best found**, this host, `a07012d7d8a` | **612.2 s** | **587.4 s** | **11988.4** | 40.6x | — | **TODO** |
| best found, this host, later run (rebuilt weight cache) | 621.0 s | 599.2 s | 12230 | 41.2x | **35.88** (min 34.69, bar 33.0) | **TODO** |

Same host, same weights, same everything, one run each, 2026-09-17: the landed ff1 + ff2 blockings
are worth **-69.9 ms/fwd, -0.58%** (steady 12057 -> 11990 ms/step; denoise 590.9 -> 587.4 s). The
isolated sweep predicted 1250.5 us/layer x 50 = 62.5 ms/step; measured 67-70. The prediction holds.

Read the three rows carefully: the other-host baseline is ~3% slower on this shape than this host's
own baseline (12435 vs 12058 ms/fwd) with identical code paths, so comparing the after-run against
the doc's tables would have claimed 3.6% -- host, not the fix. Only the same-host pair is a
measurement of the change. Total compute moved just 0.7 s because VAE decode varied +2.8 s between
the two runs, which the DiT blockings cannot touch; ms/fwd is the metric that isolates them.

CLIP was not computed on the two A/B runs. The fourth row is the same code run later with a rebuilt
weight cache (verified with `TT_DIT_CACHE_VERIFY=1` -- `verify_saved_model` in `utils/cache.py`) and carries the
CLIP: **35.88** against the other host's 36.31 at baseline. Its 12230 ms/fwd is 2.0% off the A/B pair
taken four hours earlier across two board resets; the pair was back-to-back and differs by 0.58%, so it
remains the measurement of the blockings and the 2% is run-to-run / board-state spread. **Block
device-only is TODO** because the per-op re-profile needs `test_performance_minimax_h3.py` under
Tracy, which imports the pinned `diffusers` fork this host does not have.

Cold (first-generation) figures are deliberately not tabulated: `TT_DIT_CACHE_DIR` did not exist on
this host, so the warmup built the sharded-tensor cache from 62 GB of safetensors on top of every
compile (1053.9 s, 234 s of it cold audio decode) -- the log labels it "not a perf target".

**Both blockings are numerically validated.** The block-size sweep measures timing only (no PCC
anywhere in `sweep_mm_block_sizes.py`), so a fast-but-wrong blocking would have looked like a winner;
this was checked after the fact, at M=13664 exactly, on this hardware:

| op | check | landed (8, 7, 10) | replaced | bar |
|---|---|---|---|---|
| ff2 | plain `minimal_matmul`, 1 device, vs torch fp32 | pcc 1.0000000; vs (8,8,8): max diff 0.0156 (one bf16 ulp at 3.4), mean diff 0.0 | (8, 8, 8) pcc 1.0000000 | — |
| ff1 | Wan2.2 AGMM harness, 4-device ring, fused SwiGLU, bias=False | pcc 0.9999843, rel-RMSE 0.00837 | (8, 3, 14) pcc 0.9999844, rel-RMSE 0.00837 | pcc > 0.9995, rmse < 0.02 |

ff2's per-tile-column difference is ~1e-8 and flat across N -- the partial trailing block of 9 tiles is
handled correctly. ff1 matches the Blackhole default to six decimals. Scripts: `ff2_pcc.py`,
`ff1_pcc.py` (session scratch; ~60 lines each, worth folding into the sweep as a post-check).

### Per-op: baseline vs best

| op | baseline ms | best ms | delta |
|---|---|---|---|
| RingJointSDPADeviceOperation | 172.52 | TODO | TODO |
| AllGatherMinimalMatmulAsyncOp (3) | 33.11 | TODO | TODO |
| MinimalMatmulDeviceOperation (2) | 8.84 | TODO | TODO |
| *(all others)* | 31.84 | TODO | TODO |
| **device only** | **246.31** | **TODO** | **TODO** |

## Sweeps run

All matmul numbers come from `models/tt_dit/utils/sweep_mm_block_sizes.py` against device config
`wh_4x8_ring` (4 links, 4 KB router payload, Ring). 4749 measured rows, all durations.

```bash
# One shape. MM_SWEEP_PROFILER_DUMP_EVERY is mandatory on a WH Galaxy -- see Open issues 2.
MM_SWEEP_PROFILER_DUMP_EVERY=100000 python -m pytest \
  models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
  -k "13664_5376_7168_8x8_agmm_ff1_swiglu and wh_4x8_ring" -s
```

Wormhole's compute grid is **8x9 = 72 cores** against Blackhole's 12x10, so the AGMM worker grid is
8x8 (`agmm_worker_grid` reserves the in0-mux row) rather than 12x9, and ff2's plain matmul runs on
the full 8x9. Neither grid is one Blackhole produces, so every H3 blocking the model carried for
these shapes had been swept on a grid that does not exist on the part.

### Matmul blockings — 15 s, M=13664

All four 15 s shapes swept with `sweep_mm_block_sizes.py` on `wh_4x8_ring` at the pipeline's 13664
rows/device: 1419 combos, every one `OK`. "Pre-tuning" is what the model ran before any Wormhole
entry existed -- the (K, N)-keyed `AGMM_BLOCK_SIZES` defaults for the three AGMM shapes, and the
hardcoded (8, 8, 8) for ff2's plain matmul.

| shape | op / grid | combos | pre-tuning | shipped now | rank | best measured | saved vs pre-tuning |
|---|---|---|---|---|---|---|---|
| ff1 | AGMM 8x8 | 320 | (8, 3, 14) 16718.0 us | **(8, 7, 10)** 15709.9 us | 1 | = shipped | **6.0%** |
| ff2 | matmul 8x9 | 322 | (8, 8, 8) 7013.1 us | **(8, 7, 10)** 6770.7 us | 2 | (12, 7, 8) 6668.2 us | **3.5%** (4.9% at best) |
| qkv | AGMM 8x8 | 425 | (8, 7, 12) 10401.8 us | (8, 7, 12) *unchanged* | 2 | (8, 6, 12) 10351.4 us | 0 (0.5% at best) |
| to_out | AGMM 8x8 | 352 | (8, 8, 6) 4332.8 us | (8, 8, 6) *unchanged* | 3 | (14, 8, 6) 4312.8 us | 0 (0.5% at best) |
| **total, as shipped** | | | | | | | **1250.5 us/block = 62.5 ms/step = 0.50% of the 12435 ms forward** |

ff1's landed `(8, 7, 10)` is the sweep winner outright. ff2's landed `(8, 7, 10)` is rank 2:
`(12, 7, 8)` measures 1.5% faster (102.5 us/block, ~5 ms/step) but the sweep is timing-only and
that blocking has not been PCC-validated, so it is **not landed** -- `(8, 7, 10)` was validated at
pcc 1.0000000 and stays. qkv and to_out keep their `AGMM_BLOCK_SIZES` defaults: both are within
0.5% of the best combo, comparable to the ~0.3% run-to-run spread, so neither is worth an entry.

Why the two landed entries are keyed on `(M, K, N)` rather than added to the `(K, N)`-keyed model
table, and how `get_matmul_config` falls back on equal `M_per_core` after an exact miss, is
documented in `models/tt_dit/utils/matmul.py` -- the module docstring and the `grid_88_configs` /
`grid_89_configs` entries.

### Fused MM/RS on Wormhole — stays disabled

ff2 as `minimal_matmul_strided_reduce_scatter_async`, one entry per candidate matmul grid (the
reduce-scatter takes the rows the matmul leaves; at `num_links=4` that is 1 worker/link at 8x7, 2
at 8x6, 3 at 8x5):

| matmul grid | RS workers/link | best |
|---|---|---|
| 8x7 | 1 | 3134.7 us |
| 8x6 | 2 | 3610.2 us |
| 8x5 | 3 | 3996.7 us |

Monotonically worse as the RS zone grows: every core handed to the reduce-scatter costs the matmul
more than it returns. All are far off the 2373.0 us unfused matmul, so `eab3dfbd599` keeping
Wormhole off the fused path is right on tuned configs too, not just against the broken fallback.
Not a like-for-like total — the unfused figure excludes the separate reduce-scatter and addcmul,
leaving them a 762 us budget — but combined with the measured 2.0-4.0% end-to-end gain from
disabling it, the conclusion holds.

### SDPA chunk sizes

Sweeping only q >= 256 with k <= 512 bounds the search on the wrong axis. From the CB allocation in
`ring_joint_sdpa_program_factory.cpp:1296-1308`:

```
q = 8*Sq    k = 8*Sk    v = 8*Sk    mask = Sq*Sk    qk = Sq*Sk
out_im = 4*Sq    out0 = 4*Sq    stats = Sq
```

`Sq*Sk` dominates, but Sq carries the heavier linear term — q, out_im, out0 and the statistics FIFO
all scale with it, against K/V's two buffers on Sk. That is why `(512, 256)` is L1-infeasible while
`(256, 512)` fits at the same product. The untried direction is therefore **smaller q with larger
k**, which also halves the ring K-loop iterations — the property that made k=512 beat k=256.

That prediction was **wrong**, and the widened sweep closes the question. q in {192, 256} x k in
{512, 640, 768, 1024} (q=128 excluded: it hung the op twice on 2026-09-17 on another Wormhole galaxy
at seq_local 13632 / k=512, did not reproduce on `UF-EV-B12-GWH02` the same day -- 6/6 completed on
the same code, under the profiler and the watcher, at 13632 and 13664 -- and is slower than q=192 at
every feasible k regardless):

| rank | q_chunk | k_chunk | duration | iters/core | FPU util | math util |
|---|---|---|---|---|---|---|
| 1 | **256** | **512** | **171.694 ms** | 2592 | 48.1% | 35.6% |
| 2 | 192 | 512 | 190.572 ms | 3456 | 43.3-43.7% | 32.1% |
| 3 | 192 | 640 | 193.960 ms | 2816 | 42.6-42.9% | 31.5% |
| — | 192 | 768 / 1024 | L1 infeasible | | | |
| — | 256 | 640 / 768 / 1024 | L1 infeasible | | | |

Smaller q *does* unlock a larger k — `(192, 640)` builds where `(256, 640)` does not, the first
k > 512 point run on this shape — but it is **13% slower**. Two reasons, both visible above:

  * Larger q chunks are more efficient per core. FPU utilization drops 48.1% -> 42.6% and math
    utilization 35.6% -> 31.5% going from q=256 to q=192.
  * "Larger k halves the ring K-loop" ignores that shrinking q *multiplies* the Q-chunk count.
    iters/core goes 2592 at (256, 512) to 2816 at (192, 640) — more iterations, not fewer. The two
    effects oppose each other and q dominates.

So `(256, 512)`, which `measured_sdpa_chunk_sizes[13664]` already ships, is optimal. **Chunk-size
tuning at 15 s is exhausted**: 3 feasible points measured, 5 ruled out by L1. The ~48% FPU / 35.6%
math utilization is inherent to the ring joint SDPA kernel at this shape.

### L1 envelope for the ring joint SDPA, calibrated

The four L1 failures carry exact byte counts, which fit the footprint exactly (Sq = q/32,
Sk = k/32):

```
bytes = 2048*Sq*Sk + 67584*Sq + 32768*Sk + 116032        (Wormhole max: 1,499,136)
```

Per unit that is 1 tile for `Sq*Sk`, **33 tiles for Sq and 16 for Sk** — so Sq is about twice as
expensive as Sk, which is why `(512, 256)` fails while `(256, 512)` fits at the same product. The
`Sq*Sk` coefficient being one tile rather than two also says the mask CB is not allocated here,
consistent with `is_causal=False`. Observed: `(6,24)` 1,602,880 B; `(8,20)` 1,639,744 B; `(8,24)`
1,836,352 B; `(6,32)` 1,963,328 B. Reusable for any future chunk question on this part.

## Perf experiments

| # | experiment | status | result |
|---|---|---|---|
| 1 | Matmul blockings, all 4 shapes x 3 durations, 8x8/8x9 grids | **done** | 3.5% of matmul time at 15 s; ff1 and ff2 landed, 0.5% of a forward |
| 2 | Fused MM/RS at 8x5/8x6/8x7 matmul grids | **done** | All worse than unfused; stays disabled |
| 3 | SDPA chunk sizes, q in {256,384,512} x k in {256,512} | **done** | Shipped `(256, 512)` already optimal; larger q L1-infeasible |
| 3b | `q_chunk=128` | **done** | Not a perf path: slower than q=192 at every feasible k. Hang history in one line under *SDPA chunk sizes*. |
| 4 | SDPA chunk sizes, small-q / large-k (q<=256, k>=512) | **done** | Hypothesis disproved. `(192, 640)` is feasible — the first k>512 point on this shape — but 13% slower than the shipped `(256, 512)`; larger q is more per-core efficient and shrinking q raises iters/core. Chunk tuning at 15 s is exhausted. L1 envelope calibrated as a by-product |
| 5 | Re-profile the block with landed configs | blocked | **TODO** — needs the pinned `diffusers` fork; not installed here |
| 6 | Pipeline re-run: warm total, denoise, ms/fwd, CLIP | **done** | Same-host A/B: **-69.9 ms/fwd, -0.58%**, exactly the isolated-sweep prediction. CLIP **35.88** (min 34.69, bar 33.0) on the later run |
| 11 | Numerics of the landed blockings (the sweep never checked) | **done** | ff2 (8,7,10) pcc 1.0000000 vs torch, identical to (8,8,8) to one bf16 ulp; ff1 (8,7,10) pcc 0.9999843 on the real SwiGLU ring, = (8,3,14) to 6 dp. Both PASS |
| 7 | `use_exp_ring_sdpa` on Wormhole | **done** | Brought up (header-pool and reader fixes, even-row grid, 2 or 4 links, sequential passes for shards that do not fit L1); PCC 0.99975. 15 s shard (padded to 14336 rows): exp **206.7 ms** vs normal 192.9 ms (+7%) on 56 vs 63 cores; ~5% less core time per unit of work. See *Exp ring joint SDPA on Wormhole* |
| 8 | FSDP layout conversions | not started | **TODO** — tilize/untilize go 0.13 -> 3.13 ms under FSDP, a 23x blowup and a quarter of the whole FSDP cost spent on format round-trips rather than communication. Cheapest apparent win in the breakdown |
| 9 | Ring SDPA kernel utilization | not started | **TODO** — 48% at the shipped chunk size, inherent to the kernel at this shape rather than a chunk-size miss. Note `PM FPU UTIL (%)` is the perf-model ideal divided by measured time (`tools/tracy/process_ops_logs.py`), not a hardware counter; the exp kernel shares the same inner loop, so the work remains in `compute_streaming.hpp` |
| 10 | `dit_fsdp: True` in `_PRESETS_WH` | not started | **TODO** — decision, not a measurement; costs 5.7% of the block, buys the headroom a 12 GB part needs |
| 13 | TP/SP axes and factors at 15 s / 16:9 (`test_parallel_sweep_minimax_h3.py`) | **done** | Only three configurations exist on this mesh and the shipped TP4/SP8 is the fastest: TP8/SP4 is **+4.1%** ms/fwd (untuned blockings), TP1/SP32 **hangs deterministically** in its first forward. See the section below |
| 14 | ff1 AGMM utilization (51% of HiFi2 peak) | **measured** | Roofline + six on-device experiments. The 15.7 ms splits into 8.0 ms of FPU work, **2.8 ms of serialized SwiGLU epilogue** and 4.9 ms of K-loop overhead where the compute-thread structure (4-tile DST, fp32 L1-acc pack every 7 MACs) and the operand delivery (~10 GB/s per core through the store-and-forward relay) are balanced co-limiters. fp32 dest off measures -4% alone, -8% with 8-tile subblocks, at 2x the numerical error; K_block >= 14 gives nothing. See *ff1 AGMM: where the other 49% goes* |

## Exp ring joint SDPA on Wormhole — brought up and measured (2026-09-18)

`exp_ring_joint_scaled_dot_product_attention` is the fused ring-attention kernel that measured 21% faster
than `RingJointSDPADeviceOperation` on Blackhole at the H3 shard `[1, 14, 3424, 128]`, ring 8 (`42986a68fe0`).
This section records what it took to run it on this galaxy and what it measures against the normal op. All
numbers are max over the 32 devices, `DEVICE KERNEL DURATION`, from Tracy CSVs under
`generated/profiler/reports/2026_09_18_01_4*`–`02_0*`.

### What blocked it, and the fixes

| blocker | where | fix |
|---|---|---|
| model gate `is_blackhole() and sp == 32` | `attention_minimax_h3.py` | `MINIMAX_H3_EXP_RING_SDPA=1/0` forces it on/off; unset keeps the Blackhole rule |
| SDPA rows must be even (backward/forward MUX-client halves) | `exp_ring_joint_sdpa_program_factory.cpp` "SDPA grid rows must be even" | program grid `(8, 8)`: **7x8 = 56 SDPA cores** (the normal op has 63); `num_workers_per_link = 4` |
| `num_links == 2` `TT_FATAL`; one MUX-client column per link | `exp_ring_joint_sdpa_device_operation.cpp` | the model passes 2 for this op (`MINIMAX_H3_EXP_RING_NUM_LINKS`); the factory now also lays out 4 client columns / 8 MUX kernels for `num_links=4` |
| **fabric packet-header pool**: the AG writer allocates 8 scatter + 2 unicast + 1 atomic-inc headers per RISC; Wormhole's pool is `NUM_PACKET_HEADERS / 2 = 8` per RISC (Blackhole 12), and `PacketHeaderPool::allocate_header` spins forever on exhaustion | `exp_ring_joint_writer.cpp`; `tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h:149` | rotation sized from the budget: 4 scatter headers on Wormhole, 8 on Blackhole. This was the first hang: every fabric writer on all 32 chips parked in `allocate_header` |
| reader's per-link semaphore array was `[2]` | `exp_ring_joint_reader.cpp` | `[4]` with an assert; with 4 links the overflow corrupted the reader's stack and every reader exited without work (second hang) |
| 8 KB fabric payload illegal on WH (cap 7616 B) | `fabric_context.cpp` | the H3 WH mesh already runs 4 KB; the op derives packet size from the fabric |
| `kMaxPasses = 3` | factory + device op | raised to 4 (the CB budget check is what bounds passes) |

### L1 decides the shape, not the gates

The op keeps every pass's Q chunk and flash state resident for the whole op, so per-core L1 scales with
`rows_per_device x heads_per_device / SDPA cores`. Using the model's own `_exp_sdpa_l1_bytes`: **no
(cols, segs, q, k) fits 5 s, 10 s or 15 s on the 7x8 grid** (minimum 2.45 MB at 15 s against a 1.31 MB
budget, even with Q streamed and the pass cap lifted). The only H3 shard that fits is the SP=32-equivalent
`3424 rows/device` — the `sp_sim4` shard — at q512/k128 (2 passes, streamed Q) or q256/k256 (4 passes,
streamed Q). Everything below is measured there; **the real 15 s pipeline shard cannot run the exp op as
designed**. Making it fit means sequential passes (pass-outer, ring-inner) with passes ≥ 1 reading the
gathered K/V from DRAM instead of the fabric — kernel work across reader, writer, compute and factory.

### Single op, `[1, 14, 3424, 128]`, ring 8, HiFi2 bf16, PCC 0.99975 on every exp point

| op | config | cores | per call |
|---|---|---|---|
| normal `RingJointSDPA` (sweep best of 7, `create_perf_table[minimax_h3_15s_768p_sim32]`) | q256 / k512 | 63 | **15.74 ms** |
| normal | q160 / k256 | 63 | 16.11 ms |
| exp, 2 links | q256 / k256, 4 passes | 56 | 16.65 ms |
| exp, 2 links | q512 / k128, 2 passes | 56 | 20.26 ms |
| exp, 4 links | q512 / k128, 2 passes | 56 | 20.33 ms (no change vs 2 links: the K/V gather is not on the critical path) |

### In the block (`test_minimax_h3_transformer_block_perf[wormhole_b0-sp_sim4-15s_768p-…_is_fsdp1]`)

| SDPA variant | SDPA row | block device-only |
|---|---|---|
| normal ring op (model fallback q256 / k512) | **14.76 ms** | **44.39 ms** |
| exp, q256 / k256, 4 passes (`MINIMAX_H3_EXP_RING_MAX_PASSES=4`) | 16.09 ms | 45.12 ms |
| exp, q512 / k128, 2 passes | 19.88 ms | 49.13 ms |

The normal op is **~9% faster on the SDPA row** at the one shard the exp op can hold, and the exp op runs
on 56 cores where the normal op has 63. The Blackhole 21% did not transfer: there the exp op fits Q and
state resident on 110 cores at q160/k512, here L1 forces streamed Q and k=128–256 chunks whose per-chunk
overhead costs more than the removed DRAM save/restore saves. Both ops share the same inner loop
(`sdpa_inner_loop_step`), so the Wormhole utilization gap (48% vs Blackhole's ~70% on the same kernel) is
untouched by either.

Reproduce: unit test `test_exp_ring_joint_attention.py::…[wormhole_b0-4x8_wh_h3_sim32{,_p4,_nl4}-ring]`
(PCC + timing under `--profile`), normal-op table `test_ring_joint_sdpa.py::…create_perf_table[minimax_h3_15s_768p_sim32]`,
block A/B with `MINIMAX_H3_EXP_RING_SDPA={1,0}` and `MINIMAX_H3_EXP_RING_MAX_PASSES={3,4}`.

### Sequential passes: the exp op on the 15 s shard (2026-09-18, branch `jameslee/exp_ring_sdpa_wh`)

The L1 wall above comes from the lockstep schedule: every pass's Q chunk and flash state stay resident
because all passes advance together per ring iteration. `TT_EXP_SDPA_Q_GROUPS=G` (factory + all three
kernels, one setting per process) switches the op to **pass-outer / ring-inner**: one pass runs all
ring iterations before the next starts, so one Q chunk and one flash state are live per core (the
normal op's `q_per_core == 1` scratch path, no L1 state FIFO), and a head-segment's Q chunks are split
into G groups of one chunk per column walked as extra passes. Only group 0 of a segment forwards K/V
over the fabric; later groups re-read the gathered K/V the first group landed in DRAM. Per-core L1 is
then the single-pass footprint (q256/k512: 599 tiles = 1.23 MB) at any shard size.

Constraint: `num_q_chunks % (columns x G) == 0`. 13664 rows give 54 chunks of q=256, which no
7-column layout divides, so the shard is padded to **14336 rows/device** (56 chunks = 7 x 4 x 2
segments, +4.9% work). The normal op was measured at the same 14336 rows for the comparison.

| op | layout | links | per call |
|---|---|---|---|
| normal `RingJointSDPA` (`create_perf_table[minimax_h3_15s_768p_pad14336]`) | q256 / k512, 63 cores | 4 | **192.9 ms** |
| exp, sequential, G=4 | segs=2 (pair dedup on), 4 segment-passes on rows 0-3 and 3 on rows 4-7, 56 cores | 2 | 238.0 ms |
| exp, sequential, G=4 | same | 4 | 238.2 ms |
| exp, sequential, G=2 | segs=4 (7 balanced passes per row, no dedup: every row forwards) | 2 | 212.6 ms |
| exp, sequential, G=2 | same | 4 | **206.7 ms** |

Numerics: sequential mode PCC 0.99975 at the 3424-row shard (`4x8_wh_h3_sim32_seq`, G=2), identical to the
lockstep schedule; the 14336-row runs are timing-only (the torch reference at 114688 tokens does not fit
host memory) and validated end-to-end only through that smaller PCC.

Reading: the exp op **now runs the 15 s shard on Wormhole**, best at 206.7 ms against the normal op's
192.9 ms (+7%). Per core it is ahead: 206.7 ms x 56 cores = 11.6 core-s against 192.9 x 63 = 12.2 for the
normal op, i.e. ~5% less core time for the same work. The remaining gap is exactly the 7 cores the
even-row MUX-client constraint costs on a 9-row grid; a 9-row layout (asymmetric backward/forward
client halves) would make the exp op the faster kernel at 15 s. Row balance matters more than fabric
duplication: segs=4 forwards every head 4x yet beats segs=2 with pair dedup by 11%, and only at segs=4
do 4 links help (212.6 -> 206.7 ms). Reproduce with
`TT_EXP_SDPA_Q_GROUPS={2,4} … test_exp_ring_joint_attention.py::…[wormhole_b0-4x8_wh_h3_15s_seq{,_nl4}-ring]`
under `--profile`; the normal-op row is `create_perf_table[minimax_h3_15s_768p_pad14336]`.

### Bottom-row MUX placement: 64 SDPA cores instead of 56 (2026-09-18, branch `jameslee/exp_ring_sdpa_wh`)

The 7 lost cores above come from the reserved MUX *column*: the SDPA grid is 7 wide, and the 9-row
height must round down to 8 for the equal backward/forward MUX-client halves. The op already had a
Blackhole placement experiment, `TT_EXP_SDPA_MUX_BOTTOM_ROW`, that puts the MUX kernels on the bottom
*row* instead and gave up two rows to keep the count even. On the 8x9 Wormhole grid one row is enough:
SDPA keeps all 8 columns and rows 0-7 (8x8 = **64 cores**), the 8 MUX kernels of 4 links fill row 8
exactly. The change is host-side only: the grid helper (`exp_sdpa_grid_for_user_grid`) drops one row
plus an idle row only when the remainder is odd, and the bottom-row MUX list is generalized from two
hard-coded pairs to `num_links` columns per direction (the 9-row asymmetric-halves layout the previous
section proposed would give 63 cores for far more surgery, so it is not needed).

With 8 columns the 56 chunks of q=256 divide as 8 x 7, so a segment is 7 chunks wide and the sequential
mode runs G=1: 98 segments on 8 rows -> 13 passes on rows 0-1, 12 on rows 2-7 (ideal 12.25). Every pass
forwards its head's K/V (no groups), so a row forwards 13 shards per op against 7 in the segs=4/G=2 layout.

| op | layout | links | per call |
|---|---|---|---|
| normal `RingJointSDPA` | q256 / k512, 63 cores | 4 | **192.9 ms** |
| exp, sequential, G=2 (previous best) | segs=4, 7x8 = 56 cores, reserved-column MUX | 4 | 206.7 ms |
| exp, sequential, G=1 | q256 / k512, segs=7, **8x8 = 64 cores**, bottom-row MUX | 2 | 199.6 ms |
| exp, sequential, G=1 | same | 4 | **196.2 ms** |
| exp, sequential, G=1 | q448 / k256, segs=4 (7 passes of 14 tile-rows), 64 cores | 4 | 200.9 ms |
| exp, sequential, G=1 | q192 / k512 at 13824 rows/device (72 chunks = 8 x 9; 16 passes of 6 tile-rows) | 4 | 220.2 ms |
| exp, sequential, G=1 | q128 / k512, segs=14 (25 passes of 4 tile-rows), 64 cores | 4 | 336.2 ms |
| exp, sequential, G=1 | q448 / k512 | 4 | does not build: CBs need 1.71 MB of 1.34 MB |

Numerics: PCC 0.99975 at the 3424-row shard on the bottom-row layout with 2 links (`4x8_wh_h3_sim32_seq`,
7x8, unchanged from the reserved-column number) and 0.99972 on the 64-core grid with 4 links at a
4096-row shard (`4x8_wh_h3_4096_bot_nl4`, 16 chunks = 8 x 2). The 15 s rows are timing-only as before.

Reading: 64 cores take the exp op from 206.7 to **196.2 ms**, 1.7% behind the normal op (192.9). The
per-core work model predicted 206.7 x 13/14 = 192 ms; the missing 4 ms is fabric traffic (13 forwards
per row instead of 7), visible as 2 links -> 4 links = 199.6 -> 196.2 and as a 3 ms spread across
devices that the 56-core layout did not have (206.80 / 206.84). The chunk sweep says the op's time
follows the number of inner-loop steps (passes x K chunks), not Q tile-rows: q=192 does 8% fewer
tile-rows per core but 19% more steps and is 12% slower; q=128 nearly doubles the steps and is 71%
slower; q=448 cuts the steps but only fits with k=256, and the halved K chunk costs more than the
larger Q chunk saves. q256 / k512 stays the shape. What is left between 196.2 and a win: the
13-vs-12.25 pass imbalance (6%, inherent to 98 segments on 8 rows at q=256) and the duplicate
forwarding (~2%, the 2-vs-4-link gap). With segs=7 a pass holds 8 segments of 2-3 distinct heads, so
generalizing the pair dedup to "one forwarder per (pass, head)" would cut each row's forwards from 13
to about 4 and take the traffic off the critical path even on 2 links. Reproduce with
`TT_EXP_SDPA_MUX_BOTTOM_ROW=1 TT_EXP_SDPA_Q_GROUPS=1 … [wormhole_b0-4x8_wh_h3_15s_seq{,_nl4}-ring]`
under `--profile`; the q128 / q192 / q448 rows are the `4x8_wh_h3_15s_q{128,192,448}_nl4` cases.

### Inner loop: where a 66 µs step goes, and three experiments (2026-09-18, branch `jameslee/exp_ring_sdpa_wh`)

Both ring ops run `sdpa_inner_loop_step` (`compute_streaming.hpp`) once per (Q chunk, K chunk):
at q256 / k512 that is 2912 steps per core per call at 15 s, 66 µs each (192 ms). The kernel
already carries per-phase device zones behind a `profiling_enabled` template flag; the new env knob
`TT_EXP_SDPA_PROFILE_INNER=1` (exp op) compiles them in. The L1 profiler buffer holds ~125 zones per
RISC per launch, so the log covers the first step and a half of each core, which is what the table
uses (`tools/sdpa_phase_zones.py` on the report's `profile_log_device.csv`). Step 0 on the math
thread includes a 15 µs wait for the first K chunk; the steady-state step is ~66 µs.

Per step, one core (device 0, core (1,1)), pack-4 build:

| thread | matmul zones (QK + PV) | softmax zones | outside all leaf zones |
|---|---|---|---|
| unpack (TRISC_0) | 16.6 + 21.1 µs | SUB 8.7, reduce 3.1 | 17.7 µs |
| math (TRISC_1) | 34.3 (19 steady) + 21.2 µs | SUB 10.1, init 1.9, reduce 0.8 | 10.9 µs |
| pack (TRISC_2) | 22.2 + 2.2 µs | EXP 14.4, PACK SUB_EXP 14.0, reduce 1.7 | 24.9 µs |

Reading. The pure FPU work is 1024 tile-matmuls per step (QK 8x16x4, PV 8x4x16) at 32 cycles each
for HiFi2 = 33 µs, i.e. **50% of the step**, which is the 48% "FPU util" the roofline reported.
Inside their zones the matmul blocks run at ~80% of that rate. The other half of the step is the
softmax and the thread handshakes: on the pack thread the exp (SFPU, `exp_packthread_tile`) and the
in-place pack plus the row-sum accumulate pack (every probability tile is packed twice, the second
time with packer L1-accumulate into the row-sum tile) cost 28 µs per step; on the math thread the
broadcast subtract of the row max is 10 µs (128 tiles at ~80 cycles) and 11 µs sit between zones in
`tile_regs_acquire`/`wait` handshakes and CB waits. Nothing waits on DRAM or the fabric after the
first chunk: memory and fabric are off the critical path, the core is bound by its own non-matmul
work. That is also what the chunk sweep said (time follows steps, not FLOPs).

Experiments, exp op on 64 cores at 15 s (196.2 ms base), max over 32 devices:

| change | per call | verdict |
|---|---|---|
| A. `MIN_BLOCKED_PACK_TILES` 8 -> 4 on Wormhole (one pack per 4-wide subblock row instead of 4) | **193.7 ms** | kept; normal op 192.9 -> **191.6 ms** on the same change (`create_perf_table[minimax_h3_15s_768p_pad14336]`) |
| B. full-sync 16-tile DST (`dst_full_sync_en`, now forwarded by the exp factory; test knob `TT_EXP_SDPA_TEST_DST_FULL_SYNC=1`) | 262.8 ms | rejected: the half-sync ping-pong that overlaps math and pack is worth far more than larger subblocks |
| C. approximate SFPU exp (`TT_EXP_SDPA_TEST_EXP_APPROX=1`, on top of A) | 192.7 ms | 0.5%; PCC 0.99972 unchanged at 4096 rows; the model keeps exact exp |
| D. `--profiler-capture-perf-counters=fpu,pack,unpack` (via `SAFE_PYTEST_TRACY_OPTS`) | no data | Tracy's multi-pass counter capture deadlocks on this box: the inner `python -m tracy` waits on a UMD chip lock its parent holds (21 min, killed) |

What is left in the inner loop, by size: the double pack of the probabilities (14 µs on the pack
thread; computing the row sum with the FPU reduce instead of the packer accumulate would trade
pack time for math time, and math has ~11 µs of handshake slack), the SFPU exp (14 µs; approx mode
proved it is not SFPU-op bound, so the cost is the pack-thread scheduling around it), and the
broadcast subtract (10 µs on math; a fused "exp(x - m)" on the SFPU would remove it, the current
custom LLK ignores its fidelity parameter so LoFi does not help). Each is a kernel change of a day
or more and applies to both ring ops.

## ff1 AGMM: where the other 49% goes (2026-09-18, branch `jameslee/exp_ring_sdpa_wh`)

The fused all-gather + matmul for ff1 (`all_gather_minimal_matmul_async`, per device M=13664, K=5376 gathered
over the TP=4 ring, N=7168 packed gate|up, fused SwiGLU, bf16, HiFi2, fp32 dest, 8x8 worker grid, blocking
(8, 7, 10) subblock 2x2) runs at 15.7 ms against an 8.03 ms compute roofline on the 64-core AGMM grid
(64 x 2048 FLOP/cycle x 1.0 GHz = 131.1 TFLOP/s), i.e. 51%. Its DRAM bound is 0.78 ms and its fabric bound
1.10 ms (4 links x 12.5 GB/s, bidirectional ring), so on paper the op is compute-bound by 7x. The roofline
script that produces these numbers and the Blackhole cross-reference is
`models/tt_dit/tests/models/minimax_h3/tools/agmm_roofline.py`.

Method: mine the 320-combo ff1 block sweep already on disk, read the op's kernels, then six on-device
experiments (all on this Galaxy, baseline re-measured in every run: 15,632-15,743 us). The block sweep and
the AGMM runs use `sweep_mm_block_sizes.py`; the single-device runs use a host-timed script (8 iterations
after warm-up, dispatch included, so absolute numbers are ~1-2% pessimistic).

### What the kernel does (source read, `.../all_gather_minimal_matmul_async/device/`)

- No multicast. in0 (the gathered activation, a DRAM buffer) and in1 (weights, DRAM) each travel down a
  serial store-and-forward chain of 8 cores with a request/response semaphore round-trip per hop per
  K-block (`dm_in0_sender.cpp:446-476`, `dm_in1_sender_out.cpp:511-541`). Only the chain head reads DRAM,
  tile by tile with one barrier per block.
- Loop order m -> n -> k, so in0 is re-read from DRAM once per N block (3x) and in1 once per M block (7x):
  ~1.0 GB of DRAM reads per device per op, all issued by the 16 chain-head cores, and ~127 MB of relay
  writes received and re-sent by every core.
- Compute: DST holds 4 tiles under fp32 dest (`get_dest_reg_count`), so the subblock is 2x2 and every
  K_block (7 tiles) the partial sums are packed as fp32 into an L1 intermediate CB with L1 accumulate:
  36,288 accumulating fp32 packs per core against 254,016 tile-MACs. `packer_l1_acc` is not plumbed
  (always on). The intermediate CB is 320 KB, the out CB is single-buffered.
- SwiGLU is a separate epilogue pass per output block (`swiglu_block`, `compute.cpp:28-70`): copy gate and
  up tiles from the intermediate into DST, `silu`, `mul`, pack, with the two SFPU inits re-armed per tile
  pair. It runs on the same math/pack threads after the K loop, so nothing overlaps it.

### Experiments

| # | question | result |
|---|---|---|
| 1 | Is the fp32 pack every 7 MACs the limiter? Raise K_block (the sweep never measured K>=14 because its L1 estimator over-counts this op's CBs: double-buffered full-width out and a bias CB that ff1 does not allocate) | No. (8,14,8) 16,227 us, (8,14,6) 16,281, (6,14,10) 16,898, (8,21,4) 18,131 vs 15,725 baseline; (8,14,10) and (8,21,6) fail at warm-up |
| 2 | fp32 dest off (`MM_SWEEP_FP32_DEST_ACC=0`, new knob), which allows 8-tile subblocks and a bf16 intermediate | Same blocking (8,7,10) 2x2: **15,162 us (-3.6%)**. Best (8,7,16) sb 2x4: **14,475 us (-8.0%)**; (12,7,8) sb 4x2 14,677; (8,7,14) 2x2 14,756 |
| 3 | Single device, same per-device shape: how much is the gather fusion, the SwiGLU epilogue, the base kernel? | `minimal_matmul` + SwiGLU (8,7,10) fp32: 16.56 ms, i.e. the mesh AGMM is not slower than the plain op -- the ring gather is fully hidden. Plain (no SwiGLU): 14.52 ms -> **SwiGLU costs ~2.0 ms** at every blocking tried. fp32 off (12,7,8) 4x2 plain: 12.67 ms (63%). Mainline `ttnn.matmul` auto-config 8x8 is no better: 23.4 ms fp32 on, 15.2 ms off |
| 4 | Math-bound or data-movement-bound? Same kernel at LoFi / HiFi2 / HiFi4 (ideal 4.0 / 8.0 / 16.1 ms) | Plain (8,7,10) fp32: **13.17 / 14.48 / 20.72 ms**. Halving the math saves 1.3 ms, doubling it costs 6.2 ms: at HiFi2 the kernel sits on a ~12-13 ms non-math floor. With (12,7,8) fp32 off (fewer in1 re-reads) the LoFi floor drops to 10.73 ms |
| 5 | Tracy zones in the K loop (first 60 of 504 iterations per core fit the profiler buffer) | Unpack thread waits **61 us for operands in 1,590 us** of matmul; the math thread spends **27.3 us per K-block iteration against 17.9 us of pure math** (66%). At HiFi2 the operands arrive just in time; the loss is inside `matmul_blocks` |
| 6 | Tracy zones per output block (all 21 blocks per core) | Per core: kernel 15,750 us = **K loop 12,934 + SwiGLU 2,809 (17.8%)** + output waits 0 + 10 us outside zones. Identical on every core and device |

Experiments 4 and 5 together say the K loop has two limiters of nearly equal size at HiFi2: the
compute-thread structure (DST handshake and the fp32 accumulating pack per subblock, ~27 us per iteration) and
the operand delivery through the relay chain (~24 us per iteration, ~250 KB per core per iteration, i.e.
~10 GB/s delivered per core). That is why the levers that touch one side only move a few percent: a bigger
K_block halves the packs but doubles the bytes per iteration, and fp32-off speeds up the packs until the
relay binds.

### Decomposition of the 15.7 ms

| component | ms | share | evidence |
|---|---|---|---|
| FPU work at HiFi2 peak | 8.03 | 51% | roofline |
| K-loop overhead (DST/pack structure, co-limited by operand delivery) | 4.9 | 31% | exp 4, 5, 6 |
| SwiGLU epilogue, serialized after each block's K loop | 2.8 | 18% | exp 6 (2.0 ms of it is SwiGLU-specific, exp 3) |
| operand waits, output waits, fabric, dispatch | <0.1 | <1% | exp 5, 6 |

### Levers, ranked

1. **The SwiGLU epilogue (2.8 ms, no numerical change).** `swiglu_block` re-arms `silu_tile_init` and
   `mul_binary_tile_init` inside the per-tile loop and stages every gate/up pair through the fp32
   intermediate. Hoisting the inits (all silu tiles of a block, then all multiplies), or a single fused
   SFPU pass, or applying the epilogue while the last K block's result is still in DST, would bring it
   towards the ~0.8 ms a plain copy epilogue costs. Kernel work in `compute.cpp`, no config change.
2. **Operand delivery (the ~24 us per K-block floor).** Any of: multicast instead of the 8-deep
   store-and-forward relay; keep an in1 block resident across M blocks so the weight is read once per N block
   (the fabric-bound factory already has this `c_7` scratch CB, `minimal_matmul_fabric_bound_program_factory.cpp:347-357`,
   but that path has no Wormhole grid entry in `fabric_agmm_configs`); spread the head-core DRAM reads over
   the chain; double-buffer the out CB. The (12,7,8) result (-13% on the plain kernel) shows how much cutting
   in1 re-reads alone is worth.
3. **Compute-thread structure.** fp32 dest off: measured -3.6% at the same blocking and -8% with an 8-tile
   subblock. It does **not** change the roofline (fidelity fixes the FPU rate; fp32 dest only halves DST
   and makes the packs fp32), but it is a precision decision: on the single-device shape, relative RMSE
   against fp32 torch doubles (plain 0.0050 -> 0.0109, SwiGLU 0.0087 -> 0.0169; PCC 0.99993 -> 0.99985; the
   ff1 bar is rmse < 0.02). Not landed. Only worth taking together with lever 2, since the relay then binds.
4. **Not levers here:** K_block (exp 1), the ring gather (hidden, exp 3 and 5), fabric bandwidth
   (1.1 ms bound, 7% util), aggregate DRAM bandwidth (0.78 ms bound), LoFi (halves the roofline, a quality
   decision), the grid (the mux row is the 8x8 vs 8x9 cost and is fixed by the op).

Continued in **`MiniMaxH3_wormhole_agmm_ff1_handoff.md`**: the SwiGLU attribution (silu is 2.16 of the 2.2 ms), the
bf16-grade silu result and its mesh hang, per-lever change recipes and tooling.

Housekeeping from this pass: `sweep_mm_block_sizes.py` gained `MM_SWEEP_FP32_DEST_ACC=0`; its L1 pre-filter
still over-estimates the AGMM footprint (fixing it would admit the K>=14 combos, which measured slower
anyway). The kernel zones used in experiments 5 and 6 were temporary and are not in the tree.

## TP/SP parallel-configuration sweep — 15 s / 16:9

Measured 2026-09-17 on this host at `bc1d99d05f6` plus the `matmul.py` change listed at the end
(landed on top of `2a47fd04fc6`), with `models/tt_dit/tests/models/minimax_h3/test_parallel_sweep_minimax_h3.py` (new). Driver,
logs, `results.jsonl`, the strided frame dumps and `compare.py` are in `~/h3_parallel_sweep/` on the
run host. `MINIMAX_H3_DIT_FSDP=1`, Ring, 4 links, seed 0, the fox prompt, **10 scheduler steps**
(9 forwards): ms/forward is flat across steps, so 10 steps gives the ranking at a fifth of the wall
clock. The 10-step ms/fwd runs ~3% above the 50-step figure (fixed per-request work amortised over
9 forwards instead of 49); every row here is at 10 steps, so the comparison is like for like.

### What can be configured at all

The system mesh is 8x4. Probing every 32-device shape with `FABRIC_1D_RING` at the 4 KB payload:
8x4, 4x8, 1x32 and 32x1 open and ring-all-gather on both axes; **2x16 and 16x2 are rejected** by
`system_mesh.cpp:224`. And 4x8 is exactly the transpose of 8x4 -- row *r* of the 4x8 device-id grid
is column *r* of the 8x4 grid -- so `4x8 tp0/sp1` and `8x4 tp1/sp0` are the same physical rings.
TP=16 is out regardless (56 heads). That leaves three distinct configurations, not a sweep space:

| config | mesh | TP axis | SP axis | rows/device | padded_len |
|---|---|---|---|---|---|
| `4x8_tp0_sp1` | 4x8 | 0 (TP=4) | 1 (SP=8) | 13664 | 109312 |
| `4x8_tp1_sp0` | 4x8 | 1 (TP=8) | 0 (SP=4) | 27296 | 109184 |
| `1x32_tp0_sp1` | 1x32 | 0 (TP=1) | 1 (SP=32) | 3424 | 109568 |

### Results

| config | TP/SP | enc | denoise | vae | audio | total | ms/fwd | vs shipped | video PCC vs shipped | audio PCC |
|---|---|---|---|---|---|---|---|---|---|---|
| `4x8_tp0_sp1` (shipped) | 4/8 | 3.5 | 111.4 | 14.4 | 4.1 | 133.4 | **12382** | -- | 1.0 | 1.0 |
| `4x8_tp1_sp0` | 8/4 | 4.1 | 116.0 | 14.4 | 3.4 | 137.9 | 12890 | **+4.1%** | 0.907 | 0.972 |
| `1x32_tp0_sp1` | 1/32 | -- | **hang** | -- | -- | -- | -- | -- | -- | -- |

**The shipped TP4/SP8 stays.** Neither alternative is a speedup, and one does not run.

**TP8/SP4 (+4.1%)** runs on the generic matmul blockings: none of its four AGMM shapes
(`(5376, 2688)` qkv, `(7168, 672)` to_out, `(5376, 3584)` ff1 at M=27296, plus the refiner's) has a
swept entry. The shipped config's tuned entries are worth 0.58% (experiment 6), so even a generous
allowance for tuning TP8's shapes leaves it ~3.5% behind, and the halved SP ring buys nothing the
doubled TP ring does not cost: per layer each device now receives ~257 MB per AGMM all-gather (was
110 MB) against ~294 MB of KV around the ring (was 685 MB). Its output is the same video -- same
fox, scene and motion, small pose/detail drift (mean |diff| 12-19 of 255 per frame, growing with
frame index) -- which is what a changed bf16 reduction order looks like after 9 sampling steps;
`compare_tp4_vs_tp8.png` in the results dir shows four frame pairs. Not a correctness problem.

**TP1/SP32 hangs, deterministically.** Two attempts, the second on a freshly reset board with
kernels coming from the cache (zero `BuildKernels` lines), both stall in the *first* forward:
the last log line is the generic-blocking warning for `proj_in` `(107872, 96, 5376)`, then nothing.
Fingerprint identical to `MiniMaxH3_wormhole_hang.md`: 180-365% CPU with CPU-time far past
elapsed, all ~448 threads in `futex_wait_queue`, `pytest --timeout` does not fire, board needs
`tt-smi -r all` afterwards (which warns that Galaxy CPLD FW < 1.16 should use `-glx_reset`, but did
work here). Not root-caused; TP=1 is also the configuration with the least to gain (each device
holds all 56 heads, so the KV ring moves 4x the bytes of TP4, and FSDP gathers full 5376-wide
weights over a 32-ring), so it was not pursued further. Evidence in `1x32_tp0_sp1_s10.HANG.txt`.

### Code changes this needed

* `models/tt_dit/utils/matmul.py`, `_ring_safe_k_block` in `get_agmm_config`'s generic fallback.
  `all_gather_minimal_matmul_async` on Ring asserts `K_tiles_per_device % K_block == 0` (its
  bidirectional half-block scheme has no tail block; Linear does). At TP=8 a 5376-wide input is 21
  K tiles per device and the generic `(8, 8, 8)` threw on the refiner's first matmul, 20 minutes
  into a weight load. The fallback now drops K_block to the largest divisor (7 here) and warns
  once per shape; swept table entries and caller-supplied `default_block_size` are untouched, so
  the shipped TP=4 configs are bit-for-bit what they were.
* `test_parallel_sweep_minimax_h3.py`: the harness. Env knobs `H3_SWEEP_STEPS`, `H3_SWEEP_ASPECT`,
  `H3_SWEEP_DURATION_S`, `H3_SWEEP_OUT`. Writes one JSON line per run plus every 6th frame and the
  audio, so configurations can be PCC'd against each other.

Two harness lessons, both cost time: run each configuration in its own pytest process behind a
shell `timeout` (the hang wedges the process, not just the test), and do not gate a queued run on
`pgrep -f <driver name>` -- the waiting shell's own command line matches, and it waits forever.

## Open issues

Fixed items have been removed; their forensics live in the commits and in
**`MiniMaxH3_wormhole_hang.md`**. The mid-denoise hang (root-caused, 18/18 pass), the accidental
Wormhole fused MM/RS path (`eab3dfbd599`) and the VBench setup gaps (now in `MiniMaxH3.md`) were
all closed.

1. **Cache key omits device params** — `cache.load_model` keys on parallel config, mesh
   shape, dtype and FSDP, but not `l1_small_size`/device params. A cache written under a
   broken device config is silently reused forever. This cost a long debugging detour: a
   cache built during a run with `l1_small_size=0` produced text embeddings with
   `absmax=2.5e30` and a coherent video of the wrong subject (CLIP 13.12 instead of 37.36).
   With `TT_DIT_CACHE_VERIFY=1`, `cache.load_model` reloads every tensor it wrote and compares it
   shard-by-shard with the resident weights before `cache_dict.json` is created; a mismatch leaves
   the cache unmarked so it is rebuilt next run. Opt-in: it re-reads the cache once at creation. A cache-key term for device params
   is still worth having.

2. **`sweep_mm_block_sizes.py` cannot complete on a Wormhole Galaxy at its default flush
   cadence.** With `PROFILER_DUMP_EVERY = 10` the sweep stops making progress at the first
   mid-run `ttnn.synchronize_device` + `ttnn.ReadDeviceProfiler` flush (combo 10) and the host
   spins there indefinitely; the board then needs `tt-smi -r all`. Reproduced on four shapes
   across two ops, including the pre-existing Wan2.2 `3072_5120_3840_8x8_agmm_plain` entry, so
   it is not H3-specific and not blocking-specific — combo 10 is simply the first point at
   which the loop blocks, since warmup dispatches with `sync=False`. Setting
   `MM_SWEEP_PROFILER_DUMP_EVERY` high removes the in-loop flush and the same shape then
   completes 404/404 with valid, distinct timings; that is how every number here was collected.
   Unexplained: the *ungated* flush immediately after the warmup loop succeeds in the same run,
   so "this call hangs on a 32-device WH mesh" is not the whole story. The default is left at 10
   pending a root cause, which makes the override mandatory on Wormhole.

3. **M-keyed tuning tables and the pipeline disagreed on the per-device length** — *fixed in
   `a07012d7d8a`*. Every 768P table now keys on **4736 / 9184 / 13664** rows/device (5 s / 10 s /
   15 s), which is what the pipeline packs and logs on every run (`packed sequence ... rows/device`).
   `test_performance_minimax_h3.py::_packed_sizes` counts audio in rows (two per latent,
   `packing.py:261`) and uses the gate's 39-token prompt, and both the length and the padding go
   through `packing.py` helpers (`packed_sequence_length`, `padded_sequence_length`) that the pipeline
   itself uses, so the two cannot diverge again. `get_matmul_config`/`get_agmm_config` fall back to a
   same-(K, N) entry with the same `M_per_core` after an exact miss, which makes every table robust
   to a 32-row discrepancy and to prompt length within a padding bucket. Confirmed on a live
   generation: `13664 rows/device`, and neither ff1 nor ff2 appears among the fallback warnings (qkv
   and to_out do, by design -- their shipped blockings measured optimal / within noise). Root cause
   and history: **`MiniMaxH3_rows_per_device_mismatch.md`**.

## VBench (16:9/5s, verified passing)

| dimension | score | bar |
|---|---|---|
| subject_consistency | 0.9793 | 0.95 |
| background_consistency | 0.9779 | 0.95 |
| motion_smoothness | 0.9915 | 0.97 |
| dynamic_degree | 1.0000 | 1.0 |
| imaging_quality | 0.6802 | 0.64 |

CLIP 37.36 vs 33.0 bar (docs record 37.37 for Blackhole; imaging_quality 0.6896).
Only 16:9/5s has been VBench-verified; the sweep ran with `RUN_VBENCH=0`.
