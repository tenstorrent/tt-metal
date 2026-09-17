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

Measured 2026-09-17 on `3a2f7fea259`, after the ff2 fix, with the Wormhole `GALAXY_RING` rows
added in `2076022d032`. One block, warm iteration between `start`/`stop` signposts.

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

Shape (test-reported, matching the pipeline's packing helpers): 1344x768, 362 frames -> 107 latent
frames x 24x42 patches = 107856 video + 603 audio + 512 text = seq_len 108971, padded 109056,
**13632 rows/device** at SP=8.

| op | fsdp1 ms | fsdp0 ms | delta |
|---|---|---|---|
| **RingJointSDPADeviceOperation** | **172.52** | **172.50** | +0.02 |
| AllGatherMinimalMatmulAsyncOp (3) | 33.11 | 32.57 | +0.54 |
| EmbeddingsDeviceOperation (6) | 10.37 | 10.32 | +0.04 |
| MinimalMatmulDeviceOperation (2) | 8.84 | 8.87 | -0.03 |
| AllGatherAsyncDeviceOperation (4) | 4.10 | — | +4.10 |
| AllBroadcastDeviceOperation | 3.96 | — | +3.96 |
| DitFusedDistributedRmsnorm (4) | 2.94 | 2.91 | +0.03 |
| ReduceScatterMinimalAsync | 2.75 | 2.78 | -0.02 |
| UntilizeWithUnpadding | 1.76 | 0.07 | +1.69 |
| ConcatDeviceOperation | 1.55 | — | +1.55 |
| TilizeWithValPadding | 1.37 | 0.06 | +1.31 |
| *(remaining 8 ops)* | 3.04 | 3.02 | +0.02 |
| **device only** | **246.31** | **233.10** | **+13.21 (5.7%)** |
| **device + op gap** | **259.40** | **241.67** | +17.73 |
| SDPA share of block | 70.0% | 74.0% | |

Projected over 50 layers: **12.32 s/step** device-only, 12.97 s with op gaps (616 / 648 s per
50-step video). The 16:9/15 s row above measures the same configuration end-to-end at
**12435 ms/forward**, which lands inside that bracket exactly as `project_block_perf.py` intends:
`device only` is the underestimate (no dispatch gaps), `device + op gap` the overestimate. So the
50-layer block stack is **99.1%** of the forward on the device-only figure, leaving ~0.1 s for the
refiner, input projections, `norm_out` and the output heads. Corroborating that the comparison is
apples-to-apples: 21:9 and 16:9 both give 1008 tokens/latent frame (1536x672 -> 21x48;
1344x768 -> 24x42), and their measured forwards agree to 0.1% (12447 vs 12435).

### Findings

1. **Ring SDPA is 70% of the block** at 48.8% FPU utilization (`PM FPU UTIL (%)`, consistent
   across all 32 devices). Nothing else is close.
2. **FSDP costs 5.7%** — inside the 5-11% the pipeline sweep saw, and it decomposes exactly:
   AllGatherAsync 4.10 + AllBroadcast 3.96 + Concat 1.55 + Untilize 1.69 + Tilize 1.31 = 12.61 of
   the 13.21 ms delta.
3. **Layout conversions blow up 23x under FSDP** — tilize/untilize go 0.13 -> 3.13 ms, a quarter of
   the whole FSDP cost spent on format round-trips rather than communication. Cheapest apparent win.
4. **The ff2 fix is confirmed live**: `ReduceScatterMinimalAsyncDeviceOperation` is present and
   there is no fused `Matmul_RS` row, which is what `eab3dfbd599` intended.

### SDPA chunk sizes: already optimal at 15 s

`test_ring_joint_attention_create_perf_table[minimax_h3_15s_768p]` (run plain — it self-shells
`run_device_profiler`, so wrapping it in `--profile` would nest profilers):

| rank | q_chunk | k_chunk | duration | FPU util | math util | slot waste |
|---|---|---|---|---|---|---|
| 1 | **256** | **512** | **171.693 ms** | 48.1-49.3% | 35.6% | 0.0% |
| 2 | 384 | 256 | 192.593 ms | 42.9% | 31.8% | 0.0% |
| 3 | 256 | 256 | 195.029 ms | 42.3% | 31.4% | 0.0% |
| — | 384/512, 512/256, 512/512 | | L1 infeasible | | | |

`(256, 512)` is what `measured_sdpa_chunk_sizes[13632]` already ships. The three larger-q candidates
fail with `Statically allocated circular buffers on core range [0-0 - 6-8] grow to 1844544 B which
is beyond max L1 size of 1499136 B` — Wormhole's 1.5 MB/core, and that core range is the 7x9 = 63
compute grid. The harness independently reports "63 compute + 9 CCL = 72 total cores" and 0.0% slot
waste (756 work items / 63 = 12 passes exactly), and measures SDPA at 171.693 ms against 172.52 ms
in-block, 0.5% apart.

So the ~50% FPU / 35.6% math utilization is **inherent to the ring joint SDPA kernel at this
shape, not a chunk-size miss**. At 70% of the block it is the only thing worth attacking, but the
work is in the kernel. Note the contrast with 5 s, where `q=320` at seq 4768 wastes 16.7% of the
63 slots and chunk tuning *does* have headroom.

Caveat: `CORE COUNT` for `RingJointSDPADeviceOperation` reads 71, not 63, because the profiler
counts the fused CCL workers — `ccl_core_grid_offset=(7, 0)` with `use_column_major_ccl=True`
(`attention_minimax_h3.py:572-573`) places them in the reserved last column.

## Optimization target — 15 s / 768P / 16:9

Tuning work is scoped to this one configuration. Baseline is `c825d089e31` (the per-op breakdown
above), measured at `fsdp1`.

| | warm total | denoise | ms/fwd | realtime | CLIP | block device-only |
|---|---|---|---|---|---|---|
| baseline, other host (`c825d089e31` tables) | 633.1 s | 609.3 s | 12435 | 42.0x | 36.31 | 246.31 ms |
| baseline, **this host**, tuned entries disabled | 612.9 s | 590.9 s | 12058.3 | 40.6x | — (audio gate) | **TODO** |
| **best found**, this host, `a07012d7d8a` | **612.2 s** | **587.4 s** | **11988.4** | 40.6x | — (audio gate) | **TODO** |

Same host, same weights, same everything, one run each, 2026-09-17: the landed ff1 + ff2 blockings
are worth **-69.9 ms/fwd, -0.58%** (steady 12057 -> 11990 ms/step; denoise 590.9 -> 587.4 s). The
isolated sweep predicted 1327 us/layer x 50 = 66 ms/step; measured 67-70. The prediction holds.

Read the three rows carefully: the other-host baseline is ~3% slower on this shape than this host's
own baseline (12435 vs 12058 ms/fwd) with identical code paths, so comparing the after-run against
the doc's tables would have claimed 3.6% -- host, not the fix. Only the same-host pair is a
measurement of the change. Total compute moved just 0.7 s because VAE decode varied +2.8 s between
the two runs, which the DiT blockings cannot touch; ms/fwd is the metric that isolates them.

**CLIP is blank because both runs failed `check_audio_sanity`** -- see Open issues 4. The generation
completes and the timing table prints before that assertion, so the perf rows are valid; the gate
fires before CLIP is computed. `open_clip` is installed here, so CLIP follows a passing audio gate.
**Block device-only is TODO** because the per-op re-profile needs `test_performance_minimax_h3.py`
under Tracy, which imports the pinned `diffusers` fork this host does not have.

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
  -k "13632_5376_7168_8x8_agmm_ff1_swiglu and wh_4x8_ring" -s
```

Wormhole's compute grid is **8x9 = 72 cores** against Blackhole's 12x10, so the AGMM worker grid is
8x8 (`agmm_worker_grid` reserves the in0-mux row) rather than 12x9, and ff2's plain matmul runs on
the full 8x9. Neither grid is one Blackhole produces, so every H3 blocking the model carried for
these shapes had been swept on a grid that does not exist on the part.

### Matmul blockings — time saved against what the model ran before

| shape | op / grid | 5 s (M=4768) | 10 s (M=9216) | **15 s (M=13632)** |
|---|---|---|---|---|
| ff1 | AGMM 8x8 | 16.0% | 9.6% | **6.3%** |
| ff2 | matmul 8x9 | 15.2% | 6.8% | **4.0%** |
| qkv | AGMM 8x8 | 8.4% | 3.5% | **0.0%** |
| to_out | AGMM 8x8 | 7.5% | 2.6% | **0.4%** |
| total matmul | | 12.8% | 6.7% | **3.5%** |
| per denoise step | | 100 ms | 89 ms | **67 ms** |
| share of one forward | | 3.5% | 1.4% | **0.5%** |

Winning blockings, and the reason they are keyed per-M:

| shape | 5 s | 10 s | 15 s |
|---|---|---|---|
| ff1 | (10, 7, 10) | (12, 7, 8) | (8, 7, 10) |
| ff2 | (6, 8, 12) | (10, 8, 4) | (8, 7, 10) |
| qkv | (10, 7, 8) | (10, 7, 8) | (8, 7, 12) *(= shipped)* |
| to_out | (10, 8, 8) | (12, 8, 6) | (14, 8, 6) |

Every winner differs by duration. `AGMM_BLOCK_SIZES` is keyed on `(K, N)` alone, on the argument
that block shape does not track M — established on Blackhole's 120-core grid and **false here**:
ff2's 5 s winner `(6, 8, 12)` ranks 71st of 314 at M=9216 and is 14.6% off that length's best,
*worse than the untuned default*. Landing one duration's winner through a `(K, N)`-keyed table
would speed up 5 s and regress 10 s. `M_per_core` goes 19 -> 36 -> 54 across the three, and the
winners' `M_block` tracks it. Both entries landed are therefore keyed on `(M, K, N)`.

Only ff1 and ff2 were landed for 15 s. qkv's shipped `(8, 7, 12)` measured rank 1 of 407 -- already
optimal -- and to_out's best beat its shipped blocking by 15 us on 4327, i.e. 0.4%, inside the
~0.3% run-to-run spread measured from repeat rows.

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

First pass at 15 s (q in {256, 384, 512} x k in {256, 512}) found the shipped `(256, 512)` already
best; see the section above. That search was bounded on the wrong axis. From the CB allocation in
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
{512, 640, 768, 1024} (q=128 excluded, see below):

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

So `(256, 512)`, which `measured_sdpa_chunk_sizes[13632]` already ships, is optimal. **Chunk-size
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

### q_chunk=128 hangs the op

`q=128` is excluded from the list because it **hangs**, twice, at seq_local 13632 with k=512 — the
second time on a board freshly recovered with `tt-smi -glx_reset` and verified to open and map the
fabric, so this is not board degradation. Signature: ~6 cores spinning on the dispatch poll with
flat RSS, no I/O, no compilation; recovery needs another `glx_reset`.

Nothing static rules it out. Unlike the exp path, `use_streaming_compute` in
`ring_joint_sdpa_program_factory.cpp:1340` is just `!fp32_dest_acc_en` and does not depend on
`Sq_chunk_t`. But the same factory documents a sibling failure at lines 1388-1397, where Phase-2
reserves the full `Sq_chunk_t*vDHt` output in a single `reserve_back` and "blocks forever (deadlock
seen at q_chunk=256 causal)". Same family, different trigger. Worth a bug report with this repro;
it is not on the path to a faster 15 s, since q=128 was slower than q=192 in every feasible k.

## Perf experiments

| # | experiment | status | result |
|---|---|---|---|
| 1 | Matmul blockings, all 4 shapes x 3 durations, 8x8/8x9 grids | **done** | 3.5% of matmul time at 15 s; ff1 and ff2 landed, 0.5% of a forward |
| 2 | Fused MM/RS at 8x5/8x6/8x7 matmul grids | **done** | All worse than unfused; stays disabled |
| 3 | SDPA chunk sizes, q in {256,384,512} x k in {256,512} | **done** | Shipped `(256, 512)` already optimal; larger q L1-infeasible |
| 3b | `q_chunk=128` | **done** | Reproducibly hangs the op (2x, clean board). Not a perf path — q=128 was slower than q=192 at every feasible k — but worth reporting |
| 4 | SDPA chunk sizes, small-q / large-k (q<=256, k>=512) | **done** | Hypothesis disproved. `(192, 640)` is feasible — the first k>512 point on this shape — but 13% slower than the shipped `(256, 512)`; larger q is more per-core efficient and shrinking q raises iters/core. Chunk tuning at 15 s is exhausted. L1 envelope calibrated as a by-product |
| 5 | Re-profile the block with landed configs | blocked | **TODO** — needs the pinned `diffusers` fork; not installed here |
| 6 | Pipeline re-run: warm total, denoise, ms/fwd, CLIP | **done** (perf) | Same-host A/B: **-69.9 ms/fwd, -0.58%**, exactly the isolated-sweep prediction. CLIP **TODO**: both runs fail the audio gate before CLIP is computed (issue 4) |
| 11 | Numerics of the landed blockings (the sweep never checked) | **done** | ff2 (8,7,10) pcc 1.0000000 vs torch, identical to (8,8,8) to one bf16 ulp; ff1 (8,7,10) pcc 0.9999843 on the real SwiGLU ring, = (8,3,14) to 6 dp. Both PASS |
| 12 | Audio gate: 3.3% of samples at full scale at 15 s / 16:9 on this host | not started | **TODO** — identical 3.3% with tuned entries on and off, so not tuning. Deterministic. Leads: `audio_vae` denormalization (`latents_mean/std` in its config), and the weights resolver pins no HF revision (this host pulled `42ed227e`; the other host's is unrecorded) |
| 7 | `use_exp_ring_sdpa` on Wormhole | not started | **TODO** — gated on `is_blackhole() and sp_factor == 32`, but `exp_ring_joint_sdpa_program_factory.cpp` has no arch gate and the sp check is described in-tree as "a proxy for the 4x32 shape". On WH the other conditions already hold (`tp_factor == 4`, `exp_ring_num_passes = ceil(14/9) = 2 <= 3`). A different kernel on the op that is 70% of the block, so the largest single lever available — but it needs PCC and CLIP validation, not a timing check |
| 8 | FSDP layout conversions | not started | **TODO** — tilize/untilize go 0.13 -> 3.13 ms under FSDP, a 23x blowup and a quarter of the whole FSDP cost spent on format round-trips rather than communication. Cheapest apparent win in the breakdown |
| 9 | Ring SDPA kernel utilization | not started | **TODO** — 48.8% FPU / 35.6% math at the shipped chunk size, shown to be inherent to the kernel at this shape rather than a chunk-size miss. Work is in the kernel |
| 10 | `dit_fsdp: True` in `_PRESETS_WH` | not started | **TODO** — decision, not a measurement; costs 5.7% of the block, buys the headroom a 12 GB part needs |

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
   Consider a validity marker.

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

3. **M-keyed tuning tables key on a per-device length the pipeline never runs.** Every 768P
   table keys on 4768 / 9216 / 13632 rows/device; the pipeline runs 4736 / 9184 / 13664 and
   logs it on every run (`packed sequence ... rows/device`). The constants trace to
   `test_performance_minimax_h3.py::_packed_sizes`, which counts audio latents once where the
   pipeline packs two rows per latent (`packing.py:261`) and assumes a 512-token prompt where
   the gate runs 39. The lookups are exact-key, so nothing hit.

   *Fixed in `a07012d7d8a`.* `_packed_sizes` counts audio in rows and uses the gate's 39-token
   prompt; both the length and the padding go through new `packing.py` helpers
   (`packed_sequence_length`, `padded_sequence_length`) that the pipeline itself now uses, so the
   two cannot diverge again. `get_matmul_config`/`get_agmm_config` fall back to a same-(K, N) entry
   with the same `M_per_core` after an exact miss, which makes every table robust to a 32-row
   discrepancy and to prompt length within a padding bucket. All M literals re-keyed to
   4736 / 9184 / 13664. Confirmed on a live generation: `13664 rows/device`, and neither ff1 nor
   ff2 appears among the fallback warnings (qkv and to_out do, by design -- their shipped
   blockings measured optimal / within noise). Root cause and history:
   **`MiniMaxH3_rows_per_device_mismatch.md`**.

4. **`check_audio_sanity` fails at 15 s / 16:9 on this host: 3.3% of samples at |x| >= 0.999**
   (gate < 1%). Reproduced twice, and **identical to the decimal with the tuned blockings enabled
   and disabled**, so it is not caused by the tuning and not sensitive to matmul rounding order --
   which points at a systematic scaling issue rather than noise. The gate's comment says exactly
   that: widespread clipping "means the denormalization is wrong, not that the mix is loud". The
   other host passed this gate 18/18. Not yet investigated. The `.wav` is not saved on failure
   (the test asserts before `write_artifacts`), so the first step is capturing it. Leads:
   `audio_vae` denormalization (`latents_mean`/`latents_std`, 32 channels, in its `config.json`),
   the WH audio-decode path, and the fact that `weights_minimax_h3.py` pins no HF revision -- this
   host resolved `42ed227ee7df`, the other host's snapshot is unrecorded, so the two may differ.
   Blocks the CLIP column above.

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

## Code changes backing these numbers

All committed on `jameslee/bringup_h3_wh_galaxy`; nothing here needs local patches.

| commit | change |
|---|---|
| `88563f81db5` | Wormhole bringup: `weights_minimax_h3.py` (snapshot resolver), `_PRESETS_WH` + arch-aware `resolve_mesh_preset`, `coresident` passthrough, `_release_audio()`, `is_fsdp` on the DiT build **and** its `cache.load_model`, `_L1_SMALL_WH = 32768` / `_ring_4k`, arch-aware `log_timing_table`, HF download docs |
| `eab3dfbd599` | `has_mmrs_config` takes the device core grid and asks `resolves_fused_mmrs_config`; Wormhole falls back to matmul + `reduce_scatter_minimal_async`. Also a guard in `FusedMMRSConfig.get_params` that raises instead of deadlocking when the RS zone yields <1 worker/link |
| `2076022d032` | Wormhole rows in `GALAXY_RING` (both FSDP settings) so the block perf test can run here at all, `_BH_ONLY`/`_WH_ONLY` arch marks (both 4x8 rows ask for 32 devices, so only the arch can separate them), and `sp_simulate > 1` skips off Blackhole |
| `3a2f7fea259` | `MeshConfig.detect()` is arch-aware — it hardcoded Blackhole's 12x10 for any 32-device host, so a WH Galaxy reported 110 SDPA cores instead of 63. Plus the `minimax_h3_{5s,10s,15s}_768p` perf configs, which `measured_sdpa_chunk_sizes` cited but which were absent from the tree |
| `8cd05961cbe` | Perf sweep restated at 18/18 after the MM/RS gate fix |
| `c825d089e31` | Per-op device breakdown of one block at 15 s / 16:9, and `tools/project_block_perf.py` |
| `50908410b42` | Wormhole H3 shapes in `sweep_mm_block_sizes.py` (4 shapes x 3 durations, plus fused MM/RS at three matmul grids), `MM_SWEEP_PROFILER_DUMP_EVERY` override, `L1_BUDGET_KB` note |
| `9e97f1541bc` | `grid_88_configs` ff1 and `grid_89_configs` ff2 entries (keyed 13632 -- dead, see below); widened 15 s SDPA chunk lists; the sweep write-up |
| `e5c39cbdd47` | 15 s SDPA chunk search closed out: shipped `(256, 512)` optimal; calibrated L1 envelope; `q=128` hang recorded |
| `c0af23ba607` | Documents the rows/device mismatch: tables keyed on 13632, pipeline runs 13664 |
| `664578b377e` | Point fix: ff1/ff2 re-keyed 13632 -> 13664 |
| `a07012d7d8a` | Structural fix: `_packed_sizes` audio/text bug, `packing.py` helpers routed through the pipeline, `M_per_core` matching in `get_matmul_config`/`get_agmm_config`, every M literal re-keyed. Live-confirmed; blockings PCC-validated |

`dit_fsdp` defaults **off**, overridable with `MINIMAX_H3_DIT_FSDP`. Given these results,
`dit_fsdp: True` belongs in `_PRESETS_WH` (12 GB/chip needs the headroom far more than it needs
5.7% of the block); Blackhole at 32 GB can stay unsharded. That decision is deliberately left
open — experiment 10 above.

### A note on the L1 budget in the sweep harness

`L1_BUDGET_KB` stays at 1400 for both architectures. Scaling it to 1328 for Wormhole's smaller L1
(1,499,136 B against Blackhole's 1,572,864 B) looked prudent and was wrong: combos estimated up to
~1424 KB build and run there, and the tighter bound excluded ff1's actual optimum `(10, 7, 10)` at
1380 KB along with qkv's shipped `(8, 7, 12)` at 1352 KB — so the sweep could not measure the
baseline it was meant to beat. Both were recovered with an explicit-combo pass.
