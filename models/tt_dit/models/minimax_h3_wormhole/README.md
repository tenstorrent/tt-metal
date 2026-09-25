# MiniMax-H3 t2va on Wormhole Galaxy (4x8, 32 chips) — baseline, breakdown and per-op optimization

Re-measured 2026-09-17 on `tt-metal` @ `eab3dfbd599` (Wormhole bringup + the fused MM/RS
gate fix). Supersedes the 2026-09-16 run at `3a016b74847`, which stalled at 13/18.
Mesh param `MESH_4X8_RING_WH` (`4x8_WH`), TP=4 axis 0 / SP=8 axis 1, Ring, 4 links.
50 scheduler steps => 49 forwards. `RUN_VBENCH=0` (CLIP still gated).

Raw logs are **not** committed (too large to be useful in-tree); they were kept at
`~/h3_wormhole_results/*.log.gz` on the run host, with `parse.py` there to regenerate
these tables from any of them. Current run: `sweep_fixed.log.gz`.

This directory is organized in four parts. **Part 1** is the baseline: the end-to-end measurements taken
before any optimization experiment. **Part 2** breaks that baseline down: the Tracy per-op profile of one
transformer block, the roofline it is measured against, and the same profile re-taken on 2026-09-21 with
run-to-run statistics. **Part 3** indexes the per-op optimization documents, one per op, each written as
baseline zone breakdown -> experiments per zone -> results. **Part 4** holds everything else that was measured
on this part: the end-to-end A/B of the landed blockings, the sweeps, the experiments index, the TP/SP
configuration sweep, open issues and VBench.

## Part 1 — Baseline measurements (2026-09-17, `eab3dfbd599`, before any experiment)

### Command

```bash
TT_DIT_CACHE_DIR=~/tt_dit_cache MINIMAX_H3_DIT_FSDP=1 RUN_VBENCH=0 \
  python -m pytest models/tt_dit/tests/models/minimax_h3/test_pipeline_minimax_h3.py \
  -k "4x8_WH" -q
```

Drop `RUN_VBENCH=0` for the VBench gate (verified working, see below).
Drop `MINIMAX_H3_DIT_FSDP=1` for the unsharded baseline.

### Headline

DiT FSDP is the fix for the memory limits. Without it only 5 s fits; with it 10 s and 15 s run.

| | DiT alloc/bank | free/bank | largest contig |
|---|---|---|---|
| FSDP off | 799.5 MiB | 221.7 MiB | 221.7 MiB |
| FSDP on  | **101.8 MiB** | **919.4 MiB** | **917.7 MiB** |

7.85x reduction (SP=8 sharding), costs 5-11% denoise time, **bit-identical output**
(CLIP equal to 2 dp on all six 5 s cases).

Sweep outcome: **18/18 passed** (3 h 14 m, zero failures). The previous run reached 13/18
before an intermittent device hang blocked the rest; that hang was root-caused to the fused
MM/RS gate (the accidental Wormhole fused MM/RS path, root-caused and closed in
`eab3dfbd599`) and the 5 blocked points now all pass.

Removing the accidental fused ff2 path also made every case **2.0-4.0% faster (mean 3.0%)**.
Per-forward, old -> new: 5 s 2839 -> 2754 (21:9), 1477 -> 1417 (1:1); 10 s 6588 -> 6422 (16:9);
15 s 12700 -> 12447 (21:9). The six cases that never completed before are new measurements.

### Timings — FSDP ON (seconds unless noted)

#### 5 s / 124 frames

| aspect | canvas | MPix | cold | warm | enc | denoise | vae | audio | ms/fwd | realtime | CLIP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 21:9 | 1536x672 | 1.03 | 160.9 | 146.6 | 3.5 | 134.9 | 5.8 | 2.4 | 2754 | 28.4x | 35.95 |
| 16:9 | 1344x768 | 1.03 | 161.3 | 147.6 | 3.4 | 135.3 | 5.1 | 3.7 | 2761 | 28.6x | 37.42 |
| 9:16 | 768x1344 | 1.03 | 161.0 | 146.7 | 3.4 | 134.4 | 5.7 | 3.1 | 2744 | 28.4x | 37.02 |
| 4:3 | 1024x768 | 0.79 | 120.4 | 104.9 | 3.4 | 94.0 | 3.7 | 3.8 | 1919 | 20.3x | 37.26 |
| 3:4 | 768x1024 | 0.79 | 118.7 | 105.3 | 3.5 | 94.2 | 4.0 | 3.5 | 1922 | 20.4x | 36.54 |
| 1:1 | 768x768 | 0.59 | 92.7 | 79.3 | 3.5 | 69.4 | 3.0 | 3.4 | 1417 | 15.3x | 36.32 |

#### 10 s / 243 frames

| aspect | canvas | cold | warm | enc | denoise | vae | audio | ms/fwd | realtime | CLIP |
|---|---|---|---|---|---|---|---|---|---|---|
| 21:9 | 1536x672 | 349.0 | 332.8 | 3.5 | 314.2 | 11.6 | 3.5 | 6413 | 32.9x | 33.84 |
| 16:9 | 1344x768 | 344.6 | 333.4 | 3.5 | 314.7 | 11.0 | 4.4 | 6422 | 32.9x | 36.92 |
| 9:16 | 768x1344 | 344.7 | 332.8 | 3.5 | 315.0 | 10.6 | 3.8 | 6428 | 32.9x | 36.40 |
| 4:3 | 1024x768 | 255.7 | 240.5 | 3.5 | 226.3 | 7.5 | 3.2 | 4618 | 23.8x | 37.09 |
| 3:4 | 768x1024 | 252.6 | 241.3 | 3.5 | 226.5 | 7.4 | 3.8 | 4623 | 23.8x | 36.89 |
| 1:1 | 768x768 | 173.8 | 159.2 | 3.5 | 146.4 | 5.8 | 3.6 | 2988 | 15.7x | 37.54 |

#### 15 s / 362 frames

| aspect | canvas | cold | warm | enc | denoise | vae | audio | ms/fwd | realtime | CLIP |
|---|---|---|---|---|---|---|---|---|---|---|
| 21:9 | 1536x672 | 648.5 | 633.2 | 3.4 | 609.9 | 16.0 | 3.9 | 12447 | 42.0x | 35.27 |
| 16:9 | 1344x768 | 644.3 | 633.1 | 3.5 | 609.3 | 15.8 | 4.6 | 12435 | 42.0x | 36.31 |
| 9:16 | 768x1344 | 645.7 | 633.8 | 3.5 | 610.7 | 15.2 | 4.3 | 12464 | 42.0x | 35.50 |
| 4:3 | 1024x768 | 457.3 | 434.6 | 3.3 | 416.0 | 10.9 | 4.3 | 8491 | 28.8x | 36.01 |
| 3:4 | 768x1024 | 447.6 | 434.9 | 3.3 | 416.6 | 11.2 | 3.8 | 8501 | 28.8x | 36.40 |
| 1:1 | 768x768 | 281.8 | 266.6 | 3.5 | 251.2 | 8.5 | 3.6 | 5126 | 17.7x | 38.26 |

### Timings — FSDP OFF (baseline, for the 5 s comparison)

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

### Scaling notes

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

## Part 2 — Baseline breakdown (before improvements)

### Per-op device breakdown — one transformer block, 15 s / 16:9 (Tracy, 2026-09-17)

Measured 2026-09-17 on `bc1d99d05f6`: Wormhole `GALAXY_RING` rows, the re-keyed 15 s ff1/ff2
blockings, and `_packed_sizes` producing the length the pipeline runs. One block, warm iteration
between `start`/`stop` signposts, both FSDP settings.

```bash
scripts/run_safe_pytest.sh --profile \
  "'models/tt_dit/tests/models/minimax_h3/test_transformer_minimax_h3.py::test_minimax_h3_transformer_block_perf[wormhole_b0-sp_sim1-test_prompt_text_tokens-15s_768p-4x8sp1tp0nl4_ring_is_fsdp1]'" \
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
| *(remaining 8 ops — adaLN modulate/gate, head split/merge, slice/typecast; itemised under [The small ops](#the-small-ops--whole-block-roofline-per-op-2026-09-23))* | 17 | 2.81 | 2.78 | +0.04 |
| **device only** | | **246.92** | **233.41** | **+13.51 (5.8%)** |
| **device + op gap** | | **250.80** | **238.46** | +12.34 |
| SDPA share of block | | 70.7% | 74.8% | |

Projected over 50 layers: **12.35 s/step** device-only, 12.54 s with op gaps (617 / 627 s per
50-step video). The 16:9/15 s row above measures the same configuration end-to-end at
**12435 ms/forward**, which lands inside that bracket exactly as `project_block_perf.py` intends:
`device only` is the underestimate (no dispatch gaps) at 99.3% of the forward, `device + op gap`
the overestimate at 100.8%. The 50-layer block stack is therefore the whole forward to within the
bracket's width; the refiner, input projections, `norm_out` and output heads fit in the remainder.

#### Findings

1. **Ring SDPA is 71% of the block** at 48.3% FPU utilization (`PM FPU UTIL (%)`, 47.5-48.7 across
   all 32 devices). Nothing else is close: the three fused all-gather+matmuls are 13%, the six adaLN
   embedding gathers 4%, the two plain matmuls 3.5%. The SDPA work is in [sdpa.md](sdpa.md); the three AGMMs are
   [ff1.md](ff1.md), [to_qkv.md](to_qkv.md) and [to_out.md](to_out.md).
2. **FSDP costs 5.8%** (13.51 ms/block) — inside the 5-11% the pipeline sweep saw. Three ops exist
   only with FSDP on: AllBroadcast 3.99 + AllGatherAsync 3.96 + Concat 1.51 = 9.46 ms. The rest is
   layout conversion (+3.00, next item), a slightly larger fused AG+matmul (+0.64) and RMSNorm (+0.35).
3. **Layout conversions grow 24x under FSDP** — tilize + untilize go 0.13 -> 3.12 ms, 22% of the
   FSDP cost spent on format round-trips rather than communication. Cheapest apparent win.
4. **ff2 ran the unfused Wormhole path** in this baseline: `ReduceScatterMinimalAsyncDeviceOperation` is present
   and there is no fused `Matmul_RS` row. Since 2026-09-23 the 15 s shape takes a swept fused
   `minimal_matmul_strided_reduce_scatter_async` instead (matmul + reduce-scatter + addcmul in one op, -1.76 ms per
   layer like-for-like). See [ff2.md](ff2.md).

### Roofline — speed of light on this part (derived 2026-09-17)

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

#### Cross-check against the Tracy block breakdown (15 s, per device, per layer)

Same arithmetic per op, on the grid each op actually runs on:

| op | FLOP | min at HiFi2 peak on its grid | measured | util |
|---|---|---|---|---|
| [RingJointSDPA](sdpa.md) (63 cores) | 10.71 T | 83.0 ms | 174.6 ms | **48%** — Tracy's `PM FPU UTIL` reads 47.5-48.7% |
| [ff1 AGMM](ff1.md) (8x8) | 1.05 T | 8.0 ms | 15.7 ms | 51% |
| [qkv AGMM](to_qkv.md) (8x8) | 0.79 T | 6.0 ms | 10.4 ms | 58% |
| [to_out AGMM](to_out.md) (8x8) | 0.26 T | 2.0 ms | 4.3 ms (the plain sweep number; 5.3 ms with the fused epilogue the model runs) | 46% (38%) |
| [ff2 matmul](ff2.md) (8x9) | 0.53 T | 3.6 ms | 6.8 ms | 53% |
| **block** | **13.34 T** | **90 ms** (72 cores) | **247 ms** | **36%** |

The derived SDPA utilization lands on the profiler's FPU-utilization counter exactly, which
validates both the FLOP count and the 2048 FLOP/cycle/core constant.

#### Nothing but the FPU binds

Per device per layer at 15 s, against 4 links x 12.5 GB/s = 50 GB/s of ring ingress
(`tech_reports/EthernetMultichip/BasicEthernetGuide.md`):

| traffic | bytes | time at 50 GB/s | overlaps |
|---|---|---|---|
| KV ring all-gather (14 heads x 7/8 x S x 128 x 2 x bf16) | 686 MB | ~14 ms | 175 ms of SDPA |
| three AGMM activation gathers | ~370 MB | ~7 ms | 30 ms of matmul |
| FSDP weight gather (7/8 x 193 MB bf16) | ~170 MB | ~3.4 ms | (measured FSDP cost 13.5 ms is mostly the extra ops + layout, not the bytes) |
| DRAM: 193 MB weights + ~1-2 GB activation passes at 288 GB/s | | < 10 ms | everything |

Even at 100% FPU the collectives sit under compute by 5-10x. The floor is the matrix engine.

#### The small ops — whole-block roofline, per op (2026-09-23)

The per-op cross-check above stops at the five compute ops. `tools/transformer_roofline.py` whole-block mode
(`load_block_profile`, `transformer_roofline.py:1027`) reads the same 09-17 fsdp1 CSV (`transformer_roofline.py:914`),
gives every op code a bound by class and folds every op group under 1% of the block (`OTHER_SHARE`,
`transformer_roofline.py:925`) into one "other (small ops)" row on the two block figures. That row is 9.12 ms,
3.7% of the block, 12 op groups, 48 calls — larger than any single non-SDPA op except ff1 and to_qkv, so it gets
its own figure (`fig_block_other`, `transformer_roofline.py:1320`) and indented sub-rows in the `--dump` table.
Every op in it moves DRAM-resident tensors once, so the bound is bytes in + out / 288 GB/s (`DRAM_BOUND_NAMES`,
`transformer_roofline.py:938`), except that an op averaging under 10 µs of bytes per call sits at the kernel-launch
floor and stays measured only (`LAUNCH_FLOOR_S`, `transformer_roofline.py:929`); the adaLN matmul keeps its FLOP +
DRAM model and is DRAM-bound at M=32.

```bash
python models/tt_dit/tests/models/minimax_h3/tools/transformer_roofline.py --dump --figs block_stacked,block_ops,block_other
# writes transformer_roofline_out/block_{stacked,ops,other}_wh_M13664.png; --profile-csv <csv> for a newer profile
```

| op group | calls | measured ms | ideal ms | util | headroom | bound |
|---|---|---|---|---|---|---|
| UntilizeWithUnpadding | 21 | 1.764 | 0.935 | 53% | 1.9x | 269 MB / 288 GB/s |
| MinimalMatmul M=32 (adaLN) | 1 | 1.672 | 0.463 | 28% | 3.6x | 2·32·2688·24192 FLOP on 72 cores; 133 MB DRAM |
| Concat | 1 | 1.511 | 0.903 | 60% | 1.7x | 260 MB |
| TilizeWithValPadding | 8 | 1.355 | 0.907 | 67% | 1.5x | 261 MB |
| BinaryNg (adaLN modulate) | 4 | 0.997 | 0.512 | 51% | 1.9x | 147 MB |
| Ternary (adaLN gate + residual) | 1 | 0.629 | 0.510 | 81% | 1.2x | 147 MB |
| NLPConcatHeads (heads merge) | 1 | 0.573 | 0.340 | 59% | 1.7x | 98 MB |
| NlpCreateHeads (q/k/v split) | 1 | 0.568 | 0.340 | 60% | 1.7x | 98 MB |
| Slice, Typecast, ReshapeView, Unary | 10 | 0.047 | — | — | — | < 1 MB per call: launch-latency floor, no bandwidth model |
| **other (small ops)** | **48** | **9.12** | **4.91** | **54%** | **1.86x** | 99% of the group has a bound |

Reading it:

1. **The FSDP layout round-trips are half the group** — untilize + tilize + concat = 4.63 ms against a 2.75 ms
   DRAM floor. The 24x blowup in Finding 3 is a count problem (21 untilize calls, 8 tilize) more than a per-call
   one: each call already runs at 53-67% of DRAM bandwidth. Removing the round-trips, not speeding them up, is the
   lever (experiment 8 in the index below).
2. **adaLN costs 3.3 ms outside the embeddings** — the M=32 modulation matmul at 28% (a 32-row matmul cannot fill
   72 cores; 3.6x headroom, the worst ratio in the group), four BinaryNg modulates and one Ternary gate+residual,
   each a full pass over the 36.7 MB activation. The Ternary at 81% is the best-utilised op in the block. The
   modulates and the gate are candidates for fusing into the neighbouring RMSNorm / matmul epilogues rather than
   tuning in place.
3. **The head split and merge around SDPA are 1.14 ms** of pure data movement at 60% of DRAM; a head-major
   to_qkv writer or an SDPA that reads the [S, heads·d] layout directly would remove both.
4. **The tail is noise** — slice, typecast, reshape and the single unary total 47 µs across 10 calls, 3-8 µs each:
   kernel-launch cost, not bandwidth, so the tool gives them no ideal.

The group's own headroom (1.86x) is lower than the block's (2.26x, `--dump` footer): these ops are closer to
their floor than the matmuls and SDPA are, so the ~4 ms recoverable here comes from removing ops, not from
tuning them.

### Baseline re-measured, and run-to-run variance (2026-09-21)

The same two profiles re-taken on this host at `d4fca5e3f23` + the working tree (the ff1 silu landing and the SDPA pack-4
commit are in; nothing else of the block changed), six runs each, ~2 min per run. New report
directories only; the 09-17 CSVs the table above came from are untouched (report `2026_09_17_21_33_20` fsdp1, `2026_09_17_21_35_04` fsdp0). Tool:
`tools/block_profile_stats.py compare|runs|devices` (same signpost isolation and device merge as
`project_block_perf.py`).

**Against the 09-17 table** (one run each side; today = the first of the six):

| op | 09-17 fsdp1 | 09-21 fsdp1 | delta | 09-17 fsdp0 | 09-21 fsdp0 | delta |
|---|---|---|---|---|---|---|
| RingJointSDPA | 174.56 | 172.15 | **-2.40** | 174.57 | 172.21 | **-2.36** |
| AllGatherMinimalMatmulAsync (3) | 32.25 | 31.76 | **-0.49** | 31.61 | 31.24 | -0.37 |
| AllGatherAsync (FSDP, 4) | 3.96 | 4.27 | +0.31 | — | — | |
| every other op | | | within ±0.08 | | | within ±0.05 |
| **device only** | **246.92** | **244.25** | **-2.67 (-1.1%)** | **233.41** | **230.82** | **-2.59 (-1.1%)** |
| device + op gap | 250.80 | 249.24 | -1.56 | 238.46 | 238.16 | -0.31 |

The two drops are the two landings since 09-17: the blocked pack at width 4 in the SDPA inner loop
([sdpa.md](sdpa.md), inner-loop experiment A; the op went 192.9 -> 191.6 ms on the padded shard), and the bf16-grade silu in the ff1 epilogue ([ff1.md](ff1.md); -0.56 ms per call on the mesh bench). Projected over 50 layers the block is 12.21 s/step device-only (was 12.35).

**Run-to-run, same host and commit** (6 runs per setting; sample std):

| | fsdp1 mean ms | std | max-min | CoV | fsdp0 mean | std | max-min | CoV |
|---|---|---|---|---|---|---|---|---|
| RingJointSDPA | 172.16 | 0.031 | 0.09 | 0.02% | 172.20 | 0.039 | 0.11 | 0.02% |
| AllGatherMinimalMatmulAsync (3) | 31.85 | 0.086 | 0.22 | 0.27% | 31.22 | 0.031 | 0.09 | 0.10% |
| Embeddings (6) | 10.40 | 0.010 | 0.03 | 0.10% | 10.35 | 0.001 | 0.00 | 0.01% |
| MinimalMatmul (2) | 8.56 | 0.027 | 0.06 | 0.32% | 8.59 | 0.029 | 0.08 | 0.34% |
| AllGatherAsync (FSDP, 4) | 4.30 | 0.100 | 0.27 | 2.33% | — | | | |
| AllBroadcast (FSDP) | 3.95 | 0.065 | 0.17 | 1.65% | — | | | |
| DitFusedDistributedRmsnorm (4) | 2.91 | 0.058 | 0.17 | 2.00% | 2.66 | 0.036 | 0.10 | 1.37% |
| ReduceScatterMinimalAsync | 2.77 | 0.013 | 0.03 | 0.46% | 2.82 | 0.028 | 0.07 | 0.99% |
| NlpCreateHeads / NLPConcatHeads | 0.5-0.6 | 0.01-0.03 | 0.04-0.07 | 3-5% | 0.5-0.6 | 0.01 | 0.02-0.03 | 1-2% |
| **device only** | **244.35** | **0.20** | **0.60** | **0.08%** | **230.76** | **0.065** | **0.15** | **0.03%** |
| device + op gap | 247.97 | 0.82 | 2.09 | 0.33% | 236.80 | 1.16 | 2.55 | 0.49% |

Device time is repeatable to 0.2 ms in 244 (SDPA to 0.03 ms, the three AGMMs together to <0.1 ms); the FSDP
collectives and the small per-head ops are the noisy rows at 2-5% of their own value, still <0.1 ms absolute. The
host dispatch gaps are 5-15x noisier than the device time, which is why the forward is bracketed by `device only`
and `device + op gap` rather than quoted as one number.

**Across the 32 devices within one run** (fsdp1, 09-21 r0; the 09-17 run shows the same pattern, so this is
structural, not noise). The merge decides which number the table shows: mean over devices for collectives
(the AGMM row), max otherwise.

| op | mean over devices | slowest device | fastest device | spread |
|---|---|---|---|---|
| RingJointSDPA | 172.10 | 172.15 | 172.06 | 0.1 ms: every device in lockstep |
| AllGatherMinimalMatmulAsync (3, reported as mean) | 31.76 | 33.69 | 30.20 | 3.5 ms, 11% |
| Embeddings (6, reported as max) | 10.14 | 10.41 | 8.83 | 1.6 ms, 18% |
| AllGatherAsync (4, mean) | 4.27 | 7.37 | 2.67 | 2.75x |
| DitFusedDistributedRmsnorm (4, max) | 2.60 | 2.91 | 2.41 | 21% |

Rule of thumb for this table: a per-op change under ~0.1 ms (under ~0.3 ms for the FSDP collectives) is not
resolvable from one pair of runs; anything over 0.5 ms is. The -2.67 ms above is 13 run-to-run standard deviations
and the 09-17 total sits 2.2 ms above today's slowest run: real. The +0.31 ms on AllGatherAsync is 3 of its own
standard deviations with the 09-17 value just below today's range: probably real, small, worth a second host.

Reproduce: the `run_safe_pytest.sh --profile` command above, once per setting and run (each writes a new
`generated/profiler/reports/<ts>/`), then
`python models/tt_dit/tests/models/minimax_h3/tools/block_profile_stats.py runs r0=<csv> r1=<csv> ...`,
`... compare baseline=<09-17 csv> today=<csv>`, `... devices <csv>`.

### Per-op durations are not additive under FSDP (2026-09-24)

With `dit_fsdp` the weight all-gathers (`AllGatherAsync`) run on the CCL sub-device concurrently with the matmul that
consumes them: on every device to_out's gather launches 0.12 ms before to_out and finishes inside its window, and to_out's
kernel duration is compute plus whatever part of the gather it waits for. That makes to_out 5.15 ms on mesh rows 3 and 5 and
7.2 ms on rows 0 and 6 with identical work, and it lets a timing shift upstream move time between the two rows of the table
without changing the layer's wall time. For any comparison that shifts timing, use the **device-busy union per layer**
(`tools/block_device_busy.py`: the union of every op's device interval, repeatable to 0.1 ms; 238.1 ms today) rather than the
"device only" sum, which counted 240.8 for the same profile and read a real -1.2 ms as zero (ff1.md §3.4).

### Per-op: baseline vs current

The 09-17 baseline column is the `c825d089e31` breakdown at `fsdp1`; the current column is the mean of the six 2026-09-21
runs above (same host, `d4fca5e3f23` + working tree). Experiment 5's full re-profile with the landed *configurations*
(the ff1/ff2 blockings) is still blocked on the pinned `diffusers` fork; the blockings themselves were measured
end-to-end in Part 4.

| op | baseline ms (09-17) | current ms (09-21, 6-run mean) | delta | where the change came from |
|---|---|---|---|---|
| RingJointSDPADeviceOperation | 174.56 | 172.16 | -2.40 | inner-loop pack-4, [sdpa.md](sdpa.md) |
| AllGatherMinimalMatmulAsyncOp (3) | 32.25 | 31.85 | -0.40 | bf16-grade silu in ff1, [ff1.md](ff1.md) |
| MinimalMatmulDeviceOperation (2) | 8.54 | 8.56 | +0.02 | — |
| *(all others)* | 31.57 | 31.78 | +0.21 | FSDP collectives, within their run-to-run noise |
| **device only** | **246.92** | **244.35** | **-2.57 (-1.0%)** | |

Same-session single-run A/B of the 2026-09-23 K-loop MVMUL reorder ([kloop_refill_reorder_handoff.md](kloop_refill_reorder_handoff.md)), device ms per block:
fsdp1 three AGMMs 31.53 -> **31.02**, fused MM/RS 8.62 -> **8.21**, device only 242.77 -> 241.85 (-0.92); fsdp0 31.28 -> 30.48, 8.69 -> 8.37,
229.11 -> 227.87 (-1.24). Numerics bit-identical.

### Roofline with every optimization on — one block, 15 s / 16:9 (2026-09-24)

Experiment 5 (the re-profile with the landed configurations) with the two measured-but-not-landed precision levers also
switched on, as one Tracy profile: same host (`UF-EV-B12-GWH02`), `f1575653d72` + the working tree (the K-loop MVMUL reorder
in `matmul_block_kloop`), `fsdp1`, report `2026_09_24_02_51_06`. Same profiling command as Part 2; the roofline is

```bash
python models/tt_dit/tests/models/minimax_h3/tools/transformer_roofline.py --dump --figs block_stacked,block_ops,block_other \
  --profile-csv generated/profiler/reports/2026_09_24_02_51_06/ops_perf_results_2026_09_24_02_51_06.csv --out-dir transformer_roofline_out/all_on
python models/tt_dit/tests/models/minimax_h3/tools/block_profile_stats.py compare baseline=<09-17 csv> all_on=<csv>
python models/tt_dit/tests/models/minimax_h3/tools/block_device_busy.py baseline=<09-17 csv> all_on=<csv>
```

#### How to run with every optimization on (handoff)

Everything that is landed is on by default: check out the branch, build (`build_metal.sh` recreates `python_env`; reinstall the
pinned MiniMax-H3 `diffusers` fork with `uv` afterwards, or the perf test and the pipeline fail at import), and set the two
exploration switches. The LUT switch is host code, so an incremental rebuild needs the install step as well
(`ninja -C build install`, or `cmake --install build_Release` after `ninja -C build`): Python loads `build/lib/_ttnncpp.so`,
the *installed* copy, and a bare `ninja -C build` leaves it stale. Both switches are read once at model construction, so
they must be in the environment of the process that builds the model:

```bash
# the two switches; leave either unset to run the production numerics for that op
export MINIMAX_H3_MM_FP32_DEST=0      # ff1 (8,7,16) 2x4 and to_qkv (12,7,8) 4x2 with fp32 dest accumulation off
                                      # ("ff1" or "qkv" selects one of them; unset or "1" = production, fp32 dest on)
export TT_MM_SWIGLU_LUT_SILU=1        # LUT-sigmoid SwiGLU epilogue in every fused-SwiGLU matmul kernel (Wormhole only)

# one block, per-op device profile (2-3 min; prints "SAFE_PYTEST: PROFILER CSV: <csv>")
scripts/run_safe_pytest.sh --profile \
  "'models/tt_dit/tests/models/minimax_h3/test_transformer_minimax_h3.py::test_minimax_h3_transformer_block_perf[wormhole_b0-sp_sim1-test_prompt_text_tokens-15s_768p-4x8sp1tp0nl4_ring_is_fsdp1]'" \
  -s --timeout 3600
python models/tt_dit/tests/models/minimax_h3/tools/transformer_roofline.py --dump --figs block_stacked,block_ops,block_other \
  --profile-csv <csv> --out-dir transformer_roofline_out/all_on

# the 15 s / 16:9 video, 50 steps, with the CLIP gate (~25 min with the weight cache; artifacts in ~/h3_t2va_artifacts/)
TT_DIT_CACHE_DIR=~/tt_dit_cache MINIMAX_H3_DIT_FSDP=1 RUN_VBENCH=0 \
  python -m pytest "models/tt_dit/tests/models/minimax_h3/test_pipeline_minimax_h3.py::test_t2va_end_to_end[wormhole_b0-4x8_WH-16x9_15s]" \
  -q -s --timeout 7200
```

Two checks that the switches took: the profile's `AGMM ff1` row reads ~12.1 ms (15.7 baseline, 14.5 with both switches off), and
the freshly built SwiGLU kernel's `defines_generated.h` under `~/.cache/tt-metal-cache/<key>/kernels/compute/<hash>/` contains
`SWIGLU_LUT_SILU` (`grep -l SWIGLU_LUT_SILU ~/.cache/tt-metal-cache/*/kernels/compute/*/defines_generated.h`). Neither switch
changes the roofline; both are precision decisions (item 5 below), which is why they are switches and not defaults. Drop
`--profile` and `-s` for a plain run; `pytest.ini`'s 300 s timeout is too short for either command.

What is on, and how:

| optimization | how it was switched on |
|---|---|
| SDPA inner-loop pack-4, bf16-grade silu, swept ff1/ff2 blockings, fused ff2 MM+RS+addcmul (8x7 grid, (6,7,8) 2x2), K-loop MVMUL reorder | default in the tree, nothing to set |
| fp32 dest off for ff1 ((8,7,16) 2x4) and to_qkv ((12,7,8) 4x2) | `MINIMAX_H3_MM_FP32_DEST=0` (`transformer_block_minimax_h3.py:162`, `attention_minimax_h3.py:236`) |
| LUT-sigmoid SwiGLU epilogue in ff1 | `TT_MM_SWIGLU_LUT_SILU=1`: the fused-SwiGLU program factories add the `SWIGLU_LUT_SILU` define (`compute_throttle_utils.cpp`, `add_swiglu_lut_silu_define_if_needed`), which is part of the kernel hash, so LUT and exact builds coexist in the kernel cache. The profile itself predates the switch by two hours: it forced the guard in `swiglu_lut.hpp:39` and parked the compute-kernel JIT cache, since `Kernel::compute_hash` (`tt_metal/impl/kernels/kernel.cpp:556`) hashes defines and compile-time args, not header text. Verified on the binaries: only the `FUSE_SWIGLU` kernel's math TRISC changed (text 6092 -> 4004 B). Reproduced through the env var once the switch existed (report `2026_09_24_04_34_24`): ff1 12.13 ms, union 235.47 ms, `SWIGLU_LUT_SILU` in the kernel's `defines_generated.h` |

Per op, merged as the roofline merges (mean over devices for collectives, max otherwise). Ideal = HiFi2 speed of light on
the op's own grid. Baseline is the 09-17 `fsdp1` profile of Part 2.

| op | ideal ms | baseline ms | **all on ms** | delta | FPU util baseline -> all on | headroom baseline -> all on |
|---|---|---|---|---|---|---|
| RingJointSDPA | 82.98 | 174.56 | **172.14** | -2.42 | 48% -> 48% | 2.10x -> 2.07x |
| AGMM ff1 | 8.03 | 15.70 | **12.11** | **-3.59** | 51% -> **66%** | 1.95x -> **1.51x** |
| AGMM to_qkv | 6.03 | 10.26 | **9.68** | -0.58 | 59% -> 62% | 1.70x -> 1.61x |
| ff2: plain matmul + RS + gated residual -> fused MM+RS | 3.57 (72 cores) / 4.59 (56-core fused) | 6.86 + 2.78 + 0.63 = 10.27 | **8.24** | **-2.03** | 52% -> 56% (of the smaller grid) | 1.92x -> 1.79x |
| AGMM to_out | 2.01 | 6.29 | **6.11** | -0.18 | 32% -> 33% | 3.13x -> 3.04x |
| everything else (embeddings, FSDP collectives, norms, layout, small ops) | | 29.84 | **29.99** | +0.15 | | |
| **device only** (sum) | 109.1 (sum of ideals) | **246.92** | **238.27** | **-8.65 (-3.5%)** | 36% -> 37% | **2.26x -> 2.18x** |
| **device-busy union** (`block_device_busy.py`) | | **244.05** | **235.34** | **-8.71 (-3.6%)** | | |
| device + op gap | | 250.80 | 240.37 | -10.43 | | |
| per forward, 50 layers, device only | 5.45 s | 12.35 s | **11.91 s** | -0.44 s | | |

Reading it:

1. **All on is -8.71 ms per layer of device-busy wall time (-3.6%)**, -8.65 ms of `device only`; over 50 layers that is
   -0.44 s per denoise step, 12.35 -> 11.91 s projected device-only (the baseline row of the 15 s table above is 12435 ms/fwd).
   The block's roofline headroom moves from 2.26x to 2.18x: the whole gain is on the matmul class, and the ideal is unchanged.
2. **ff1 becomes the best-utilised matmul in the block**: 15.70 -> 12.11 ms, 51% -> 66% of HiFi2 peak, 1.51x from its
   roofline; to_qkv sits at 62%, the fused ff2 at 56% of its 56-core grid, to_out stays at 33% (delivery co-limited,
   [to_out.md](to_out.md)). ff2's reduce-scatter and gated-residual Ternary disappear as separate ops into the fused row.
3. **SDPA does not move beyond pack-4 and is now 72% of the block** (172.14 of 238.27); at 48% of peak it holds 89 of the
   129 ms of headroom that remain. Every further percent of the forward is in `compute_streaming.hpp` ([sdpa.md](sdpa.md)).
4. **Nothing else moved**: embeddings, norms, layout conversions and the small ops are within 0.1 ms of the baseline; the
   FSDP `AllGatherAsync` is +0.2 ms, inside its own run-to-run band (0.3 ms, 3 std).
5. **The two precision levers are measured here, not landed.** The fp32-off switch defaults to on (ff1.md §3.4: -0.84 ms/layer
   at production-level CLIP, 50-step A/B) and the LUT is opt-in through `TT_MM_SWIGLU_LUT_SILU=1`
   ([ff1_swiglu_lut_handoff.md](ff1_swiglu_lut_handoff.md) §4). The end-to-end video below (Part 4, *Optimization target*)
   carries both: **CLIP 35.90** (min 34.70) against 35.75 production on this host, 12015.6 ms/fwd. VBench with the LUT enabled has
   not been run. Adopting either is the owner's call.

Figures: `transformer_roofline_out/all_on/block_{stacked,ops,other}_wh_M13664.png`; the baseline figures stay in
`transformer_roofline_out/`.

### From the breakdown to the per-op work

1. Speed of light at HiFi2 is ~2.75x today's 15 s forward and ~3.7x the 5 s one; a realistic 70%-util target is
   ~6.5 s/fwd at 15 s (317 s, 21x RT) and ~1.1 s/fwd at 5 s (52 s, 10x RT).
2. Ring SDPA is 71-80% of the FLOPs at 48% util. Chunk tuning is exhausted and the remaining ~2x on that op is
   kernel work: the inner-loop zones and experiments are in [sdpa.md](sdpa.md).
3. The three AGMMs run at 46-58% of peak with the same 2x2 fp32 pipeline pace; ff1 adds a serialized SwiGLU
   epilogue, to_qkv a chunked writer, to_out an addcmul epilogue and operand waits on the relay:
   [ff1.md](ff1.md), [to_qkv.md](to_qkv.md), [to_out.md](to_out.md). ff2's plain matmul and its blocking sweep
   are in [ff2.md](ff2.md).
4. 5 s is occupancy-limited, not FLOP-limited: 27% util against 36% at 15 s, from SDPA slot waste (q=320 idles
   16.7% of the 63 slots) and small per-device M in the matmuls.
5. Going below HiFi2 (bfp8 / LoFi operands) doubles the ceiling again, to ~2.3 s/fwd at 15 s, but that is a
   quality decision rather than a tuning one; the same precision decision recurs per op (fp32 dest off, LUT silu).

## Part 3 — Per-op optimization documents

One document per op, each ordered the same way: the op and its baseline, the Tracy zone breakdown of the
unoptimized op, the experiments proposed for each zone with their results, the results ladder, what is left, and
the tooling. Numbers are per call, per device, at 15 s / 768P / 16:9.

| op | share of block | baseline (09-17 block) | roofline | current | status | doc |
|---|---|---|---|---|---|---|
| ring joint SDPA | 70.7% | 174.56 ms | 83.0 ms | 172.16 ms | pack-4 landed; softmax half of the inner loop is the lever | [sdpa.md](sdpa.md) |
| ff1 AGMM (fused SwiGLU) | 6.4% | 15.7 ms | 8.03 ms | 15.29 ms | bf16 silu landed (-3.6%); K-loop MVMUL reorder landed 2026-09-23 (-0.35 ms per call on the mesh bench, [kloop_refill_reorder_handoff.md](kloop_refill_reorder_handoff.md)); fp32 dest off + 2x4 (-8%) and a 6-segment LUT silu (-9.2%) measured, both precision decisions, not landed | [ff1.md](ff1.md) |
| to_qkv AGMM (chunks=3) | 4.2% | 10.4 ms | 6.03 ms | 10.30 ms | behaves like ff1; K-loop reorder landed 2026-09-23 (11.25 -> 11.03 ms per call, fp32 dest on); fp32 dest off + 4x2 -8.7% measured, not landed | [to_qkv.md](to_qkv.md) |
| to_out AGMM (fused addcmul) | 2.1% | 5.3 ms (4.33 in the plain sweep) | 2.01 ms | 5.29 ms | delivery co-limited (relay waits 1.1 ms) + two-pass epilogue 0.7 ms; compute-side levers do not transfer | [to_out.md](to_out.md) |
| ff2 matmul + reduce-scatter + addcmul | 2.7% + 1.1% + ~0.3% | 10.70 ms host-timed (7.45 + 2.54 + 0.71) | 4.59 ms (56-core matmul) | **8.94 ms fused** (8.68 after the K-loop reorder, 2026-09-23) | fused MM/RS re-evaluated like-for-like at M=13664 and landed 2026-09-23 (8x7 matmul, (6,7,8) 2x2, L1 window): -1.76 ms/layer, -1.88 ms/block in the Tracy profile, **-166 ms/fwd (-1.35%) in a 10-step same-host A/B**, CLIP 35.58 vs 35.71; unfused RS hyperparameters measured (-5%), not landed | [ff2.md](ff2.md) |

## Part 4 — Other findings

### Optimization target — 15 s / 768P / 16:9 (end-to-end A/B, 2026-09-17)

Tuning work is scoped to this one configuration. Baseline is `c825d089e31` (the per-op breakdown
in Part 2), measured at `fsdp1`.

| | warm total | denoise | ms/fwd | realtime | CLIP | block device-only |
|---|---|---|---|---|---|---|
| baseline, other host (`c825d089e31` tables) | 633.1 s | 609.3 s | 12435 | 42.0x | 36.31 | 246.31 ms |
| baseline, **this host**, tuned entries disabled | 612.9 s | 590.9 s | 12058.3 | 40.6x | — | **TODO** |
| **best found**, this host, `a07012d7d8a` | **612.2 s** | **587.4 s** | **11988.4** | 40.6x | — | **TODO** |
| best found, this host, later run (rebuilt weight cache) | 621.0 s | 599.2 s | 12230 | 41.2x | **35.88** (min 34.69, bar 33.0) | **TODO** |
| **every optimization on**, this host, `f1575653d72` + tree, 2026-09-24 (tree default + `MINIMAX_H3_MM_FP32_DEST=0` + LUT SwiGLU; Part 2 *Roofline with every optimization on*) | 611.9 s | **588.8 s** | **12015.6** | 40.6x | **35.90** (min 34.70, max 37.01) | **238.27 ms** (union 235.34) |

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

### Sweeps run

All matmul numbers come from `models/tt_dit/utils/sweep_mm_block_sizes.py` against device config
`wh_4x8_ring` (4 links, 4 KB router payload, Ring). 4749 measured rows, all durations.

```bash
# One shape. MM_SWEEP_PROFILER_DUMP_EVERY is mandatory on a WH Galaxy -- see Open issues, item 2.
MM_SWEEP_PROFILER_DUMP_EVERY=100000 python -m pytest \
  models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
  -k "13664_5376_7168_8x8_agmm_ff1_swiglu and wh_4x8_ring" -s
```

A second sweep set, run from the AGMM unit test rather than the harness, compares the fused op against the unfused
all-gather + matmul pair with each step swept on its own knobs (all-gather hyperparameters, matmul blocking on
8x8 and on the model's 8x9), at 4736 / 9184 / 13664 rows per device, with a PCC per combo:
[agmm_fused_vs_unfused.md](agmm_fused_vs_unfused.md).

Wormhole's compute grid is **8x9 = 72 cores** against Blackhole's 12x10, so the AGMM worker grid is
8x8 (`agmm_worker_grid` reserves the in0-mux row) rather than 12x9, and ff2's plain matmul runs on
the full 8x9. Neither grid is one Blackhole produces, so every H3 blocking the model carried for
these shapes had been swept on a grid that does not exist on the part.

#### Matmul blockings — 15 s, M=13664

All four 15 s shapes swept with `sweep_mm_block_sizes.py` on `wh_4x8_ring` at the pipeline's 13664
rows/device: 1419 combos, every one `OK`. "Pre-tuning" is what the model ran before any Wormhole
entry existed -- the (K, N)-keyed `AGMM_BLOCK_SIZES` defaults for the three AGMM shapes, and the
hardcoded (8, 8, 8) for ff2's plain matmul.

| shape | op / grid | combos | pre-tuning | shipped now | rank | best measured | saved vs pre-tuning |
|---|---|---|---|---|---|---|---|
| [ff1](ff1.md) | AGMM 8x8 | 320 | (8, 3, 14) 16718.0 us | **(8, 7, 10)** 15709.9 us | 1 | = shipped | **6.0%** |
| [ff2](ff2.md) | matmul 8x9 | 322 | (8, 8, 8) 7013.1 us | **(8, 7, 10)** 6770.7 us | 2 | (12, 7, 8) 6668.2 us | **3.5%** (4.9% at best) |
| [qkv](to_qkv.md) | AGMM 8x8 | 425 | (8, 7, 12) 10401.8 us | (8, 7, 12) *unchanged* | 2 | (8, 6, 12) 10351.4 us | 0 (0.5% at best) |
| [to_out](to_out.md) | AGMM 8x8 | 352 | (8, 8, 6) 4332.8 us | (8, 8, 6) *unchanged* | 3 | (14, 8, 6) 4312.8 us | 0 (0.5% at best) |
| **total, as shipped** | | | | | | | **1250.5 us/block = 62.5 ms/step = 0.50% of the 12435 ms forward** |

ff1's landed `(8, 7, 10)` is the sweep winner outright. ff2's landed `(8, 7, 10)` is rank 2:
`(12, 7, 8)` measures 1.5% faster (102.5 us/block, ~5 ms/step) but the sweep is timing-only and
that blocking has not been PCC-validated, so it is **not landed** -- `(8, 7, 10)` was validated at
pcc 1.0000000 and stays. qkv and to_out keep their `AGMM_BLOCK_SIZES` defaults: both are within
0.5% of the best combo, comparable to the ~0.3% run-to-run spread, so neither is worth an entry.

Caveat found 2026-09-21: the to_out rows were swept with the harness's `"plain"` use case, without the fused addcmul
epilogue the model runs and with `math_approx_mode=False`; the real op is 5.3 ms, not 4.33. See [to_out.md](to_out.md).

Why the two landed entries are keyed on `(M, K, N)` rather than added to the `(K, N)`-keyed model
table, and how `get_matmul_config` falls back on equal `M_per_core` after an exact miss, is
documented in `models/tt_dit/utils/matmul.py` -- the module docstring and the `grid_88_configs` /
`grid_89_configs` entries.

The fused MM/RS study (the 2026-09-17 grid sweep was not like-for-like; re-done at M=13664 and landed 2026-09-23) is in
[ff2.md](ff2.md); the SDPA chunk-size
sweeps and the L1 envelope are in [sdpa.md](sdpa.md).

### adaLN table gathers — a DRAM-bank hotspot, not bandwidth (2026-09-24)

Every block gathers its six modulation tensors with `ttnn.embedding` (`transformer_block_minimax_h3.py:265`):
13664 rows per device from a table of `num_timesteps x 3` rows, six times, 221 MB of output. The all-on profile
(`2026_09_24_18_33_49`) put that at **10.45 ms per block, 7% of DRAM bandwidth**, while a plain read+write of the same
bytes (`ttnn.add` on the output shape) takes 0.36 ms. The op is not slow in general: its tilized reader
(`embeddings_tilize.cpp`) issues one row read per packed row, and an interleaved table keeps each 2.7 KB row in one
bank, so with three distinct rows all 72 cores queue on the same bank. Single-chip microbenchmark of the shape, per
gather: 1 copy 1.52 ms; 4 copies 0.56; 8 copies 0.41; **12+ copies 0.35 ms**, the plain read+write figure.

Landed in `2b9b8f8dce9`: `_modulation_tables` interleaves `ADALN_TABLE_COPIES = 16` copies of every row once per block on the
tiny joint table (`transformer_block_minimax_h3.py:37`, `:233`; 0.02 ms) and keeps the tables ROW_MAJOR; `forward`
maps each index to `row * 16 + position % 16` (`spread_adaln_indices`, `:251`), with the `position % 16` vector built once
per padded length outside the traced block loop (`transformer_minimax_h3.py:408`). The index contract at the block
boundary (`t * MODALITY_NUM + modality`) is unchanged.

Measured on this host, 15 s / 16:9, `fsdp1`, both switches on (report `2026_09_24_19_10_41`, figures in
`transformer_roofline_out/all_on_adaln_spread/`):

| | before | after |
|---|---|---|
| Embeddings (adaLN tables), 6 calls | 10.45 ms (7% DRAM) | **2.39 ms** (32% DRAM) |
| block, device only | 238.5 ms | **230.8 ms** (-3.2%) |
| denoise step, 49 steps | 12.26 s (10:23) | **11.93 s** (9:44) |
| block PCC vs torch (`test_minimax_h3_transformer_block`) | 99.9995% | 99.9995% |

The remaining 0.40 ms per gather is the reader's 32-row read + barrier structure, not the bank; the two final-norm
gathers in `transformer_minimax_h3.py` run once per forward and were left alone.

### Perf experiments — index

One row per experiment, in the order they were numbered (12 was never allocated); the detail lives in the per-op docs.

| # | experiment | status | result |
| 1 | Matmul blockings, all 4 shapes x 3 durations, 8x8/8x9 grids | **done** | 3.5% of matmul time at 15 s; ff1 and ff2 landed, 0.5% of a forward. Table in Part 4 *Matmul blockings*; per op in [ff1.md](ff1.md), [ff2.md](ff2.md), [to_qkv.md](to_qkv.md), [to_out.md](to_out.md) |
| 2 | Fused MM/RS at 8x5/8x6/8x7 matmul grids | **superseded** | The 2026-09-17 verdict ("all worse") compared a 5 s fused total against the matmul alone. Re-done like-for-like at M=13664 (2026-09-22/23, experiment 2b): fused 8x7 with a swept blocking **8.94 ms vs 10.70 unfused** (MM + RS + addcmul), pcc 0.99993, 2000-call soak clean, 10-step A/B **-166 ms/fwd** at equal CLIP; **landed** as an 8x9 `fused_mmrs_configs` entry. Unfused RS hyperparameters (4 workers/dir, chunks 16-32) are worth -5% on the unfused path, not landed. [ff2.md](ff2.md) |
| 3 | SDPA chunk sizes, q in {256,384,512} x k in {256,512} | **done** | Shipped `(256, 512)` already optimal; larger q L1-infeasible. [sdpa.md](sdpa.md) |
| 3b | `q_chunk=128` | **done** | Not a perf path: slower than q=192 at every feasible k. Hang history in [sdpa.md](sdpa.md), chunk-shape zone. |
| 4 | SDPA chunk sizes, small-q / large-k (q<=256, k>=512) | **done** | Hypothesis disproved. `(192, 640)` is feasible — the first k>512 point on this shape — but 13% slower than the shipped `(256, 512)`; larger q is more per-core efficient and shrinking q raises iters/core. Chunk tuning at 15 s is exhausted. L1 envelope calibrated as a by-product. [sdpa.md](sdpa.md) |
| 5 | Re-profile the block with landed configs | **done** | 2026-09-24, this host, every optimization on (tree default + fp32 dest off + LUT SwiGLU): **246.92 -> 238.27 ms** device-only (-3.5%), 244.05 -> 235.34 device-busy union (-3.6%), ff1 at 66% of HiFi2 peak, block headroom 2.26x -> 2.18x. Table and per-op roofline in Part 2, *Roofline with every optimization on* |
| 6 | Pipeline re-run: warm total, denoise, ms/fwd, CLIP | **done** | Same-host A/B: **-69.9 ms/fwd, -0.58%**, exactly the isolated-sweep prediction. CLIP **35.88** (min 34.69, bar 33.0) on the later run |
| 8 | FSDP layout conversions | not started | **TODO** — tilize/untilize go 0.13 -> 3.13 ms under FSDP, a 23x blowup and a quarter of the whole FSDP cost spent on format round-trips rather than communication. Cheapest apparent win in the breakdown |
| 9 | Ring SDPA kernel utilization | not started | **TODO** — 48% at the shipped chunk size, inherent to the kernel at this shape rather than a chunk-size miss. Note `PM FPU UTIL (%)` is the perf-model ideal divided by measured time (`tools/tracy/process_ops_logs.py`), not a hardware counter; the work is in `compute_streaming.hpp`. Inner-loop zones and experiments in [sdpa.md](sdpa.md) |
| 10 | `dit_fsdp: True` in `_PRESETS_WH` | not started | **TODO** — decision, not a measurement; costs 5.7% of the block, buys the headroom a 12 GB part needs |
| 11 | Numerics of the landed blockings (the sweep never checked) | **done** | ff2 (8,7,10) pcc 1.0000000 vs torch, identical to (8,8,8) to one bf16 ulp; ff1 (8,7,10) pcc 0.9999843 on the real SwiGLU ring, = (8,3,14) to 6 dp. Both PASS |
| 13 | TP/SP axes and factors at 15 s / 16:9 (`test_parallel_sweep_minimax_h3.py`) | **done** | Only three configurations exist on this mesh and the shipped TP4/SP8 is the fastest: TP8/SP4 is **+4.1%** ms/fwd (untuned blockings), TP1/SP32 **hangs deterministically** in its first forward. See *TP/SP parallel-configuration sweep* below |
| 14 | ff1 AGMM utilization (51% of HiFi2 peak) | **measured** | Roofline + six on-device experiments. The 15.7 ms splits into 8.0 ms of FPU work, **2.8 ms of serialized SwiGLU epilogue** and 4.9 ms of K-loop overhead (first read as compute/delivery co-limiters; row 16 corrected this to pipeline issue efficiency). fp32 dest off measures -4% alone, -8% with 8-tile subblocks, at 2x the numerical error; K_block >= 14 gives nothing. [ff1.md](ff1.md) |
| 15 | ff1 AGMM SwiGLU epilogue: bf16-grade `silu_tile<false>` (2026-09-19) | **landed** | Device kernel 15,852 → **15,289 us (-3.6%)** on the mesh, PCC 0.99998 unchanged to the 4th decimal; the mesh "hang" it was first blamed for was a semaphore-reuse race in `sweep_mm_block_sizes.py` (one semaphore pair for back-to-back calls; the model ping-pongs two), fixed. [ff1.md](ff1.md) |
| 16 | ff1 AGMM K-loop attribution with per-thread device zones (2026-09-19) | **measured** | The 5 ms above the FPU time is **pipeline issue efficiency**, not delivery: on a 2x2 fp32 subblock the MATH thread issues at 47 cycles per tile-MAC (nominal 32), UNPACK is busy 42 per tile and PACK 309 per fp32 L1-acc tile, with 21-26 cycles of DST waits — all three ~95% busy. A relay-protocol prefetch changed nothing (16.08 → 16.06 ms); 4x1 subblocks cost +2.2 ms; doubling K_block -1.3%. Remaining levers are precision decisions (fp32 dest off + 2x4: -8% measured; LUT sigmoid) or tt-llk work on `matmul_block`. [ff1.md](ff1.md) |
| 17 | to_qkv AGMM attribution (2026-09-21) | **measured** | Behaves like ff1: K loop 10,053 of 10,228 us per core (98%), operand waits 0.9 us of a 30 us iteration, sampled pipeline at the same 47 cycles per tile-MAC; the `chunks=3` writer split costs 0.08 ms and the copy epilogue 0.17. fp32 dest off with a 4x2 subblock: **11.30 → 10.32 ms on the mesh (-8.7%)**, rel-RMSE 0.0044 → 0.0107 (bar 0.02). LoFi saves only 1.0 of 3.0 ms of math (unpacker-paced). [to_qkv.md](to_qkv.md) |
| 20 | fp32 dest off for ff1 / to_qkv, end to end (2026-09-24) | **measured** | Two SwiGLU epilogue fixes first (live-pairs-only, truncating multiply): ff1 fp32 off (8,7,16) 2x4 on the mesh bench 15.63 -> **14.30 ms** (-8.5%). Exploration switch `MINIMAX_H3_MM_FP32_DEST` in the model. Block, as device-busy wall time per layer (`tools/block_device_busy.py`, the union of op intervals; per-op sums double count the FSDP gathers that overlap to_out): **ff1 alone -0.91 ms**, to_qkv alone -0.35, both -1.22, additive. to_out's longer *kernel duration* with to_qkv off is the concurrent weight gather being re-apportioned, not a cost. Pipeline 10-step: time unresolvable (0.35% of a forward), output a different sample of the same prompt (mean abs diff 33 of 255 with both ops off vs 1.2 run-to-run). **50-step, same session, back to back: ff1-only CLIP 35.91 (min 34.96) vs production 35.75 (min 34.83); per forward 12087 vs 12143 ms (-56 ms, -0.46%)**; output shift 16 of 255 (the TP8 range). Verdict in [ff1.md](ff1.md) §3.4: -0.84 ms per layer at production-level CLIP, adoption is the user's call; switch defaults to fp32 on |
| 19 | AGMM K loop: what paces the 2x2 fp32 subblock (2026-09-23) | **answered** | Source accounting of the 47 cycles per tile-MAC (per-tile `SETC16` + MOP + bank switch on MATH; 4 unpacks per 4 tile-MACs with a context round-trip on UNPACK) and a three-signature decision table in [ff1.md](ff1.md) §3.1; the K1-K5 ladder in ff1.md §5. Capture attempt on the mesh bench found two traps: the tracy parent deadlocks on the chip lock without `TT_METAL_DEVICE_ARCH=wormhole_b0`, and the ring op hangs in its first call with `-DPROFILE_PERF_COUNTERS` (board reset needed). Answered the same day by an engine-isolation study on GWH01 (ff1.md §3.1, exp 16): the bare unpack stream is 185 of the 208 cycles per K-tile step and HiFi4 lands at 256 + 24, so the loop is **unpacker-paced** (4 x 2 KB per 4 tile-MACs at ~44 B/cycle); math-thread issue work (K2-K4) is closed, the levers are bytes per tile-MAC: fp32 dest off with 2x4 (measured) and bfp8 in1 / in0 (projected -25% / -35%, precision decisions). **Verified on this galaxy the same evening** (ff1.md exp 17): all eleven variants within 1-2%, and the hardware counters show FPU 60%, math thread never stalled, unpacker requests half-blocked by overwrite protection and never by the L1 port: the floor is the src-register handshake (4 per step) plus the exposed srcB refill of the 2x2 scheme, so one tt-llk lever remained before the precision levers: hide the refill by alternating the MVMUL order between K tiles. **Built and landed the same evening** as `matmul_block_kloop` (ff1.md exp 18): -3.5% of the K loop at HiFi2 (209.7 -> 202.3 cycles per step), bit-exact; mesh ff1 16.04 -> 15.69 ms, to_qkv 11.25 -> 11.03, ff2 fused 8.98 -> 8.68. Half the projection: the isolation ladder re-run before/after puts the exposed refill at 5 cycles (nopack 197.9 -> 192.8, i.e. on the 193 mock floor; both mock rows unchanged) and the remaining 202 vs 193 at the packer's interaction with the loop (the per-subblock DST handoff; L1-port refusals stay 0%), which no MVMUL order touches; the lever is exhausted at 2x2, to be carried into 2x4 / 4x2 with fp32 dest off. Plan, trace and result in [kloop_refill_reorder_handoff.md](kloop_refill_reorder_handoff.md) |
| 18 | to_out AGMM attribution (2026-09-21) | **measured** | The op the model runs (fused addcmul, approx on) is **5.29-5.31 ms** on the device, not the 4.33 ms the blocking sweep recorded with the `plain` use case: the addcmul epilogue is 0.7 ms (two passes over the fp32 intermediate) and the K loop **waits on the in0/in1 relay 5.7 us of every 23 us iteration (~1.1 ms)** -- to_out needs ~12.8 GB/s per core of operands at its MAC pace and the store-and-forward relay delivers ~10. Relay prefetch: no gain (5.52 vs 5.43); fp32 dest off: -3.5% only, larger subblocks / M_block 16 / K_block 14 nothing on the mesh (they help single-device, where the loop does not wait). Levers left: a one-pass epilogue (~-0.35 ms) and a higher-bandwidth in0 path (multicast); [to_out.md](to_out.md) |
| 21 | adaLN table gathers: interleave 16 row copies, spread indices by position (2026-09-24) | **landed** | `2b9b8f8dce9`. Embeddings 10.45 -> **2.39 ms** per block, block 238.5 -> **230.8 ms** (-3.2%), step 12.26 -> 11.93 s; PCC unchanged. Part 4 *adaLN table gathers*. |

### TP/SP parallel-configuration sweep — 15 s / 16:9

Measured 2026-09-17 on this host at `bc1d99d05f6` plus the `matmul.py` change listed at the end
(landed on top of `2a47fd04fc6`), with `models/tt_dit/tests/models/minimax_h3/test_parallel_sweep_minimax_h3.py` (new). Driver,
logs, `results.jsonl`, the strided frame dumps and `compare.py` are in `~/h3_parallel_sweep/` on the
run host. `MINIMAX_H3_DIT_FSDP=1`, Ring, 4 links, seed 0, the fox prompt, **10 scheduler steps**
(9 forwards): ms/forward is flat across steps, so 10 steps gives the ranking at a fifth of the wall
clock. The 10-step ms/fwd runs ~3% above the 50-step figure (fixed per-request work amortised over
9 forwards instead of 49); every row here is at 10 steps, so the comparison is like for like.

#### What can be configured at all

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

#### Results

| config | TP/SP | enc | denoise | vae | audio | total | ms/fwd | vs shipped | video PCC vs shipped | audio PCC |
|---|---|---|---|---|---|---|---|---|---|---|
| `4x8_tp0_sp1` (shipped) | 4/8 | 3.5 | 111.4 | 14.4 | 4.1 | 133.4 | **12382** | -- | 1.0 | 1.0 |
| `4x8_tp1_sp0` | 8/4 | 4.1 | 116.0 | 14.4 | 3.4 | 137.9 | 12890 | **+4.1%** | 0.907 | 0.972 |
| `1x32_tp0_sp1` | 1/32 | -- | **hang** | -- | -- | -- | -- | -- | -- | -- |

**The shipped TP4/SP8 stays.** Neither alternative is a speedup, and one does not run.

**TP8/SP4 (+4.1%)** runs on the generic matmul blockings: none of its four AGMM shapes
(`(5376, 2688)` qkv, `(7168, 672)` to_out, `(5376, 3584)` ff1 at M=27296, plus the refiner's) has a
swept entry. The shipped config's tuned entries are worth 0.58% (experiment 6, *Optimization target* above), so even a generous
allowance for tuning TP8's shapes leaves it ~3.5% behind, and the halved SP ring buys nothing the
doubled TP ring does not cost: per layer each device now receives ~257 MB per AGMM all-gather (was
110 MB) against ~294 MB of KV around the ring (was 685 MB). Its output is the same video -- same
fox, scene and motion, small pose/detail drift (mean |diff| 12-19 of 255 per frame, growing with
frame index) -- which is what a changed bf16 reduction order looks like after 9 sampling steps;
`compare_tp4_vs_tp8.png` in the results dir shows four frame pairs. Not a correctness problem.

**TP1/SP32 hangs, deterministically.** Two attempts, the second on a freshly reset board with
kernels coming from the cache (zero `BuildKernels` lines), both stall in the *first* forward:
the last log line is the generic-blocking warning for `proj_in` `(107872, 96, 5376)`, then nothing.
Fingerprint identical to the mid-denoise hang (see "If a hang recurs" below): 180-365% CPU with
CPU-time far past
elapsed, all ~448 threads in `futex_wait_queue`, `pytest --timeout` does not fire, board needs
`tt-smi -r all` afterwards (which warns that Galaxy CPLD FW < 1.16 should use `-glx_reset`, but did
work here). Not root-caused; TP=1 is also the configuration with the least to gain (each device
holds all 56 heads, so the KV ring moves 4x the bytes of TP4, and FSDP gathers full 5376-wide
weights over a 32-ring), so it was not pursued further. Evidence in `1x32_tp0_sp1_s10.HANG.txt`.

#### Code changes this needed

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

### Open issues

Fixed items have been removed; their forensics live in the commits. The intermittent mid-denoise
hang was root-caused to Wormhole taking the fused MM/RS fallback by accident (one reduce-scatter
worker per link on the 8x9 grid, 50 times per step) and closed by gating that path on a blocking that
resolves for the real core grid (`eab3dfbd599`): 2744 denoise steps and a full 18/18 sweep since, no
recurrence, and 4.7% faster. The VBench setup gaps moved to `../MiniMaxH3.md`.

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
   `test_transformer_minimax_h3.py::_packed_sizes` counts audio in rows (two per latent,
   `packing.py:261`) and uses the gate's 39-token prompt, and both the length and the padding go
   through `packing.py` helpers (`packed_sequence_length`, `padded_sequence_length`) that the pipeline
   itself uses, so the two cannot diverge again. `get_matmul_config`/`get_agmm_config` fall back to a
   same-(K, N) entry with the same `M_per_core` after an exact miss, which makes every table robust
   to a 32-row discrepancy and to prompt length within a padding bucket. Confirmed on a live
   generation: `13664 rows/device`, and neither ff1 nor ff2 appears among the fallback warnings (qkv
   and to_out do, by design -- their shipped blockings measured optimal / within noise). Root cause:
   the block-perf harness counted audio latents once where the pipeline packs two rows per latent,
   and assumed a 512-token prompt, so every table was keyed on 4768 / 9216 / 13632 -- lengths the
   pipeline never produces (13632 is unreachable at 15 s at any prompt length).

### If a hang recurs

Symptom fingerprint of the two hangs seen here (the closed mid-denoise one and the open TP1/SP32 one):

- No further log output, indefinitely. No exception, no traceback.
- Process **spinning, not idle**: 170-370% CPU, CPU-time climbing past elapsed, every thread in
  `futex_wait_queue` -- host dispatch busy-polling a device that stopped retiring work:
  ```bash
  P=$(pgrep -f "^python -m pytest models/tt_dit" | head -1)
  ps -o pid,stat,etime,time,wchan:24,pcpu -p $P
  for t in /proc/$P/task/*; do echo "$(basename $t) $(cat $t/wchan)"; done | sort | uniq -c
  ```
- `@pytest.mark.timeout` does not fire (pytest-timeout cannot interrupt a C-level stall), and the
  board is wedged afterwards (`failed to initialize FW`, `Timed out waiting for ETH heartbeat`). A
  graceful `kill -TERM` exits cleanly but does not un-wedge it.

Run so that a stall becomes a raised timeout with a device-state dump, instead of destroying the
evidence with a shell `timeout` and a reset:

```bash
export TT_METAL_HOME=/home/jameslee/tt-metal
export TT_METAL_INSPECTOR=1
export TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT=1
export TT_METAL_OPERATION_TIMEOUT_SECONDS=300     # >> the 12.7 s worst-case step
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="$TT_METAL_HOME/tools/tt-triage.py --disable-progress --triage-summary-path=$OUT/triage_summary.txt --sqlite-output-path=$OUT/triage.sqlite"
```

The triage scripts that matter: `dump_op_mesh` (op-ID skew across the mesh shows which chip
stopped), `dump_callstacks`, `check_eth_status`, `check_noc_status`. Watcher
(`TT_METAL_WATCHER=30 TT_METAL_WATCHER_APPEND=1`) adds per-RISC waypoints but perturbs timing; hold
it in reserve. Keep any SIGBUS log: a fault in the hugepage readback is what a device dropping off
PCIe looks like, and `dmesg -T` would show the AER event.

Recovery, after confirming nobody else is on the box:

```bash
for p in $(ls /proc | grep -E '^[0-9]+$'); do
  ls -l /proc/$p/fd 2>/dev/null | grep -q tenstorrent && \
    echo "pid $p user=$(stat -c %U /proc/$p) cmd=$(tr '\0' ' ' </proc/$p/cmdline | cut -c1-60)"
done
tt-smi -r      # supersedes the deprecated -glx_reset
python -c "import ttnn; d=ttnn.open_mesh_device(ttnn.MeshShape(1,1)); print('OK'); ttnn.close_mesh_device(d)"
```

### VBench (16:9/5s, verified passing)

| dimension | score | bar |
|---|---|---|
| subject_consistency | 0.9793 | 0.95 |
| background_consistency | 0.9779 | 0.95 |
| motion_smoothness | 0.9915 | 0.97 |
| dynamic_degree | 1.0000 | 1.0 |
| imaging_quality | 0.6802 | 0.64 |

CLIP 37.36 vs 33.0 bar (docs record 37.37 for Blackhole; imaging_quality 0.6896).
Only 16:9/5s has been VBench-verified; the sweep ran with `RUN_VBENCH=0`.
