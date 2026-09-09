# MiniMax-H3 audio decoder: where the time goes, the hardware floor, and the path down (2026-09-09)

Scope: `MiniMaxH3AudioDecoder` on `main` (`204e681e6d6`), 4x8 Blackhole Galaxy (c02u08), T-shard factor 8 on mesh
axis 1 (main's default), fp32, shipping decoder settings. Branch with the measurement tools:
`rouzbeh/audio-decoder-benchmark`. Clips: 207 latents (5.17 s) and 600 latents (15 s); batch 1 = one request.

## 1. What the decoder computes (per chip, at factor 8)

BigVGAN-v2 AMP1 vocoder: `dec_in_proj` (32 -> 2048, k1) then `conv_pre` (2048 -> 1024, k7) on the full T, then 7
upsample bands (x5, x5, x2, x2, x2, x2, x2; channels 1024 -> 512 -> ... -> 8), each = one transposed conv + 3 AMP
blocks x 3 branches x (anti-aliased SnakeBeta, dilated conv, anti-aliased SnakeBeta, conv, residual add), then
`act_post`, `conv_post` (8 -> 1, k7), clamp, T all-gather. An anti-aliased activation is `UpSample1d(2x, 12 taps)
-> snake_beta -> DownSample1d(2x, 12 taps)`; there are 18 of them and 18 convs per band, 126 + 1 per decode.

Roofline (`audio_decode_roofline.py`), per chip, 600 latents, batch 1, factor 8 (60000 waveform rows per chip):

| stage | GFLOP | DRAM MB as implemented | DRAM MB fused ideal | t at HiFi4 | t mem as impl. | t mem ideal |
|---|---|---|---|---|---|---|
| conv_pre (full T, replicated) | 17.6 | 66 | 66 | 0.10 ms | 0.13 ms | 0.13 ms |
| band 0 (1024->512, x5) | 28.6 | 582 | 195 | 0.16 | 1.14 | 0.38 |
| band 1 (512->256, x5) | 36.2 | 1114 | 148 | 0.21 | 2.18 | 0.29 |
| band 2 (256->128, x2) | 17.2 | 1101 | 120 | 0.10 | 2.15 | 0.23 |
| band 3 (128->64, x2) | 9.0 | 1095 | 114 | 0.05 | 2.14 | 0.22 |
| band 4 (64->32, x2) | 4.9 | 1093 | 112 | 0.03 | 2.13 | 0.22 |
| band 5 (32->16, x2) | 2.8 | 1093 | 112 | 0.02 | 2.13 | 0.22 |
| band 6 (16->8, x2) | 1.8 | 1093 | 111 | 0.01 | 2.13 | 0.22 |
| act_post + conv_post + clamp | 0.05 | 69 | 4 | 0.00 | 0.14 | 0.01 |
| **total** | **118** | **7305** | **982** | **0.67 ms** | **14.3 ms** | **1.9 ms** |

Assumptions: 130 Tensix at 1.35 GHz, 1.35 TFLOPS/core at HiFi4 (5.4 at LoFi) from `tech_reports/GEMM_FLOPS`;
DRAM 512 GB/s is the GDDR6 datasheet figure (not measured on Blackhole in this repo; Wormhole's report reaches
~93 % of its spec). "As implemented" counts every op's read + write in fp32, including the 2x intermediate of each
activation, the tilize/untilize hops around `snake_beta`, halo copies and the T-pad tail multiplies; "fused ideal"
reads and writes each band's tensor once per fused branch pass.

**So the arithmetic is worth well under 1 ms per chip and even the as-implemented DRAM traffic only ~14 ms.** The
15 s decode measures on the order of a second. The decoder is not compute- or bandwidth-bound anywhere; the cost is
the number and shape of its ops (~1700-2000 per decode) and what each one costs to launch and run at low occupancy.
Every band below band 2 moves the same ~1.1 GB because the row count doubles as the channel count halves: from
band 2 on, the work per op is constant and tiny (60000 x 8 fp32 = 1.9 MB tensors in band 6).

## 2. Where the time goes (measured)

### 2a. Device time per op, one eager decode (600 latents, batch 1, factor 8; profiler C++ report, chip 0)

Source: `tools/tracy_audio_decode_t8_harness.py` under `python -m tracy -p`, summarized by `summarize_cpp_perf.py`
(`adb_cpp_perf_600lat_b1_dev0.csv`). All 32 chips agree within 4 % (580-601 ms).

**One decode = ~5,400 device ops per chip, 581 ms of device firmware time (492 ms inside kernels), mean 105 µs and
median 23 µs per op. Launch/teardown inside the FW window is 16 µs/op (89 ms).**

| op type | ops | device ms | mean µs | share |
|---|---|---|---|---|
| Conv3d (the 1-D convs, as 3 bf16-split conv3d each) | 408 | 185.7 | 455 | 32 % |
| BinaryNg (adds, multiplies, the split's hi/lo arithmetic) | 492 | 134.6 | 274 | 23 % |
| UntilizeWithUnpadding (TILE -> ROW_MAJOR after snake and after each depthwise filter) | 678 | 45.5 | 67 | 8 % |
| NeighborPadAsync (T halo exchange, convs and resamplers) | 381 | 39.2 | 103 | 7 % |
| Ternary = `snake_beta` | 127 | 31.2 | 245 | 5 % |
| Concat (resampler phase interleave, halo assembly) | 222 | 30.4 | 137 | 5 % |
| Typecast (fp32 -> bf16 -> fp32 operand split) | 272 | 22.6 | 83 | 4 % |
| Conv2d (12-tap depthwise resampler filters) | 489 | 20.7 | 42 | 4 % |
| InterleavedToSharded + ShardedToInterleaved + Halo + Move (depthwise filter plumbing) | 1956 | 27.0 | 14 | 5 % |
| TilizeWithValPadding (ROW_MAJOR -> TILE before snake) | 142 | 14.0 | 99 | 2 % |
| Slice, Permute, Unary, MeshPartition, Untilize | 360 | 24.2 | 67 | 4 % |
| AllBroadcast (7 composite band gathers) + AllGatherAsync (final) | 8 | 6.0 | 750 | 1 % |

Per band (segment k = band k-1's AMP blocks + band k's transposed conv; band 6's segment also holds act_post,
conv_post and the next pass's dec_in_proj + conv_pre, ~32 ms):

| segment | ops | device ms | mean µs/op | note |
|---|---|---|---|---|
| conv_pre + band 0 transposed conv | 25 | 32 | 1285 | 2048->1024->512 ch, big convs |
| band 0 AMP (512 ch) + band 1 ups | 1473 | 57 | 39 | 512 ch: depthwise filters chunked over C, 2.3x the ops |
| band 1 AMP (256 ch) | 645 | 47 | 73 | |
| band 2 AMP (128 ch) | 645 | 27 | 42 | |
| band 3 AMP (64 ch) | 645 | 31 | 49 | |
| band 4 AMP (32 ch) | 646 | 51 | 79 | |
| band 5 AMP (16 ch) | 685 | 114 | 166 | |
| band 6 AMP (8 ch) + post + next pre | 771 | 222 | 288 | ~190 ms is band 6 |

**The last two bands (16 and 8 channels, 30k and 60k rows per chip) take ~300 of the 581 ms**, and their per-op
cost rises as the channel count falls: in band 6 a single add on a 60000x8 fp32 tensor (1.9 MB) takes 490-610 µs, a
`snake_beta` 760-980 µs, an untilize 150-840 µs, a halo exchange 200-700 µs. That is 2-10 GB/s per op against a
512 GB/s part: with 8 fp32 channels a ROW_MAJOR row is a 32-byte page, and in TILE layout the 8 channels are padded
to a 32-wide tile (4x the work). One activation->conv unit in band 6 is **39 ops and ~8.7 ms**: snake (1 op), untilize,
downsample (halo, shard, filter, unshard, untilize = 6 ops), phase/permute plumbing (6), conv (halo, 2 typecasts,
3 conv3d, 3 binaries, slice = 10), upsample of the next activation (2 x (halo, shard, filter, unshard, untilize) +
concat + tilize = 13).

### 2b. Wall time, eager vs traced (`test_audio_decode_wall_matrix`, best of 3 after a warm call)

| factor | clip | batch | eager | traced | eager - traced | vs unsharded traced |
|---|---|---|---|---|---|---|
| 8 | 5.2 s (207 latents) | 1 | 0.684 s | **0.274 s** | 0.41 s | 3.6x |
| 8 | 5.2 s | 2 | 0.802 s | 0.526 s | 0.28 s | 3.8x |
| 8 | 15 s (600 latents) | 1 | 0.756 s | **0.526 s** | 0.23 s | 5.6x |
| 8 | 15 s | 2 | 1.101 s | 1.018 s | 0.08 s | 5.8x |
| 1 | 5.2 s | 1 | 1.167 s | 0.997 s | 0.17 s | |
| 1 | 5.2 s | 2 | 2.075 s | 2.010 s | 0.07 s | |
| 1 | 15 s | 1 | 2.982 s | 2.938 s | 0.04 s | |
| 1 | 15 s | 2 | 5.902 s | 5.902 s | 0.00 s | |

Reading:
- **Traced factor-8 decode = device time.** 0.526 s for the 15 s clip against ~0.5-0.58 s of summed per-op device
  time in 2a (the profiled window carries ~135 one-time setup ops and profiler overhead, so it reads slightly high).
  Trace leaves nothing on the host side to recover; every further gain has to come from the op graph itself.
- **Eager costs 0.23 s (15 s) to 0.41 s (5 s) on top of that** = the host dispatch gaps. The short clip pays more
  because tile-floor padding (207 latents -> 32 rows/shard -> 49 pad rows) adds the 247 T-pad tail-maintenance
  calls (~700 ops); at 600 latents (75 rows/shard, no padding) there is none. Per op the exposed gap is ~40-70 µs; at
  batch 2 / 15 s the ops are long enough to hide dispatch entirely (eager = traced). The profiler's own op-to-op
  latency column (median 105 µs) overstates this: the profiler's mid-run dumps add to it. Section 3 uses the
  measured 0.23 s.
- **Batch is real work**: batch 2 doubles traced time (0.526 -> 1.018 s at 15 s); batch 1 is the deployment case.
- **Clip length is sub-linear**: 207 -> 600 latents (2.9x rows) costs 1.9x (0.274 -> 0.526 s), i.e. ~0.15 s of the
  15 s decode is per-op fixed cost that does not scale with rows.
- Unsharded, the decode is device-bound already (eager = traced) and 3.6-5.8x slower than factor 8.

### 2c. Eager stage split with mesh syncs (`test_audio_decode_stage_timing`; shares of the synced eager wall)

15 s clip, batch 1, factor 8 (synced wall 1.244 s; the syncs themselves add ~0.5 s over the 0.756 s unsynced eager):

| stage | ms | share |
|---|---|---|
| latent projection (`dec_in_proj`: upload + conv + readback) | 21 | 1.7 % |
| vocoder upload | 7 | 0.5 % |
| conv_pre | 16 | 1.3 % |
| band 0 AMP blocks (512 ch) / transposed conv | 168 / 23 | 13.5 / 1.9 % |
| band 1 AMP (256 ch) / ups | 113 / 15 | 9.0 / 1.2 % |
| band 2 AMP (128 ch) / ups | 122 / 7 | 9.8 / 0.5 % |
| band 3 AMP (64 ch) / ups | 113 / 8 | 9.1 / 0.6 % |
| band 4 AMP (32 ch) / ups | 115 / 9 | 9.3 / 0.7 % |
| band 5 AMP (16 ch) / ups | 145 / 16 | 11.6 / 1.3 % |
| band 6 AMP (8 ch) / ups | 251 / 33 | 20.1 / 2.6 % |
| act_post + conv_post | 14 | 1.1 % |
| readback + host crop | 31 | 2.4 % |
| of which: halo exchanges, convs (128 calls) + resamplers (254 calls) | 54 + 103 | 12.6 % |
| of which: final T all-gather | 3 | 0.2 % |

5.2 s clip, batch 2 (synced wall 1.513 s): same shape, plus **T-pad tail maintenance 206 ms (13.6 %, 247 calls)**
and halo exchanges 61 + 119 ms (12 %).

In eager terms band 0 costs as much as band 5 despite 3x less device time (57 vs 51 ms in 2a) because its depthwise
filters are chunked over the 512 channels (1,473 ops vs ~645 per band) -- op count, not bytes. The 382 halo
exchanges are ~13 % of eager time (39 ms of device time in 2a: mostly launch and the 1-2 row transfers).

## 3. How fast it could go

Four floors, per 15 s clip at factor 8, from the hardware numbers in section 1 and the profile in section 2:

| floor | time | what it assumes |
|---|---|---|
| arithmetic | 0.7 ms | 118 GFLOP/chip at HiFi4 peak; 0.2 ms at LoFi |
| memory, fused ideal | ~2 ms | each band's tensor read and written once per fused branch pass, 512 GB/s |
| memory, as implemented | ~14 ms | every current op reads and writes its fp32 operands at full bandwidth |
| launch floor at the current op count | 0.1-0.2 s | ~5,400 ops x (16 µs FW launch/teardown measured + ~5-20 µs replay gap) |
| current device work = traced today | 0.53 s | a perfectly dispatched (traced) run of today's op graph, measured |
| eager today (what the pipeline runs) | 0.76 s | device work + 0.23 s host dispatch idle (0.41 s on a 5 s clip) |

Reading: the hardware could do this decode in a few milliseconds; today's op graph needs ~0.5 s of device time
because its ~5,400 ops each move a small ROW_MAJOR fp32 tensor at 2-10 GB/s and pay ~16 µs of launch each, and eager
dispatch adds another 30-60 %. So the ceiling is set by op count and per-op efficiency, in that order: (1) remove the
host gaps (trace), (2) cut the op count per activation->conv unit (39 today), (3) make each remaining op move its
bytes at a sane fraction of bandwidth (wide pages, no tile padding of 8 channels, no untilize/tilize hops).
A realistic target with today's kernels is the launch floor plus fused memory time: **~0.15-0.25 s per 15 s clip**,
i.e. 2-3.5x below the traced number and 3-5x below eager; below that needs new fused kernels.

## 4. Steps toward the floor

Ordered by payoff per unit of work, with the device-profile numbers they rest on. Payoffs are per 15 s clip at
factor 8 on 4x8; "device ms" are from section 2a and add up only if the op graph is otherwise unchanged.

1. **Run the pipeline's audio decode traced (host-side change, no kernel work).** `_decode_audio` calls
   `audio_decoder(latents)` eagerly on main; the decoder already supports `traced=True` (needs a trace region on the
   mesh). Removes the host dispatch idle: 0.76 -> 0.53 s on the 15 s clip (-30 %), 0.68 -> 0.27 s on the 5 s clip
   (-60 %). The other H3 session's PR #55877 is this change; it measured as noise inside a 300 s end-to-end run,
   which is why it is not merged, but on the decode stage alone it is the cheapest step. Quad-safe (same code path).
   Companion: at 207 latents the tile-floor padding adds 247 tail-maintenance calls (206 ms of synced eager time);
   pipelines that can supply 32 extra latents of context, or a per-shard row count >= 32, avoid it entirely.
2. **Drop the fp32 hi/lo operand split where it buys nothing (`split_mode`).** Each 1-D conv is 2 typecasts +
   3 conv3d + 3 binary ops; Conv3d + Typecast + the split's share of BinaryNg is ~250 ms of the 581 ms device time.
   `split_mode="weight"` (2 conv3d, weight-only split) or `"off"` (1 conv3d, bf16 operands) cut that by a third to
   two thirds. Cost is accuracy, and it has to be measured in isolation: the Aug frontier only varied the levers
   jointly (defaults incl. the split: 81.9 dB vs the CPU reference on a 5 s clip; the "weight" variant was measured
   with the other fast levers on and landed near 48-51 dB, so the split's own share is unknown). Try per band: the
   late bands (C<=32) hold most of the conv time; the wide early bands likely hold the precision sensitivity.
3. **Fuse the anti-aliased Snake (UpSample1d -> snake_beta -> DownSample1d) into one depthwise kernel.**
   Today it is ~25 ops per activation (two 12-tap depthwise filters, each wrapped in shard/halo/unshard/untilize,
   plus concat, tilize, snake, untilize, phase permutes): 126 activations x ~25 ops = ~3,200 of the ~5,400 ops and
   roughly 200 ms of device time, most of it moving 1.9-3.8 MB tensors at a few GB/s. A single fused kernel (12-tap
   upsample, snake, 12-tap downsample, all elementwise per channel) would read and write each tensor once: this is
   the step that attacks both op count and per-op efficiency. Related prior work: PR #52991 (snake fused into the
   depthwise conv output, unmerged, never measured under sharding).
4. **Keep the late bands in a wide layout.** With C=8..32 fp32 a ROW_MAJOR row is 32-128 bytes and TILE pads the
   channels to 32: every op in bands 4-6 runs at 2-10 GB/s. Packing k consecutive time rows into one row
   (`(T/k, k*C)`) for the elementwise stretch between convs (snake, adds, tail masks) makes pages >= 1 KB at no
   arithmetic cost; the convs and resamplers need the `(T, C)` view but conv3d already reshapes internally. Expect
   2-5x on the ~300 ms of bands 5-6 elementwise/layout time; needs a layout-tax experiment first (`layout_tax.py`).
5. **Native interleaved depthwise 1-D filter.** Each 12-tap filter is 6 ops (interleaved->sharded, halo, move,
   conv2d, sharded->interleaved, untilize) of which conv2d is 42 µs and the plumbing ~100 µs; 489 filters per decode.
   A depthwise conv1d kernel that reads interleaved ROW_MAJOR directly removes ~1,900 ops and ~30 ms, and is a
   prerequisite for step 3 if it is built as a composite rather than one kernel.
6. **Smaller items visible in the profile:** the 7 composite band all-gathers (`AllBroadcast`, 3.8 ms, single core
   each) could be direct all-gathers if the band's T were tile-aligned; the 8 `MeshPartition`s and the
   gather-then-partition around every transposed conv (band gathers full T just to zero-stuff and re-split) could be
   replaced by a halo-only transposed conv; `dec_in_proj` still round-trips through the host (upload, conv, readback,
   upload) before the vocoder.

Not worth pursuing (measured or reasoned): channel parallelism (bands 0-2 are 14 % of device time), factor 32 on a
single galaxy (pads T to 1024 latents for 207), sweeping conv blockings (compute is 0.7 ms of the 581).
