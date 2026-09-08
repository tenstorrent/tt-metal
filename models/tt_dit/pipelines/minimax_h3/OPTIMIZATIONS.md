# FastH3 pipeline optimizations

FastH3 is `MiniMaxH3Pipeline` with a distillation adapter (4 forwards), optionally VSA, and a
T-sharded audio vocoder. This file is the ranked list of open levers, what each is worth **measured**,
and what has already been closed — so nobody spends device time re-deriving a settled result.

Companion docs: `models/MiniMaxH3.md` (component perf), `models/transformers/minimax_h3/VSA_PLAN.md`
(VSA status). Measurement log: `~/.tt-buddy/notes/h3-fasth3-optimization-gaps.md`.

## Where the time goes

Job 841, 4x8 BH Galaxy, `tt-metal@fcdcae065c0`, 1344x768 / **15 s** / 362 frames, FastH3 LoRA
`vsa-synthetic-step1900`, VSA sparsity 0.9, `MINIMAX_H3_AUDIO_T_FACTOR=8`,
`vae_output_type="yuv420"`, warm:

| stage | now (yuv420) | before (float) | base 49-fwd | share |
|---|---|---|---|---|
| Encoder | 0.4 s | 0.4 s | — | 2% |
| **Denoise (4 fwd)** | **13.6 s** | 14.2 s | 265.7 s | **68%** |
| VAE decode | 4.7 s | 10.5 s | — | 23% |
| Audio decode | 1.4 s | 1.4 s | — | 7% |
| **Total (compute)** | **20.1 s** | 26.4 s | 296.6 s | |

20.1 s for a 15 s video is **1.33x realtime**. Denoise breakdown:
`preamble 0.1s (rope 0.1s) | first step 3.4s | steady 10.2s over 3 steps (3387 ms/step)`.

Run-to-run variance at fixed shape and seed is **±8%**. Any lever worth under ~1.6 s here cannot be
established by a single run.

## The step is the block stack, and nothing else

Job 842 Tracy capture, `test_minimax_h3_vsa_block_perf -k 15s_768p`, signpost window, 32 devices
merged — **one VSA block**:

| | |
|---|---|
| device FW | **63.25 ms** |
| op-to-op gap | **4.53 ms** (6.7% of window wall) |
| window | **67.78 ms** |

`50 blocks x 67.78 ms = 3389 ms` against a measured `3387 ms/step`. The block stack is the step to
within 0.06%, and 63.25 ms independently matches `VSA_PLAN.md`'s 63.7 ms.

Per block, and scaled by `x50 blocks x4 forwards` to a share of the 20.1 s run:

| component | ms/block | % block | share of run |
|---|---|---|---|
| `VsaSdpaOperation` (fine stage) | 19.81 | 31.3% | **3.96 s / 19.7%** |
| `AllGatherMinimalMatmulAsyncOp` x4 | 14.78 | 23.4% | **2.96 s / 14.7%** |
| `AllGatherAsyncDeviceOperation` x2 (VSA K/V AG) | 8.09 | 12.8% | 1.62 s / 8.1% |
| TM/layout ops (~60) + their dispatch bubbles | ~10.0 | ~15% | ~2.0 s / 10% |
| `EmbeddingsDeviceOperation` x6 (AdaLN gathers) | 5.32 | 8.4% | 1.06 s / 5.3% |
| `MinimalMatmulStridedReduceScatterAsync` | 2.66 | 4.2% | 0.53 s / 2.6% |
| `MatmulDeviceOperation` x6 | 2.42 | 3.8% | 0.48 s / 2.4% |
| `DitFusedDistributedRmsnorm` x4 | 1.62 | 2.6% | 0.32 s / 1.6% |
| `TopkLargeIndices` | 0.19 | 0.3% | — |

Collectives and collective-fused matmuls together are **40.4% of the block** — larger than the SDPA
that dominates any single row.

## Open levers, ranked by measured size

### O1 — `program_config` + L1 input on the four fused-collective matmuls

**2.96 s of the run (14.7%). Firmest evidence, no algorithmic change.**

All four are flagged `SLOW` with the identical advisory — *input 0 in `DEV_0_DRAM_INTERLEAVED`, no
`program_config` specified*:

| shape | ms | % block | DRAM % | FLOPs % | what it is |
|---|---|---|---|---|---|
| 14400 x 1344 x 7168 | 5.89 | 8.7% | **9.0%** | 15.2% | `to_gate_compress` (5376->7168), the VSA gate |
| 14400 x 1344 x 5376 | 3.81 | 5.6% | 12.9% | 17.6% | |
| 14400 x 1792 x 1344 | 2.85 | 4.2% | 19.0% | 7.9% | |
| 14400 x 1344 x 1792 | 2.22 | 3.3% | 18.7% | 10.1% | |

9-19% DRAM utilization on 112 cores. This is the same untuned-matmul diagnosis `MiniMaxH3.md`
records for the VAE decoder (O3), in a stage with a **3x larger denominator**. Start with the
14400 x 1344 x 7168 gate projection: 8.7% of every block at 9.0% DRAM is the worst offender.

Leave `MinimalMatmulStridedReduceScatterAsync 14400 x 3584 x 5376` alone — 125.8% FLOPs at HiFi4,
reported "Optimized".

### O2 — The VSA fine-stage SDPA

**3.96 s of the run (19.7%). Biggest single row, hardest to move.**

`VsaSdpaOperation`, 19.81 ms/block on 120 cores, already running the v3 leader/worker streaming
program factory that fixed v1/v2's DRAM-boundness
(`ttnn/cpp/ttnn/operations/transformer/sdpa/device/vsa_sdpa_stream_program_factory.cpp`). No easy
config lever; this is kernel work.

Cheaper question to ask first: at sparsity 0.9, k=179 of 1782 candidates, is 19.81 ms consistent with
the sparsity actually being exploited? `VSA_PLAN.md` measures the whole block at only 15% faster than
dense at 15 s. If the fine stage is not scaling with k, that is a bug-shaped finding, not a tuning
one — and it is worth a sparsity sweep on this same test before touching the kernel.

### O3 — Layout thrash inside the block

**~2.0 s of the run (10%), of which ~0.8 s is pure dispatch bubble.**

**86% of the block's 4.53 ms op-to-op gap (3.91 ms) sits behind 47 ops with under 20 us of device
time** — a run of `UntilizeWithUnpadding` / `TilizeWithValPadding` / `Slice` at 1-5 us device time
each with **140-190 us gaps**. Add ~6.1 ms of TM device time (`Transpose` 1.20, `Permute` 0.96,
`Concat` 0.95, `Untilize` 0.69, `NlpCreateHeads` 0.61, `Ternary` 0.54, `Tilize` 0.53, ...).

Same class as `MiniMaxH3.md`'s audio-decode finding (*"layout, not arithmetic"*). Attack the
untilize/tilize round trips: 47 sub-20 us ops per block, each paying full dispatch latency, is a
fusion or layout-contract problem rather than a kernel one.

### O4 — Tune the VAE decoder's linears

**Up to a few hundred ms. Firm evidence, modest ceiling.**

`MiniMaxH3.md`: 189.3 ms device FW per work unit over 940 ops, matmul 36.6%, and *"most matmuls run
at 26-52% of peak with input 0 in DRAM and no `program_config`"*. The SDPA was tuned
(`decoder_minimax_h3.py:91`, q=k=192 HiFi2, ~2.95x default blocking) and the elementwise ops moved
off HiFi4; the `Linear`s (`to_qkv`, `to_out`, `ff1`, `ff2`, `proj_in`, `proj_out`) were not.

Denominator after the yuv420 landing is 4.7 s of 20.1 s (23%), and matmul is 36.6% of that. Do the
same `program_config` work as O1 — the diagnosis is identical, the payoff smaller.

### O5 — Make `yuv420` the serving default

Landed as an opt-in knob in `fcdcae065c0` and measured: VAE decode **10.5 -> 4.7 s (-55%)**, total
**26.4 -> 20.1 s (-24%)**. Largest realized win to date.

`vae_output_type` still defaults to `"float"` because every pixel-comparing gate reads the float
path, and callers must branch on `MiniMaxH3Output.video_format`. **The open item is the gates, not
the knob** — until a pixel-comparing gate runs against yuv420, the 24% is only available to callers
who know to ask.

### O6 — Widen the VSA K/V all-gather

**1.62 s of the run (8.1%), and it runs on 20 cores of 120.**

`AllGatherAsyncDeviceOperation` x2, 8.09 ms/block. `VSA_PLAN.md` already named the K/V all-gather as
the fixed VSA-only cost that makes VSA *slower* than dense at 5 s. A 20-core collective on a
120-core part is worth a look at links and core assignment before anything algorithmic.

### O7 — The six AdaLN table gathers

**1.06 s of the run (5.3%).** `EmbeddingsDeviceOperation` x6 per block at ~890 us each on 120 cores.
The precomputed-AdaLN design deliberately trades modulation compute for these gathers; six per block
is the cost of that trade. Worth confirming six is the true minimum — the block needs modulation for
norm1/norm2/ff and their gates, so a fused or wider gather may serve several at once.

### O8 — Export `TT_DIT_CACHE_DIR`

Not in the 20.1 s compute total, but job 841 ran without it and paid
`CACHE MISS: weight_load minimax-h3/text_encoder (88.9s)` — a 90.9 s cold Encoder row against 0.4 s
warm. The cache exists (`/home/pshah/tt_dit_cache`, 219 G). Free.

### O9 — Close the audio decoder's projection->vocoder host round trip

`decoder_minimax_h3_audio.py:_project_latents_device` uploads, runs `dec_in_proj`,
`local_device_to_torch`s the result and transposes on host; `Vocoder._upload_BCT` transposes again
and re-uploads. Device -> host -> device on a `(2, 2048, T)` fp32 tensor between two device stages.
The module's own header docstring claims it *avoids* this round trip; the docstring is wrong.

Audio decode is 1.4 s of 20.1 s (7%) after the T-shard, so the ceiling is a few hundred ms.
Right-sized as a cleanup. Instrument `_decode_audio` (projection / upload / device / readback / post)
before treating it as a perf project.

### O10 — Repack the VAE decode waves

Wave occupancy is 87.5% at 5 s (196 units, 7 waves of 32) and 97% at 15 s — the last wave is padded
by repeating a unit (`vae_minimax_h3.py:913`). Worth ~0.6 s at 5 s and less at 15 s. Below the
variance band at this working point.

### O11 — Hoist the token refiner out of the step loop

`transformer_minimax_h3.py:509` runs `token_refiner(context_embedder(prompt))` every step; it is
step-invariant (no AdaLN, no rotary, no mask). 2 blocks over ~48 rows against 50 blocks over 14400
rows/device. Free to hoist, ~0.4%.

## Settled — do not re-run

### S1 — The ROW_MAJOR packing path is not where the time is

`transformer_minimax_h3.py:498-550` converts every stream to ROW_MAJOR, concats, pads and tilizes the
assembled sequence at full width **before** `mesh_partition`, on every forward. It looked like a free
8x. It is not worth anything: `50 x 67.78 ms = 3389 ms` against a measured `3387 ms/step` leaves
nothing outside the block stack. The packing path, `_vsa_gather_rows`, `proj_in`, `mesh_partition`,
the per-step index uploads and the host scheduler round trip are **collectively inside the noise**.

Do not build a full-forward profiled target for this. (A full forward is ~5000 ops and would blow
Tracy's 1000-op-per-device drop limit anyway.)

### S2 — Tracing the denoise on 4x8: dead

Warm pass: `first step 3.4s | steady 10.2s over 3 steps (3387 ms/step)`. The first step is **not**
anomalous when warm, so a trace has nothing to amortize. The cold pass's `first step 7.3s` is
per-process compile + lazy `CCLManager` persistent-buffer allocation, paid once per process, not per
request. `trace_denoise` stays quad-only, where a step at SP=32 really is dispatch-bound.

### S3 — Prompt-embedding disk cache: dead

The 0.4 s warm Encoder row is a genuine warm conditioner encode, so LTX's
`_device_embed_cache_path` (`pipeline_ltx.py:751`) is capped at 2% here. `MiniMaxH3.md`'s ~2.8 s
estimate for a cold co-resident encode does not describe this working point.

### S4 — Tracing the VAE decode: null result

`MiniMaxH3.md:373` — 6.887 s traced against 6.934 s untraced at 768P/15s. Replay costs 223 ms/chunk
and issues no per-op host work, so that is real device execution; the ~144 ms/chunk of eager dispatch
was already hidden underneath. **The decode stage is device-bound, not dispatch-bound.**

Corollary: `MINIMAX_H3_TIME_DISPATCH`'s post-synchronize number is only the tail the synchronize
still waits for. Reading it as total device time makes the stage look dispatch-bound when it is not.

### S5 — Tracing the audio vocoder: null result, and a net loss

`h3-audio-decode.md` iter 1, jobs 464/465: 6.6 s both arms, trace verifiably replayed, capture costs
+8.6 s on the first generation. Reserving the 1.2 GB trace region is itself free.

`decoder_minimax_h3_audio.forward`'s docstring — *"The vocoder is ~70% host-bound, so this is its
dominant lever"* — **is wrong**, and so is the same claim in `learn-minimax-h3-fasth3-pipeline.md`.
Tracing only removes per-op dispatch inside `_forward_device`; the stage's host time is mostly
outside it (O9). Independently, the T-shard measured 2.34x on the whole stage, which Amdahl caps at a
serial fraction below ~30%.

### S6 — Parallel video + audio decode: not worth it

Rejected on arithmetic, not difficulty.

* Audio needs only 8 of 32 chips — `ParallelFactor(factor=8, mesh_axis=sp_axis)` shards T across one
  axis and **replicates across the other**, so 3 of 4 Galaxy rows do redundant work for the stage.
* But video decode is data-parallel over a fixed tile work set (`wave_size = num_devices *
  waves_per_device`). Taking a row for audio sends 196 units from 7 waves to 9 — **+1.1 s on video to
  hide 1.1 s of audio** at 5 s. Net zero.
* Both stages are device-bound (S4), so there is no host/device asymmetry to exploit by interleaving
  on one queue.
* No precedent in this repo for concurrent dispatch to two submeshes from two host threads; every
  `create_submesh` call site carves one submesh for the whole run.
* At the measured numbers audio is **1.4 s of 20.1 s (7%)**, so a perfect free overlap would buy 7%
  — and it is not free.

The T-shard (landed) already captured the win this would have chased, at one constructor argument.

### S7 — Weight paging: not active

`coresident: True` on both `_PRESETS_BH` entries. The cost of *not* being co-resident is in
`_make_resident`'s docstring (VAE decode 6.0 s against 17.6 s) — that is the reason for the default,
not an open lever.

### S8 — Denoise precision: ~0.5%

`h3-bfp8-dit.md`. Only the activation-dtype arm has headroom.

### S9 — Watcher cannot run on this configuration

`_PRESETS_BH[(4,8)]` is `topology=Ring, num_links=2`. On Blackhole, eth kernel text counts against
the 25600 B `ACTIVE_ETH` kernel-config buffer and 2-erisc fabric packs two router binaries into it;
ring mode is already code-space-tight. `TT_METAL_WATCHER` (including `run_safe_pytest.sh --dev`)
overflows it and fails at `open_mesh_device` during "Initializing Fabric" with
`Program size (27968) too large for kernel config buffer (25600)`. The escape hatches
(`TT_METAL_DISABLE_FABRIC_TWO_ERISC=1`, `TT_METAL_FABRIC_OPT_LEVEL=Oz`) cost ~6% device time, so they
poison any measurement taken under them. Watcher is for hangs and memory corruption; it is not a
perf tool.

## Recommended order

1. **O1** — `program_config` + L1 input 0 on the four fused-collective matmuls, gate projection
   first. 2.96 s, firm diagnosis, no algorithmic risk.
2. **O2's cheap question** — sparsity sweep on `test_minimax_h3_vsa_block_perf` to check the fine
   stage actually scales with k, before any kernel work on the 3.96 s row.
3. **O3** — the 47 sub-20 us layout ops per block, ~2.0 s.
4. **O5** — pixel-comparing gate on yuv420 so the 24% becomes the default.
5. **O6 / O4 / O7** — 1.62 s, few hundred ms, 1.06 s.
6. **O8** — free, do it now.
7. Everything else is at or below the ±8% variance band.

## Profiling recipe

```bash
scripts/run_safe_pytest.sh --profile \
  models/tt_dit/tests/models/minimax_h3/test_vsa_performance_minimax_h3.py -k 15s_768p
python_env/bin/tt-perf-report <csv> --start-signpost start --end-signpost stop
```

* One duration per run (`-k`): a multi-parameter profiled run yields a CSV containing only the first
  parameter's ops.
* `--profile` **masks pytest's exit code** — read the pass/fail out of the log, never the exit status.
* `tt-perf-report` is not in `python_env` by default; install with `python_env/bin/uv pip install
  tt-perf-report` (plain `pip` misbehaves in this uv-managed env).
* Never combine with `--dev` / watcher — see S9.

## Standing caveat

Nobody has watched a FastH3 video. CLIP is -2.4 to -2.9 against base and cannot separate detail from
artifact; VBench has never run (`~/vbench_env` absent on g03blx04). `check_spatial_seams` logs
horizontal 0.666 at y=[160, 336, 512] against vertical 1.033 — a 33% deviation on three horizontal
VAE tile boundaries, diagnostic only, never asserted. **Every number in this file is latency.**
