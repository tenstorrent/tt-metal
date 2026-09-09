# FastH3 pipeline optimizations

FastH3 is `MiniMaxH3Pipeline` with a distillation adapter (4 forwards), optionally VSA, and a
T-sharded audio vocoder. This file is the ranked list of open levers, what each is worth **measured**,
and what has already been closed — so nobody spends device time re-deriving a settled result.

Companion docs: `models/MiniMaxH3.md` (component perf),
`models/transformers/minimax_h3/VSA_README.md` (VSA entry point), `VSA_STREAM_DESIGN.md` (fine-stage
kernel design and its measured ceiling), `VSA_PLAN.md` (VSA journal). Measurement log:
`~/.tt-buddy/notes/h3-fasth3-optimization-gaps.md`.

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

Per block, and scaled by `x50 blocks x4 forwards` to a share of the 20.1 s run. The `post` column is
job 885 on `68615f7c206`, same test / signpost window / 32-device merge, after the VSA op was brought
up to `cglagovich/fast_h3_vsa@2b72da0e1ae`:

| component | ms/block | post | % block (post) | share of run |
|---|---|---|---|---|
| `VsaSdpaOperation` (fine stage) | 19.81 | 21.52 | 35.6% | **4.30 s / 21.4%** |
| `AllGatherMinimalMatmulAsyncOp` x4 | 14.78 | 14.79 | 24.4% | **2.96 s / 14.7%** |
| `AllGatherAsyncDeviceOperation` (VSA K/V + pooled AG) | 8.09 (x2) | 8.40 (x4) | 13.9% | 1.68 s / 8.4% |
| TM/layout ops + their dispatch bubbles | ~10.0 | ~4.4 | ~7% | ~0.9 s / 4.4% |
| `EmbeddingsDeviceOperation` x6 (AdaLN gathers) | 5.32 | 5.32 | 8.8% | 1.06 s / 5.3% |
| `MinimalMatmulStridedReduceScatterAsync` | 2.66 | 2.65 | 4.4% | 0.53 s / 2.6% |
| `MatmulDeviceOperation` x6 | 2.42 | 1.65 | 2.7% | 0.33 s / 1.6% |
| `DitFusedDistributedRmsnorm` x4 | 1.62 | 1.62 | 2.7% | 0.32 s / 1.6% |
| `TopkLargeIndices` | 0.19 | 0.21 | 0.3% | — |
| **block device FW** | **63.25** | **60.54** | | |

Collectives and collective-fused matmuls together are **38.3% of the block** — larger than the SDPA
that dominates any single row.

Reading the port: device-side index assembly and the two coarse-stage `program_config`s took the
TM/layout bucket from ~10.0 to ~4.4 ms (`Concat` 0.95 -> 0, sub-20 us layout ops 47 -> 31) and the six
small matmuls from 2.42 to 1.65 ms; padded pooled gathers turned the composite pooled gather into two
aligned ring all-gathers (`AllGatherAsync` 2 ops -> 4, +0.31 ms, replacing a broadcast+concat chain
counted in the TM bucket before). Against that the v19 exact-numerics kernel costs **+1.71 ms**, not
the +26% (+5.2 ms) its standalone bench predicts — the deeper stream ring it enables and the
dense-row dealing absorb most of it. Net **-2.71 ms/block (-4.3%)**.

The block delta does **not** show up end to end. Job 886 on `02d862e71c4`, same shape / LoRA / knobs
as job 841, warm: Encoder 0.3 | Denoise **13.4** | VAE 4.8 | Audio 1.4 | **Total 19.9 s** against
20.1 s. The -2.71 ms/block predicts denoise 13.0 s; 13.4 s is what the pipeline reports, and a 0.2 s
run delta is far under the ~1.6 s single-run resolution this file's +-8% variance band implies.

**Treat the port as end-to-end flat.** What it bought is the exactness (dense-list rows at the bf16
floor instead of a 70%-off running sum), not latency. The block-level -4.3% is real in a more precise
instrument but is ~0.55 s of a 20 s run — below the noise floor of a single generation. Settling the
sign would take a repeated-run A/B, which is not worth the device time at this size.

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

`VsaSdpaOperation`, 21.52 ms/block on 120 cores, running the v19 leader/worker streaming program
factory (`ttnn/cpp/.../device/vsa_sdpa_stream_program_factory.cpp`). No easy config lever; this is
kernel work.

**The "is the sparsity actually exploited" question is answered, and the answer is that this row is
near its structural floor.** `VSA_STREAM_DESIGN.md` section 4 measures the kernel at 23-25% of HiFi2
peak on the listed math with a per-TRISC busy of PACK 93% / MATH 89% / UNPACK 87%, and localizes the
floor: at head dim 128 a 64-key visit's ~1024 FPU cycles are matched by ~1.1k cycles of SFPU exp plus
~0.9k of max/corr/rescale/sum that dense pays once per 512 keys. A row lists ~1 block in 9, so a
12-slot window averages ~1.3-2.4 selected blocks and the bookkeeping cannot amortise. Practical
ceiling of the design is ~26-28%, ~30% with every remaining pack/unpack trim. The levers already
measured and closed there: rows-for-depth, MOP PV, conditional rescale, q-tile pairing, selective K/V
gather, fp32 DEST, and the v18 distributed-window kernel (2x slower, NoC-bound).

What remains is a *granularity* change, not a tuning one: a 256-token VSA block would amortise every
per-visit cost 4x, but it changes the model's selection granularity and needs a quality gate.

### O3 — Layout thrash inside the block

**Largely closed by the port: TM/layout device time ~10.0 -> ~4.4 ms, sub-20 us ops 47 -> 31.**

The original finding was that 86% of the block's 4.53 ms op-to-op gap (3.91 ms) sat behind 47 ops with
under 20 us of device time — `UntilizeWithUnpadding` / `TilizeWithValPadding` / `Slice` at 1-5 us each
with 140-190 us gaps — plus ~6.1 ms of TM device time. Moving the coarse stage's index assembly onto
the device removed the concat / tilize / int32-blend / typecast / untilize chain outright (`Concat`
0.95 -> 0 ms).

What is left is **9 `Transpose` at 1.20 ms and `NlpCreateHeads` 0.61 / `Ternary` 0.53 /
`NLPConcatHeads` 0.32**, and 31 sub-20 us ops still paying full dispatch latency. `VSA_README.md`
names the transposes as foldable into the pooling layout (~3 ms of the coarse stage at 15 s, of which
0.9 ms is the DRAM-bound transposes). Right-sized as the next cleanup, not a perf project.

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

### O6 — Overlap the VSA K/V all-gather: blocked by sub-device ownership

**1.68 s of the run (8.4%). Tuning is settled — it is link-bound; only overlap is left.**

`AllGatherAsyncDeviceOperation`, 8.40 ms/block. `VSA_STREAM_DESIGN.md` section 5a swept every
`all_gather_async` configuration (Ring/Linear, persistent vs barrier semaphores, chunks_per_sync,
workers_per_link, buffers, and the generic `ttnn.all_gather`): a device receives 7 x 51.8 MB = 363 MB
per gather and the best configuration moves it at **87-93 GB/s, ~90% of 2 x 50 GB/s**. Ring is 1.9x
Linear; nothing else moves it, and the axis has only 2 ethernet channels so `--num-links 4` is not
available. The serial time cannot be tuned away.

The second-command-queue overlap **does not work**, and the reason is the dispatch model, not the
fabric. Job 908: a 4x8 ring-fabric mesh opens fine with `num_command_queues=2` on Blackhole (the
`ACTIVE_ETH` worry was unfounded), and `all_gather` alone measures 4.53 ms against 12.54 ms of filler
matmuls. But issuing the gather on CQ1 and compute on CQ0 aborts:

    TT_FATAL: Sub device id 0 currently in use by cq 1. Can't enqueue program from cq 0.
              Finish or wait for an event to transfer ownership.

A *program* enqueued on CQ1 takes exclusive ownership of the sub-device, so CQ0 cannot enqueue any
program until CQ1 finishes. That is why every `cq_id=1` call site in this repo is a host<->device
transfer — transfers do not enqueue programs and so never take ownership; trace + input streaming is
the only 2-CQ shape the dispatch model supports as-is. "Wait for an event to transfer ownership" is
serialization, which is the opposite of the goal.

Unblocking it needs **disjoint sub-devices**: CCL cores in one, compute cores in another, so the two
queues own separate sub-devices. `CCLManager._init_subdevice` and the AGMM grid's reserved CCL core
column mean the spatial split is half-built already. Sizing first, though: the hideable window is not
the full 8.40 ms. The gather's result is needed by `vsa_sdpa`, and between them sit the coarse stage
(~3.9 ms, of which ~0.5 ms is itself collective) and the independent gate branch (2.93 ms) — so ~6.3
ms, about 1.0-1.3 s of the run. A model-wide sub-device partition for 5-6% is a poor trade until
something else needs the same infrastructure.

The other structural option is unchanged: stream remote blocks inside the kernel over the fabric.
Selective gather stays dropped — a device needs 79% of the sequence on average.

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

### S1b — The VSA gate's matmul blocking: no headroom

`to_gate_compress` (`14400 x 1344 x 1792`, 5.89 ms, the largest of O1's four rows) was the only AGMM
shape whose `(K, N)` entry came from the divisibility constraints rather than a sweep. Job 905 swept
it at M=4768, 302 combos, 9m37s:

| combo | us | note |
|---|---|---|
| `(8, 7, 7)` sb `(4, 1)` | 746.6 | global best; `default_block_size` forces sb `(2, 2)`, so unreachable |
| `(8, 7, 10)` sb `(2, 2)` | 752.5 | best reachable |
| **`(8, 7, 8)` — shipped** | **755.5** | **0.4% off the reachable optimum** |
| median of 302 | 1157.7 | +55% |
| worst | 3126.1 | +319% |

Blocking matters enormously for this shape — only 9 of 302 combos land within 2% of best — and the
by-construction entry is already one of the 9. **There is nothing to win here**; do not re-sweep it.

What remains of O1 is therefore (a) the M=14400 re-sweep for the other three shapes, which is a
different question (does the optimum move with M?) and needs a top-N re-timing rather than a full
sweep — one shape at M=14400 extrapolates to ~29 min against the ~1506 s job cap — and (b) getting
input 0 out of `DEV_0_DRAM_INTERLEAVED`. Note also that an AGMM's device time includes its
all-gather (~116 MB/device over 2 links), so a meaningful part of these four rows is link-bound
communication that no blocking change can touch.

### S1c — Cross-request pipelining: there is no host tail to recover

Retracted before it was built. The idea was that ~18% of a generation is host time (the VAE's 40.7%
`readback` row plus audio's round trip) and could hide under the next request's denoise. It cannot,
because that 18% was derived by treating the VAE profile's phase timers as a device-vs-host split.
They are not: they are serial regions of the host wave loop, and `vae_minimax_h3.py`'s own schedule
already defers wave k's readback until wave k+1 is enqueued, so the transfer runs under the next
wave's compute. The `profile` flag that makes `device` and `readback` separable *serializes them to
do it* and is off by default.

S4 had already settled this from the other direction: 6.887 s traced against 6.934 s untraced, and
"the ~144 ms/chunk of eager dispatch was already hidden underneath". **The decode stage is
device-bound.** Device work is conserved across concurrent requests, so pipelining recovers only
host time, and the host time is already hidden.

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
   first. 2.96 s, firm diagnosis, no algorithmic risk. Now 24.4% of the block and the single largest
   lever by a wide margin; O2 is bigger on paper but is at its structural floor.
2. **O6's overlap** — the K/V all-gather on a second command queue behind the coarse stage. Tuning is
   settled (link-bound at ~90% of line rate), so overlap is the only move: ~8 ms/block, 1.68 s.
3. **O5** — pixel-comparing gate on yuv420 so the 24% becomes the default.
4. **O4 / O7** — few hundred ms, 1.06 s.
5. **O3's remainder** — fold the coarse stage's 9 transposes into the pooling layout, ~1 ms/block.
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
