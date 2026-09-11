# LTX-2.3 distilled: trace-bucket ladder

Branch: `nwoodall/ltx-trace-bucket-ladder` (off main `3125b4ba98f`). Hardware target: one
Blackhole Galaxy 4x8 (SP=8, TP=4, ring).

## Problem

The traced distilled AV pipeline captured one ttnn trace per exact denoise shape (stage 1 and
stage 2 of the warmup resolution/frame count). Any other resolution, fps, or duration meant a
new capture at request time: slow, and unsafe once other traces are live (a fresh allocation can
land inside another trace's activation region and get overwritten on replay). The console needs
a resolution/duration change to be a trace *replay*.

## What was built

The same idea MiniMax-H3 uses for variable-length prefill, ported to LTX:

1. **Pad the video sequence to a bucket rung.** `LTX_BUCKET_LADDER` in `models/tt_dit/utils/ltx.py`
   is 11 rungs, `8704 .. 261120`, each a multiple of 256 (`32 * SP`) and ~1.4x the previous one.
   A request's stage-1 and stage-2 token counts each round up to the smallest rung that fits.
2. **Mask the padded tail on-device.** Ring joint SDPA accepts `logical_n` as either an `int` or
   a one-element `uint32` device tensor (`LogicalLength`). The pipeline holds one such tensor per
   rung, writes the real token count into it before each request (`ttnn.copy`, address unchanged),
   and the kernels' reader/writer NoC-read it on replay. Self-attention and the V2A cross
   attention (`is_cross=True`) both use it. Padded key positions get no weight; padded query
   rows are zeroed by the existing padding masks.
3. **One trace per rung, all captured at startup.** A trace is stage-agnostic (only the token
   count matters), so the served grid of 128 `(canvas, fps, duration)` configs collapses to 52
   distinct token shapes and 11 rungs. Warmup preallocates every rung's persistent I/O *before*
   the first capture, then per rung runs an eager compile pass at the rung's exact shapes and a
   capture. After that, `generate()` only ever replays; a request that would need an unwarmed
   rung raises `ValueError` instead of capturing under live traces.

### Served envelope

| Canvas (name -> HxW)                    | fps            | duration (s) |
| --------------------------------------- | -------------- | ------------ |
| `720p-landscape` 704x1280, `720p-portrait` 1280x704   | 24, 25, 48, 50 | 6 .. 20 |
| `1080p-landscape` 1088x1920, `1080p-portrait` 1920x1088 | 24, 25, 48, 50 | 6 .. 20 |

`1440p-*` and `4k-*` canvases are defined in `LTX_CANVASES` but rejected in traced mode
(`LTX_SERVED_CANVASES`). Frame counts follow LTX's `8k+1` rule: `ceil((fps*s - 1)/8)*8 + 1`.
The audio latent is always run at the 512-frame bucket (`LTX_AUDIO_N_BUCKET`) and the decoded
waveform cropped to the clip duration.

### Files

| Area | Files | Change |
| ---- | ----- | ------ |
| SDPA kernel API (C++) | `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp`, `device/ring_joint_sdpa_device_operation.cpp`, `device/ring_joint_sdpa_program_factory.cpp`, `device/kernels/dataflow/ring_joint_writer.cpp`, `device/kernels/ring_joint_derived_slots.hpp` | H3's `LogicalLength = int \| Tensor` cherry-picked onto main (merged with main's `kv_actual_isl` metadata path); tensor `logical_n` allowed on the `is_cross` path. |
| Routing utilities | `models/tt_dit/utils/ltx.py` | `LTX_BUCKET_LADDER`, `LTX_CANVASES`, `LTX_SERVED_CANVASES`, `LTX_AUDIO_N_BUCKET`, `route_ltx_request` / `route_ltx_config` -> `LTXBucketRoute`, `ltx_served_configs`, `validate_bucket_ladder`. |
| Transformer | `models/tt_dit/models/transformers/ltx/{attention_ltx,transformer_ltx,rope_ltx}.py` | `video_logical_n_tensor` threaded to `inner_step`; ring SDPA called with `logical_n=<tensor>`; program configs keyed by the physical (padded) N with entries for every rung; RoPE built at the padded length. |
| Pipeline | `models/tt_dit/pipelines/ltx/{pipeline_ltx,pipeline_ltx_distilled}.py` | Trace state keyed by rung; per-rung `video_logical_n` StateTensor; statics refreshed in place; `warmup_buffers(served_configs=..., exact_hot_rungs=True)` with prealloc -> compile pass -> capture per rung; routing in `generate()`. |
| Matmul configs | `models/tt_dit/utils/matmul.py` | Fused MM+RS entries for per-device M = 11872 / 16640 / 23296 / 32640 (K=N=4096). |
| Device params | `models/tt_dit/tests/models/ltx/ltx_mesh_params.py` | `trace_region_size` 500 MB -> 1.6 GB. |
| Tests | `models/tt_dit/tests/models/ltx/{test_bucket_ltx,test_transformer_ltx,test_pipeline_ltx_distilled}.py`, `models/tt_dit/tests/unit/test_ring_joint_attention.py` | See below. |

### Knobs

- `LTX_SERVED_CONFIGS` (env) / `served_configs=` (`warmup_buffers`): `"hot"` (default: only
  the warmup shape), `"all"` (every rung, ~1 min and ~110 MB of trace region each on the 4x8),
  or `"canvas:fps:dur,canvas:fps:dur,..."`. Use `"hot"` while validating one configuration;
  opt into `"all"` for the final console startup once every rung is validated.
- `exact_hot_rungs` (default on): the warmup shape's own SP-padded lengths are added to the
  ladder as rungs, so the primary config pays nothing for bucketing.
- `trace_region_size`: ~108 MB per resident rung trace (5 rungs measured 543 MB). The traced
  device params now carry 1.6 GB, enough for the 11 ladder rungs plus the 2 hot-exact rungs.

### What is *not* bucketed

- Latent upsampler: pinned to the warmup shape (DRAM GroupNorm grid). A non-hot request rebuilds
  it eagerly for that request and drops its weights afterwards.
- VAE decode: traced only at the hot shape; other shapes decode eagerly.
- Audio decode: always at the 512 bucket, so one shape covers everything (first warmup at 512
  compiles a large set of new conv kernels, ~20 min cold; cached on disk afterwards).

## How to test

Prerequisites are the same as `LTX2.md` (checkpoint, Gemma, `TT_DIT_CACHE_DIR`). All device
commands below are for the Blackhole Galaxy 4x8 ring config.

```bash
cd ~/tt-metal && source python_env/bin/activate
```

### 1. Host-only routing tests (seconds, no device)

```bash
pytest models/tt_dit/tests/models/ltx/test_bucket_ltx.py
```

Ladder alignment, routing of the full served grid (128 configs -> 52 shapes -> 11 rungs),
extremes, rejections (unserved canvas / fps / SP != 8 / above ladder), RoPE padding.

### 2. SDPA kernel: tensor `logical_n` is bit-exact and replay-safe (~5 min)

```bash
pytest models/tt_dit/tests/unit/test_ring_joint_attention.py -k "logical_tensor_trace_replay and m4x8"
```

Captures one trace, then replays it with different lengths written into the tensor and asserts
the output is bit-identical to an eager run with the scalar length. Covers the self-attention
form (joint sharded / replicated) and the `is_cross` form LTX's V2A uses.

### 3. Transformer block at a rung (~3 min)

```bash
pytest models/tt_dit/tests/models/ltx/test_transformer_ltx.py -k test_ltx_transformer_block_bucket
```

Runs one 22B block at the SP-padded length with the scalar length (reference) and at the rung
with the tensor (bucketed). Video: PCC > 0.9999 and bit-exact when the physical shapes match.
The audio comparison is deliberately loose (`pcc=0.98`, `rmse=0.10`): the audio branch of the
block is run-to-run nondeterministic on main even with identical scalar inputs (measured
PCC ~0.998 / 5.7% RMSE, also with the A<->V cross attention skipped), which is a pre-existing
issue, not a bucketing one.

### 4. One pipeline, several configs, one warmup (the real check)

`test_pipeline_distilled_bucket_multi_rung` warms up once at the first listed config and then
calls `generate()` for every listed config on the same pipeline. It asserts the routing landed
on the expected rungs, that every rung's trace I/O buffer address is unchanged after all
requests (nothing was allocated under live traces), and that each MP4 has the requested frame
count and resolution.

A low and a high config after one warmup (the high one is the largest clip the eager VAE decode
is known to fit; see "Validation status" for why the 20 s clips are not there yet):

```bash
LTX_BUCKET_TEST_CONFIGS="720p-landscape:25:6,1080p-landscape:25:8" \
pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py \
  -k test_pipeline_distilled_bucket_multi_rung 2>&1 | tee /tmp/bucket_two.log
```

Default (three configs across five rungs: 720p/24/6, 1080p/25/8, 720p/48/10):

```bash
pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py \
  -k test_pipeline_distilled_bucket_multi_rung 2>&1 | tee /tmp/bucket_pipeline.log
```

What to look for in the log:

- `warmup (distilled 2-stage): ... N served config(s) -> K rung(s) [...]` lists the rungs.
- `capturing trace...` appears only during warmup, once per rung. If it appears after the first
  `generate`, routing is wrong.
- `generate <canvas> <frames>f@<fps>fps: <s> -> /tmp/pytest-.../bucket_*.mp4` per request; the
  MP4 paths are there for a visual check.
- A `ValueError: bucket rung ... has no preallocated trace I/O` means the request routed to a
  rung that was not warmed: add its config to `LTX_SERVED_CONFIGS`.

Note the test warms only the listed configs' rungs (it sets `LTX_SERVED_CONFIGS` itself). To
exercise the full console startup path, run the normal traced pipeline with
`LTX_SERVED_CONFIGS=all` (13 rungs) and then request two different configs.

### 5. Console-style run

Any existing traced entry point picks the default up unchanged, e.g.

```bash
LTX_TRACED=1 RUN_WARMUP=1 NO_PROMPT=1 pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py \
  -k "test_pipeline_distilled and 4x8sp1tp0nl2_ring_is_fsdp0" -s --timeout 7200
```

With the default `LTX_SERVED_CONFIGS=hot`, startup builds only the requested shape's two stage
traces. Set `LTX_SERVED_CONFIGS=all` to pay the longer startup cost once and make every
configuration in the envelope a replay.

## Validation status (Sep 10, 2026, 4x8 Galaxy)

- SDPA logical-tensor replay tests: bit-exact, including the new `is_cross` case.
- Block bucket tests: 6/6 pass (video + AV at 720p_s1, 1080p_s1, near-full rung).
- Multi-rung pipeline test, configs 720p/24/6 + 1080p/25/8 + 720p/50/20 (5 rungs: 4352, 16896,
  34560, 67840, 133120), one warmup (2380 s, of which ~30 min was the one-time cold audio-decode
  kernel compile at the 512 bucket):
  - 720p/24/6 (hot shape): `generate` 11.9 s end to end -- denoise, VAE and audio all trace replays.
  - 1080p/25/8: 600 s. Denoise replayed rungs 16896 + 67840; the rest is the eager upsampler
    rebuild at 1080p, the eager 201-frame VAE decode and first-time kernel compiles for those shapes.
  - 720p/50/20 (1001 frames): denoise replayed rungs 34560 + 133120, then the eager full-res VAE
    decode ran out of DRAM (`neighbor_pad_halo_scatter` needed 539 MB, 186 MB free; 31 of 32 GB
    per device allocated). Not a bucketing failure -- a 1001-frame clip had never been decoded on
    this pipeline -- but it means the 20 s end of the envelope is not deliverable until the VAE
    decode is chunked and/or DRAM is freed (below).
  - No `capturing trace...` after warmup; every rung's trace I/O address unchanged after all
    requests.
- 720p flat-output regression fixed: persistent upsampler/VAE/audio buffers are now allocated
  before DiT trace capture, so denoise replay cannot overwrite them. Video VAE tracing also
  honors `LTX_VAE_TRACE=0`; the validated path traces only the two DiT stages.
- Construction warmup now receives the requested FPS. For example, 1080p/50/20 routes its
  1001 frames as 50 FPS (`audio N=500`) instead of incorrectly treating them as 24 FPS
  (`audio N=1043`, above the 512 audio bucket).
- Rungs 186368 and 261120 have not yet been exercised on hardware.

### DRAM budget

Per device (32 GB): transformer weights ~11 GB (22B, TP=4, bf16), 1.6 GB trace region, Gemma /
VAE / upsampler weights, plus the per-rung persistent CCL buffers. The ring-SDPA K/V gather
ping-pong buffers are `2 (K,V) x 2 (ping/pong) x N x 2 KB` per rung -- ~1.1 GB for rung 133120
alone, ~2.6 GB for the five rungs above, and roughly 7-8 GB for the full 13-rung `"all"` set --
and the fused matmul reduce-scatter buffers add ~N x 1 KB x 2 per rung. This is what squeezed the
1001-frame VAE decode out. Two follow-ups, in order of payoff:

1. Single-buffer the persistent CCL buffers behind an explicit barrier instead of ping-ponging
   (halves the per-rung cost), and share one gather buffer sized for the largest rung across
   rungs where the trace can tolerate it.
2. Chunk the VAE decode temporally so long clips decode within a bounded working set.

## Known issues and follow-ups

- **Audio branch nondeterminism (pre-existing).** The transformer block's audio output differs
  run to run with identical inputs (see test 3). Worth a separate investigation; suspects are the
  gathered-K/V masked audio self-attention or the audio FFN.
- **`utils/mmrs_rules.py` wide-N branch (pre-existing).** For short-N shapes with per-core
  M >= 24 tiles it picks `N_block = 16` while budgeting only the circular buffers; the windowed
  L1 output handoff then clashes ("Statically allocated circular buffers ... clash with L1
  buffers"). Worked around with explicit table entries for the LTX rung shapes; the rule should
  either include the window in its budget or set `mm_window_blocks=None` on that branch. The
  four new entries are unswept (they reuse the swept stage-2 blocking) and can be tuned with
  `sweep_mm_block_sizes.py`.
- **Upsampler / VAE decode for non-hot shapes run eagerly.** Correct, but this is now the
  dominant per-request cost for non-hot configs (600 s for 1080p/25/8 vs 11.9 s for the hot
  shape) and the 1001-frame decode does not fit in DRAM (see "DRAM budget"). A per-canvas
  upsampler cache, a chunked VAE decode and single-buffered CCL buffers are the next steps.
- **1440p / 4k.** The ladder already covers 4k at 24 fps up to ~10 s (stage 2 -> rung 186368) and
  1440p further; enabling them is adding the canvas to `LTX_SERVED_CANVASES`. Expect the eager
  4k VAE decode and the upsampler rebuild to be the memory problems, not the transformer.
