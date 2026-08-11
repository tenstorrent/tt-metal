# MiniMax-H3

## Folder structure

Bringup is split by component. Each component has its own folder for team members to slot work into:

```
models/tt_dit/
├── models/
│   ├── MiniMaxH3.md                  # this file
│   ├── transformers/minimax_h3/      # denoising transformer (block, attention, RoPE)
│   ├── vae/minimax_h3/               # video VAE (AutoencoderKLMiniMaxH3)
│   └── audio_vae/minimax_h3/         # audio VAE (AutoencoderKLMiniMaxH3Audio)
├── encoders/qwen3vl/                 # text encoder — shared; loader_minimax_h3.py wires H3's 50-layer tap
└── pipelines/minimax_h3/             # t2va + fl2va pipeline, host-side packing, scheduler, conditioning
```

There is no `__init__.py`: `tt_dit` uses namespace packages, matching `transformers/wan2_2/` and
`transformers/ltx/`. Import as e.g.
`from ....models.transformers.minimax_h3.attention_minimax_h3 import ...`.

Tests live under `models/tt_dit/tests/models/minimax_h3/`, one file per subsystem as in
`tests/models/ltx/` and `tests/models/wan2_2/` — component gates sit beside the whole-subsystem gate
in the same file, sharing its module-level parametrize constants:

```
tests/models/minimax_h3/
├── test_transformer_minimax_h3.py    # attention, one block, token refiner, precomputed AdaLN, whole DiT
├── test_vae_minimax_h3.py            # convs/resnets, encoder, 36-layer ViT decoder, tiling  (SINGLE_DEVICE)
├── test_vae_parallel_minimax_h3.py   # H/W sharding, data-parallel independence, device stitch  (mesh)
├── test_audio_minimax_h3.py          # weight-norm conversion, decode, T-parallel, traced
├── test_performance_minimax_h3.py    # warm pipeline latency + per-block device time
├── test_performance_vae_minimax_h3.py    # VAE perf, and the shared VAE test helpers others import
├── test_packing*_minimax_h3.py       # host-only layout parity (t2va/fl2va, and ref2va)
├── test_pipeline{,_fl2va,_ref2va}_minimax_h3.py   # one e2e mode each, one process each
└── tools/                            # not tests: table builders, golden dumps, perf projection, VBench
```

`test_packing_minimax_h3.py` is kept separate from `test_packing_ref2va_minimax_h3.py` on purpose: its
golden digests are designed to stand in when the diffusers branch is absent, and the ref2va file
`importorskip`s that branch at module level, which would turn those tests into skips.

## Running the transformer tests with real weights

`MINIMAX_H3_MODEL_PATH` points at a MiniMax-H3 diffusers snapshot (`MINIMAX_H3_SUBFOLDER` picks the
partition, default `transformer`). Without it, the real-weights cases skip and the rest still run.

```bash
export MINIMAX_H3_MODEL_PATH=/data/cglagovich/MiniMax-H3-diffusers
TEST=models/tt_dit/tests/models/minimax_h3/test_transformer_minimax_h3.py

# 2 layers, real weights, checked against the torch reference (PCC ~0.9998)
scripts/run_safe_pytest.sh $TEST -k "real_weights- and small_s2048"

# all 50 layers, real weights, device only -- no reference, so shape/finiteness checks only
scripts/run_safe_pytest.sh $TEST -k "transformer_real_weights"
```

Mind the two selectors: `real_weights-` (trailing dash) is the 2-layer parameter, while
`transformer_real_weights` is the separate full-depth test. Plain `-k real_weights` matches both.

The 2-layer case reads only the first two blocks (62 of 638 tensors) and takes about a minute; the
50-layer case spends ~145 s loading weights onto the mesh. Block performance is tracked separately in
[MiniMaxH3_perf_log.md](MiniMaxH3_perf_log.md).

## Text conditioner

MiniMax-H3 conditions on **one** hidden state of its Qwen3-VL conditioner: `hidden_states[50]`, the
*unnormalized* output of **decoder layer 49**, mid-stack. The language-model head is never used and
the final norm is never applied.

**Mind the off-by-one.** `hidden_states` holds the embedding output plus one entry per layer, so
`hidden_states[50]` is the output of layer 49 and a **50-layer stack suffices** — the remaining 14 of
the checkpoint's 64 layers are never read, and neither is `lm_head`. `loader_minimax_h3.py` builds
this as `activation_layers=(num_layers - 1,)` with `num_layers = 50`, and
`load_minimax_h3_text_state_dict` reads 552 tensors from 12 of 14 shards, 50.3 GB bf16. Note that
`activation_layers=None` returns the *normalized* final state, which diffusers is explicit is **not**
the conditioning H3 expects; `test_tap_is_not_the_post_norm_state` pins that with a measurement.

Verified against the released weights on a 4x8 Blackhole Galaxy, TP=4 axis 0 with FSDP on axis 1:
**PCC 99.9892%, RMSE/sigma 1.5%, at the production 512-token prompt** —
`tests/models/minimax_h3/test_text_encoder_minimax_h3.py::test_text_encoder_tap_matches_reference`.
Short prompts read far better (99.9999% at 13-22 tokens); a 50-layer causal stack accumulates over
its context, so the bar is set from the 512-token row. See STATE.md amendment 76.

Two conditioner facts that break naive assumptions:

- **`head_dim` is 128, not `hidden_size // num_heads`.** 5120 / 64 = 80, and the derivation fails
  *silently* because 5120 % 64 == 0. The q/k/v inner dimension is 8192, wider than the 5120 residual
  stream. Always pass `head_dim` from config. (Qwen3-VL-8B, which Ideogram4 uses, happens to satisfy
  the derivation, which is why this went unnoticed.)
- **`rope_scaling.mrope_interleaved` is true.** The chunked and interleaved rotary layouts coincide
  exactly while all three M-RoPE axes share a position — i.e. for `t2va`, where the flag is a no-op.
  A vision run makes them diverge. `create_rope_tensors(..., interleaved=True)` and
  `mrope_position_ids()` cover that; see `tests/encoders/qwen3vl/test_qwen3vl_mrope.py`.

FSDP is not used: it was required on a Wormhole 2x4, where TP=4 puts 14.9 GiB of weights on a 12 GiB
chip, but a Blackhole chip is 31.9 GiB so even TP=4 fits. Without it the weights are replicated
across the non-TP axis — load bandwidth, not capacity.

## Vision tower

`t2va` needs no vision at all: with no `pixel_values`, `Qwen3VLModel` never runs its vision tower and
never injects deepstack features, so the conditioner reduces to a plain text decoder. `fl2va` and
`ref2va` do feed vision, at **four** depths — the embedding scatter at `<|image_pad|>` positions, plus
additive deepstack injection at decoder layers 0/1/2 from vision layers `[8, 16, 24]`.

The tower is ported: `encoders/qwen3vl/vision_qwen3vl.py` (`Qwen3VlVisionModel`, replicated, no TP),
wired into the decoder by `model_qwen3vl.py`'s `vision_embeds` / `vision_runs` / `deepstack_embeds`
forward arguments — merged tokens **replace** the `<|image_pad|>` row embeddings, deepstack features
are **added** to those same rows. Gated by `tests/encoders/qwen3vl/test_qwen3vl_vision_*.py` and, on
released weights, `tests/models/minimax_h3/test_vision_conditioner_minimax_h3.py`.

**The tower is green on released weights; the fused conditioner is not.** Merged tokens read 99.6532%
at 448x448 and 99.5953% at 1344x768 (~9.4% RMSE/sigma), but `test_fused_conditioner_real_weights` is
`xfail` at PCC 98.6224% against a 0.99 bar, cause not established. Do not read a green run of that
file as `fl2va` being verified end to end. See STATE.md amendment 90.

Note the demos port at `models/demos/qwen3_vl/` is built on `LightweightModule` /
`tt_transformers`, not `tt_dit`. It is an algorithm reference, not reusable code.

Measured facts (vision config: depth 27, hidden 1152, 16 heads, patch 16,
`spatial_merge_size` 2, `out_hidden_size` 5120):

- **`head_dim` is 72, and padding to 96 is mandatory.** ttnn SDPA hard-fails unpadded with
  `TT_FATAL logical_shape[3] == legacy_shape[3]`; padded it reaches PCC 0.9997 at seq_len 128/1024/4032.
  Costs 1.33x on attention. Pad the projection weights once at load time, as the demos port does.
- **`scale` must be passed explicitly** as `72 ** -0.5`. Padding to 96 would otherwise change the
  softmax temperature via SDPA's default — wrong output, not a crash.
- **`fl2va` needs no variable-length attention.** `cu_seqlens = repeat_interleave(h*w, t).cumsum()`, so
  one image is one block: a 768x1344 keyframe is grid `[1, 48, 84]` = 4032 patches with
  `cu_seqlens = [0, 4032]`, i.e. plain full attention. Only `ref2va` (up to 9 images and 3 videos, one
  block per *frame*) needs block-diagonal masking.
- **`fast_pos_embed_interpolate` is the common path, not an edge case.**
  `num_position_embeddings` is 2304 = 48², while a 16:9 keyframe is 4032 patches.

Sequence lengths for the canvases `resolve_canvas_size` produces:

| canvas | `grid_thw` | vision patches | LLM tokens |
|---|---|---|---|
| 768x1344 (16:9, max area) | `[1, 48, 84]` | 4032 | 1008 |
| 768x1024 (4:3) | `[1, 48, 64]` | 3072 | 768 |
| 768x768 (1:1) | `[1, 48, 48]` | 2304 | 576 |

A keyframe is put onto the target canvas *before* the processor sees it, so these are the only grids
`fl2va` produces. The 1008 LLM tokens of a 16:9 keyframe are also exactly `rows_per_frame`, the
condition-row count that anchor adds to the DiT's packed sequence — the same `(H/32) x (W/32)` grid
read by two different consumers.

## Setup

MiniMax-H3 support is not in a released `diffusers` yet. Bringup is pinned to a specific commit of
the `diffusers` main repository, which provides the reference `MiniMaxH3Transformer3DModel`,
`AutoencoderKLMiniMaxH3`, `AutoencoderKLMiniMaxH3Audio` and `MiniMaxH3Scheduler`.

Install it into the environment you run the tests from:

```bash
pip install "diffusers @ git+https://github.com/huggingface/diffusers@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
```

Verify the reference classes resolve:

```bash
python -c "from diffusers import MiniMaxH3Transformer3DModel; print('ok')"
```

### If the environment was created by `uv`

A `uv`-created virtualenv (such as `container_python_env` in the dev container) has no `pip` of its
own. After activating it, a bare `pip install` silently resolves to the system `pip` and installs to
`~/.local`, where the venv will not see it — it reports success and has no effect. Install through
`uv` against the interpreter instead:

```bash
uv pip install --python <venv>/bin/python --no-deps \
  "diffusers @ git+https://github.com/huggingface/diffusers@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
```

`--no-deps` keeps the resolver from pulling a newer `numpy` / `Pillow` / `huggingface-hub` into an
environment that `ttnn` was built against. The pinned commit's dependencies are already satisfied by
an environment that had any recent `diffusers` installed. Re-check `import ttnn` after installing.

## Running `t2va` end to end

One command, prompt in and an mp4 with a soundtrack out, at the production working point
(1344x768, 124 frames @ 24 fps, 50 scheduler steps -> 49 forwards):

```bash
export MINIMAX_H3_DIFFUSERS_DIR=/data/cglagovich/MiniMax-H3-diffusers
export TT_DIT_CACHE_DIR=/data/kevinmi/tt_dit_cache        # see the warning below
scripts/run_safe_pytest.sh models/tt_dit/tests/models/minimax_h3/test_pipeline_minimax_h3.py
```

Artifacts land in `$MINIMAX_H3_ARTIFACT_DIR` (default `~/h3_t2va_artifacts`): `t2va.mp4` muxed,
`t2va_silent.mp4`, `t2va.wav`.

`RUN_VBENCH=0` / `RUN_CLIP=0` skip the tier-6 quality gates, which default **on**.

## Running `fl2va` end to end

Same command shape, plus a keyframe. `image=` is `fl2va`, `last_image=` is `fl2va_last_frame`, and
both together anchors each end of the clip:

```bash
export MINIMAX_H3_DIFFUSERS_DIR=/data/cglagovich/MiniMax-H3-diffusers
export TT_DIT_CACHE_DIR=/data/kevinmi/tt_dit_cache
export MINIMAX_H3_ARTIFACT_DIR=~/h3_fl2va_artifacts
scripts/run_safe_pytest.sh models/tt_dit/tests/models/minimax_h3/test_pipeline_fl2va_minimax_h3.py
```

Artifacts land as `fl2va_<case>.mp4` / `_silent.mp4` / `.wav` plus four inspection PNGs per case.

**The gated keyframe is frame 0 of the calibrated `t2va` artifact**, read from
`MINIMAX_H3_T2VA_ARTIFACT_DIR` (default `~/h3_t2va_artifacts`), so run the `t2va` gate first — this one
skips rather than inventing content. The reason is amendment 87: a keyframe forces the content, and
`imaging_quality` is a no-reference IQA metric, so an arbitrary photograph would invalidate the
tier-6 calibration outright. Tier-6 numbers are therefore **recorded, not gated**, for `fl2va`.

A keyframe enters at two independent places, and both matter:

| | |
|---|---|
| the conditioner | `"<Picture 1>: "` + `<|vision_start|>` + 1008 x `<|image_pad|>` + `<|vision_end|>` + the prompt. The **whole vision block is video-tagged**, which is what the DiT's AdaLN keys off |
| the video VAE | `encode_clip` at `temporal_taps=1`, sampled posterior at seed **42**, rounded through **float16** before normalizing, then `scale_noise(rows, 0.999, noise)` |

The two read the same `(H/32) x (W/32)` grid: at 1344x768 that is 1008 image tokens **and** 1008
conditioning rows. Packed sequence 39746 -> 39936 padded for one anchor, 41756 -> 41984 for two.

Measured, all three cases green (STATE.md amendment 97):

| case | decoded anchor frame vs keyframe |
|---|---|
| `first` | frame 0, **PCC 0.9971** |
| `last` | frame -1, **PCC 0.9943** |
| `first`+`last` | frame 0 **0.9971**, frame -1 **0.9946** |

CLIP runs in-process (`open_clip` is already installed). **VBench does not, and cannot**: it pins
numpy < 2 and transformers 4.33, so installing it into `python_env` would downgrade numpy
2.2.6 -> 1.26.4 and transformers 5.12.1 -> 4.33.2, breaking `ttnn` and the Qwen3-VL reference. It
runs in its own interpreter against the written mp4, which needs no mesh:

```bash
uv venv --python 3.10 /data/kevinmi/vbench_env
uv pip install --python /data/kevinmi/vbench_env/bin/python vbench decord \
    "numpy==1.26.4" "opencv-python-headless<4.11" "setuptools<81"
# VBench ships RAFT as a zip and there is no `unzip` on the box:
python -c "import zipfile; zipfile.ZipFile('$HOME/.cache/vbench/raft_model/models.zip').extractall('$HOME/.cache/vbench/raft_model')"
```

Point `MINIMAX_H3_VBENCH_PYTHON` at that interpreter if it is not at the default path. The test
skips with this command if it is missing, rather than passing.

**Set `TT_DIT_CACHE_DIR`.** Every component loads through `utils/cache.py`, and prompt embeddings
are disk-cached alongside. With it set, end-to-end is ~134 s; without it every run re-reads 62 GB of
transformer and 50 GB of text encoder and takes ~713 s. Unset degrades *silently* — one log line, no
error. First run populates ~68 GB of cache.

## Working point

The perf log is tuned for 768P/5s, and that is the shape every *component* gate runs at. The e2e
`t2va` gate additionally runs 15 s (see `test_pipeline_minimax_h3.py`'s `DURATIONS`), which is the
only thing covering a long request end to end.

| | 5 s | 15 s |
|---|---|---|
| canvas | 1344x768 (16:9, the widest 768P canvas `resolve_canvas_size` yields) | same |
| frames | 124 @ 24 fps (5.17 s) -> 37 video latent frames, 207 audio latents | 362 (15.08 s) -> 107 latent frames, 603 audio latents |
| packed sequence | 37749 rows for a 39-token prompt (38222 at 512 tokens) | 109101 rows (109574 at 512 tokens) |

Audio latents occupy **two rows each**, one per channel, so a row count is `2 x latents`. Padding is
to a multiple of `SP x TILE`, which is 256 at SP=8 but **1024 at SP=32** — so the padded length, and
therefore every program in the 50-layer stack, is keyed differently on the two meshes.

### Meshes

| mesh | TP | SP | topology | links | rows/device at 5 s / 15 s |
|---|---|---|---|---|---|
| 4x8 Blackhole Galaxy | 4, axis 0 | 8, axis 1 | ring | 2 | 4736 / 13664 |
| 4x32 quad BH Galaxy, 4 MPI hosts | 4, axis 0 | 32, axis 1 | ring | 2 | 1184 / 3424 |

TP stays at 4 and SP absorbs the extra devices, following Wan2.2's `_PRESETS_BH`. TP is intra-host on
the quad and does a collective per layer, so it has to stay local and cheap; SP hides its KV
all-gather inside ring attention and tolerates the inter-host hop. The per-shape defaults live in
`_PRESETS_BH` in `pipelines/minimax_h3/pipeline_minimax_h3.py`.

Not yet swept at SP=32: the ring-SDPA chunk sizes in `MiniMaxH3Attention.measured_sdpa_chunk_sizes`,
which is keyed on the per-device length. At 4x32 all three durations miss and take the generic
(256, 512) fallback.

The video VAE tiles this canvas **4x7 = 28** ways at both durations (`split_tiles` with the real
`DEFAULT_TILE_OVERLAP = 64`; an assumed 32 gives 4x6, which is where an earlier note here got it
wrong). `test_performance_vae_minimax_h3.py`'s `WORK_UNITS` table agrees at 28.

## Fully-warm latency

Measured by `pipelines/ltx`'s method so the two are comparable: warmup pass first, prepares and
export excluded, `Total (compute)` = sum of the stage rows. Both tasks are measured in the same
process, minutes apart, so they are directly comparable.

```bash
scripts/run_safe_pytest.sh models/tt_dit/tests/models/minimax_h3/test_performance_minimax_h3.py
```

4x8 Blackhole, TP=4 axis 0 / SP=8 axis 1, ring, 2 links · 1344x768, 124 frames @ 24 fps · 49 forwards.
`fl2va` is one `first` anchor, packed length 39936 padded against t2va's 37888:

| stage | t2va | `fl2va` | what it is |
|---|---|---|---|
| Encoder (cache) | 0.0 s | 0.0 s | prompt embeddings from disk; a *miss* pays a full device conditioner encode here, and for `fl2va` that includes the vision tower |
| Keyframe encode | — | **0.1 s** | keyframe -> VAE moments -> posterior sample -> fp16 round trip -> normalize -> patchify -> `scale_noise` at t = 0.999 |
| Denoise | 67.0 s | 58.0 s | 49 forwards of the 50-layer DiT over the packed sequence |
| VAE decode | 4.0 s | 4.1 s | 196 work units in 7 waves of 28 across 32 devices |
| Audio decode | 1.7 s | 1.7 s | one pass over 207 latents x 2 channels |
| **Total (compute)** | **72.7 s** | **63.9 s** | |
| per forward | 1366.5 ms | 1183.2 ms | |
| realtime factor | 14.1x | 12.4x | compute / video seconds |

**Do not read this as `fl2va` being faster than `t2va`.** It is one run of each, and run-to-run variance
at identical shape and seed is **±8 %** (STATE.md amendment 82 measured denoise between 56.6 and 71.3 s).
What the numbers do establish is that `fl2va` is **not materially slower** despite a 5.4 % longer packed
sequence. Claiming a direction needs repeats.

Where the time goes, from the Tracy captures in STATE.md amendment 103:

| stage | device FW | dominated by |
|---|---|---|
| VAE decode, per work unit | 189.3 ms over 940 ops | matmul 36.6 %, SDPA 30.7 % (1.24 ms each). Most matmuls run at 26-52 % of peak with input 0 in DRAM and no `program_config` |
| Audio decode, whole stage | 1506.2 ms over 6449 ops | **layout, not arithmetic** — `Conv2d` is 4.0 % while concat/untilize/slice/tilize/permute together are ~70 % |

Op-to-op gap is 32.1 % of window wall for the video unit and 25.5 % for audio. No tuning has been done
on either.

**Always warm up before quoting a number.** A first call reports ~1.4x the total (denoise 104.7 s
against 61.7 s in an earlier measurement), and the mp4 write and every weight load are excluded from the
rows by design. `warmup()` must be given the **real prompt and the real keyframes** — every program in
the 50-block stack is keyed on the padded packed length, so warming a different one warms nothing.
`test_performance_minimax_h3.py` asserts the warm and measured lengths agree; t2va only ever
got away without that by luck, since 1 and 39 tokens both round up to 37888.

## Precision

`TT_DIT_LINEAR_PRECISION` (in `layers/linear.py`, shared by every tt_dit model) raises the math fidelity
of every linear. Unset keeps the long-standing config. Measured on the `fl2va` conditioner:

| | fused conditioner PCC | massive-activation rows (reference has 7) | per forward |
|---|---|---|---|
| unset | 70.89 % | 5 | 1183.2 ms |
| `hifi4` | **85.82 %** | **7** | 1184.2 ms |
| `max` | identical to `hifi4` | identical | — |

So it is free at this shape, `packer_l1_acc` has no effect, and the vision tower is unchanged — the gain
is all in the 50-layer decoder. **The default is deliberately left alone**: one model's measurement is
not evidence about LTX, Wan or Ideogram-4. And it changes the video not at all (40-48 dB PSNR
frame-to-frame, identical anchor and CLIP numbers), so this is a conditioner-fidelity knob rather than a
quality one. See STATE.md amendments 101-102.

## Audio decode precision

`MINIMAX_H3_AUDIO_ACCURATE=1` takes the audio VAE decode from 10.5 % to **0.45 %** relative RMSE against
the diffusers reference, for ~3x the stage time. It turns on three independent levers, each of which
targets a different one of the three error sources the default 10.5 % is made of. They are strongly
complementary — the chain error is set by whichever source is worst, so enabling one moves the total far
less than enabling all three:

| `MINIMAX_H3_AUDIO_CONV_SPLIT` | `..._DEPTHWISE_MAC` (removed; see below) | `..._TAP_MATMUL` | rel RMSE | PCC | PSNR | warm |
|---|---|---|---|---|---|---|
| `off` | 0 | 0 | 0.1046 | 99.5451 % | 40.29 dB | 4.03 s |
| `full` | 0 | 0 | 0.0538 | 99.8950 % | 46.07 dB | 5.36 s |
| `off` | 1 | 0 | 0.0920 | 99.6111 % | 41.41 dB | 8.72 s |
| `full` | 1 | 0 | 0.0320 | 99.9522 % | 50.58 dB | 9.50 s |
| `full` | 0 | 1 | 0.0371 | 99.9526 % | 49.31 dB | 9.97 s |
| **`full`** | **1** | **1** | **0.0045** | **99.9990 %** | **67.53 dB** | **13.24 s** |

Why each exists — all three answer the same hardware fact, that an fp32 **multiply** on this hardware
keeps only ~11 significand bits (the FPU takes ~5 mantissa bits per fidelity pass and HiFi4's 4 passes is
the ceiling), so the error is *flat in reduction depth* and neither `fp32_dest_acc_en` nor a higher
fidelity can help. Elementwise fp32 ops, by contrast, are exact.

- **`CONV_SPLIT`** (`weight` = 2 convs, `full` = 3) splits an operand into `bf16 hi` plus its exact
  residual, so a second conv carries the mantissa bits the first dropped. A **3-way** split is
  bit-identical to a 2-way one, so 2-way already recovers the whole operand mantissa.
- **`DEPTHWISE_MAC`** — **removed.** The anti-aliased resample filters now always use
  `ttnn.experimental.depthwise_fir1d`, which is fp32-exact (6–8e-08 across every production shape)
  *and* 3–12x faster than the shift-multiply-add form, so there is no trade-off left to expose. This
  was the single largest error source: one `Activation1d` injected 1.54e-03, *all* of it from its
  downsampler, against ~7e-08 for `snake_beta` and the upsampler. The cause was two roundings in
  `ttnn.conv1d` — the on-device tilize takes the activation to tf32, and the FPU multiply is 5b x 7b
  so even HiFi4 sees 9 significand bits of SrcA. `depthwise_fir1d` avoids both: it never tilizes, and
  it accumulates the taps on the SFPU. Audio decode 4.837 s → 1.483 s at equal accuracy.
- **`TAP_MATMUL`** runs stride-1 convs as `sum_j W_j @ x[t + dilation*j]`. conv3d's residual *after*
  splitting is partial-sum rounding across `C_in_block`, which matmul does not have; worth 1.8–3.5x per
  conv.

Two things that look like levers and are not: widening `C_in_block` helps an isolated conv (1.48x) but
**not** end to end, because the chain is dominated by the 126 narrow-channel AMP convs where it cannot
widen — and 512 fails outright. And `ttnn.snake_beta` is already fp32-grade at 7.2e-08, so the fused op
is not worth replacing. Full derivation in `STATE.md` amendments 111–113.

Accurate mode needs a **larger trace region** (375463936 B measured, against the default path's 300 MB)
and runs long enough to exceed the traced audio decode's 300 s pytest timeout; pass
`--timeout=1200`. The traced output matches eager exactly (PSNR inf).
