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
├── test_transformer_minimax_h3.py    # attention, one block, token refiner, precomputed AdaLN, whole DiT, Tracy block device-perf
├── test_vae_minimax_h3.py            # convs/resnets, encoder, 36-layer ViT decoder, tiling  (SINGLE_DEVICE)
├── test_vae_parallel_minimax_h3.py   # H/W sharding, data-parallel independence, device stitch  (mesh)
├── test_audio_minimax_h3.py          # weight-norm conversion, decode (accurate defaults), encode, traced
├── test_performance_minimax_h3.py    # t2va pipeline wall-clock (`BenchmarkProfiler`)
├── test_performance_vae_minimax_h3.py    # VAE perf, and the shared VAE test helpers others import
├── test_packing_minimax_h3.py        # host-only layout parity (t2va/fl2va)
├── test_references_minimax_h3.py     # ref2va host parity (prep/layout/presentation) + device encode gate
├── test_pipeline{,_fl2va,_ref2va}_minimax_h3.py   # one e2e quality gate each (t2va wall-clock is in test_performance_minimax_h3.py; fl2va/ref2va still log stage times)
└── tools/                            # not tests: perf projection, Tracy harnesses, VBench runner
```

`test_packing_minimax_h3.py` is kept separate from `test_references_minimax_h3.py` on purpose: its
golden digests are designed to stand in when the diffusers branch is absent, and the references file
`importorskip`s that branch at module level, which would turn those tests into skips.

## Running the transformer tests with real weights

`MINIMAX_H3_MODEL_PATH` points at a MiniMax-H3 diffusers snapshot (the transformer tests read its
`transformer/` partition). Without it, the real-weights cases skip and the rest still run.

```bash
export MINIMAX_H3_MODEL_PATH=/path/to/MiniMax-H3-diffusers
TEST=models/tt_dit/tests/models/minimax_h3/test_transformer_minimax_h3.py

# 2 layers, real weights, checked against the torch reference (PCC ~0.9998)
scripts/run_safe_pytest.sh $TEST -k "prod_768p_5s_real_weights"

# all 50 layers, real weights, device only -- no reference, so shape/finiteness checks only
scripts/run_safe_pytest.sh $TEST -k "transformer_real_weights"
```

Mind the two selectors: `prod_768p_5s_real_weights` is the 2-layer checkpoint case (the checkpoint
weights run only at the production shape), while `transformer_real_weights` is the separate
full-depth test. Plain `-k real_weights` matches both.

The 2-layer case reads only the first two blocks (62 of 638 tensors) and takes about a minute; the
50-layer case spends ~145 s loading weights onto the mesh.

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
the conditioning H3 expects; `test_minimax_h3_text_conditioner` asserts the tap is distinguishable
from that post-norm state.

Verified against the released weights on a 4x8 Blackhole Galaxy, TP=8 on axis 1 (no FSDP):
**PCC 99.9993%, RMSE/sigma 0.4%, at 128 tokens** —
`tests/models/minimax_h3/test_text_encoder_minimax_h3.py::test_minimax_h3_text_conditioner`.
A 50-layer causal stack accumulates over its context, so long prompts read lower: 99.9892% measured
at a 512-token prompt (TP=4 axis 0 with FSDP on axis 1), against 99.9999% at 13-22 tokens. The 0.99
bar leaves room for the production 512-token behaviour.

Two conditioner facts that break naive assumptions:

- **`head_dim` is 128, not `hidden_size // num_heads`.** 5120 / 64 = 80, and the derivation fails
  *silently* because 5120 % 64 == 0. The q/k/v inner dimension is 8192, wider than the 5120 residual
  stream. Always pass `head_dim` from config. (Qwen3-VL-8B, which Ideogram4 uses, happens to satisfy
  the derivation, so the hazard shows no signal there.)
- **`rope_scaling.mrope_interleaved` is true.** The chunked and interleaved rotary layouts coincide
  exactly while all three M-RoPE axes share a position — i.e. for `t2va`, where the flag is a no-op.
  A vision run makes them diverge. `create_rope_tensors(..., interleaved=True)` and
  `mrope_position_ids()` cover that; see `tests/encoders/qwen3vl/test_qwen3vl_mrope.py`.

FSDP is a placement choice, and the two consumers make it differently. The pipeline builds the
encoder with `is_fsdp=True` (`pipeline_minimax_h3.py`), sharding the weights across the non-TP axis
as well — ~1.6 GB/device for the 50.3 GB stack on 32 chips. `test_text_encoder_minimax_h3.py` leaves
`is_fsdp` at its False default: a 31.9 GiB Blackhole chip holds a TP=8 shard without it, so the
weights are simply replicated across the non-TP axis — load bandwidth, not capacity. (On a Wormhole
2x4, where TP=4 puts 14.9 GiB on a 12 GiB chip, FSDP is a capacity requirement.)

## Vision tower

`t2va` needs no vision at all: with no `pixel_values`, `Qwen3VLModel` never runs its vision tower and
never injects deepstack features, so the conditioner reduces to a plain text decoder. `fl2va` and
`ref2va` do feed vision, at **four** depths — the embedding scatter at `<|image_pad|>` positions, plus
additive deepstack injection at decoder layers 0/1/2 from vision layers `[8, 16, 24]`.

The tower is ported: `encoders/qwen3vl/vision_qwen3vl.py` (`Qwen3VlVisionModel`; the pipeline runs it
replicated, no TP),
wired into the decoder by `model_qwen3vl.py`'s `vision_embeds` / `vision_runs` / `deepstack_embeds`
forward arguments — merged tokens **replace** the `<|image_pad|>` row embeddings, deepstack features
are **added** to those same rows. Gated by `tests/encoders/qwen3vl/test_qwen3vl_vision_*.py` and, on
released weights, `tests/models/minimax_h3/test_vision_conditioner_minimax_h3.py`.

**The tower is green on released weights; the fused conditioner is not.** Merged tokens read 99.5953%
at the production 1344x768 canvas (~9.4% RMSE/sigma; 99.6532% measured at 448x448, which is not a
canvas `resolve_canvas_size` yields, so the gate runs the production canvas only).
`test_fused_conditioner_real_weights` is a strict `xfail` on its massive-activation row check: with
the HiFi4 decoder linears (see Precision) whole-tensor PCC is 85.82% at the production canvas, and
the port reproduces 6 of the reference's 7 massive-activation rows — row 63 missing, one spurious row
at 156. Do not read a green run of that file as `fl2va` being verified end to end.

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
export MINIMAX_H3_MODEL_PATH=/path/to/MiniMax-H3-diffusers
export TT_DIT_CACHE_DIR=~/tt_dit_cache        # see the warning below
scripts/run_safe_pytest.sh models/tt_dit/tests/models/minimax_h3/test_pipeline_minimax_h3.py
```

Artifacts land in `~/h3_t2va_artifacts`: `t2va.mp4` muxed, `t2va_silent.mp4`, `t2va.wav`.

The tier-6 quality gates (CLIP, VBench) always run; each skips only when its dependency is missing
(`open_clip` not installed, no `~/vbench_env` interpreter).

## Running `fl2va` end to end

Same command shape, plus a keyframe. `image=` is `fl2va`, `last_image=` is `fl2va_last_frame`, and
both together anchors each end of the clip:

```bash
export MINIMAX_H3_MODEL_PATH=/path/to/MiniMax-H3-diffusers
export TT_DIT_CACHE_DIR=~/tt_dit_cache
scripts/run_safe_pytest.sh models/tt_dit/tests/models/minimax_h3/test_pipeline_fl2va_minimax_h3.py
```

Artifacts land in `~/h3_t2va_artifacts` as `fl2va_<case>.mp4` / `_silent.mp4` / `.wav` plus four
inspection PNGs per case (`ref2va` writes to `~/h3_ref2va_artifacts`; the `ref2va` reference media
is read from `~/h3_fl2va_artifacts/fl2va_first.mp4`, skipping when absent).

**The gated keyframe is frame 0 of the calibrated `t2va` artifact**, read from
`~/h3_t2va_artifacts`, so run the `t2va` gate first — this one
skips rather than inventing content. The reason: a keyframe forces the content, and
`imaging_quality` is a no-reference IQA metric, so an arbitrary photograph would invalidate the
tier-6 calibration outright. Tier-6 numbers are therefore **recorded, not gated**, for `fl2va`.

A keyframe enters at two independent places, and both matter:

| | |
|---|---|
| the conditioner | `"<Picture 1>: "` + `<|vision_start|>` + 1008 x `<|image_pad|>` + `<|vision_end|>` + the prompt. The **whole vision block is video-tagged**, which is what the DiT's AdaLN keys off |
| the video VAE | `encode_clip` at `temporal_taps=1`, sampled posterior at seed **42**, rounded through **float16** before normalizing, then `scale_noise(rows, 0.999, noise)` |

The two read the same `(H/32) x (W/32)` grid: at 1344x768 that is 1008 image tokens **and** 1008
conditioning rows. Packed sequence 39746 -> 39936 padded for one anchor, 41756 -> 41984 for two.

Measured, all three cases green:

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
uv venv --python 3.10 ~/vbench_env
uv pip install --python ~/vbench_env/bin/python vbench decord \
    "numpy==1.26.4" "opencv-python-headless<4.11" "setuptools<81"
# VBench ships RAFT as a zip and there is no `unzip` on the box:
python -c "import zipfile; zipfile.ZipFile('$HOME/.cache/vbench/raft_model/models.zip').extractall('$HOME/.cache/vbench/raft_model')"
```

The interpreter path is fixed at `~/vbench_env/bin/python`; the test skips with this command if it
is missing, rather than passing.

**Set `TT_DIT_CACHE_DIR`.** Every component loads through `utils/cache.py`. With it set, end-to-end
is ~134 s; without it every run re-reads 62 GB of transformer and 50 GB of text encoder and takes
~713 s. Unset degrades *silently* — one log line, no error. First run populates ~68 GB of cache.

## Working point

The perf log is tuned for 768P/5s, and that is the shape every *component* gate runs at. The e2e
`t2va` gate sweeps six aspect ratios x 5/10/15 s (`test_pipeline_minimax_h3.py`'s `SWEEP`), which is
the only thing covering a long request end to end.

| | 5 s | 15 s |
|---|---|---|
| canvas | 1344x768 (16:9, the widest 768P canvas `resolve_canvas_size` yields) | same |
| frames | 124 @ 24 fps (5.17 s) -> 37 video latent frames, 207 audio latents | 362 (15.08 s) -> 107 latent frames, 603 audio latents |
| packed sequence | 37749 rows for a 39-token prompt (38222 at 512 tokens) | 109101 rows (109574 at 512 tokens) |

Audio latents occupy **two rows each**, one per channel, so a row count is `2 x latents`. Padding is
to a multiple of `SP x TILE`, which is 256 at SP=8 but **1024 at SP=32** — so the padded length, and
therefore every program in the 50-layer stack, is keyed differently on the two meshes.

The video VAE tiles this canvas **4x7 = 28** ways (256px tiles, overlap 64), matching
`test_performance_vae_minimax_h3.py`'s `WORK_UNITS` table and the wave math below.

### Meshes

Measured warm (the MEASUREMENT block in `test_performance_minimax_h3.py`), 768P/15s, 362 frames,
49 forwards:

| | 4x8 Galaxy | 4x32 quad (traced) | speedup |
|---|---|---|---|
| denoise | 252.3 s (5149 ms/fwd) | **92.9 s (1896 ms/fwd)** | 2.72x |
| VAE decode | 11.5 s | 11.7 s | 0.98x |
| audio decode | 4.7 s | 18.5 s | see below |
| **total** | **268.5 s** | **125.0 s** | **2.15x** |

Run-to-run variance at fixed shape and seed is around 8 %: a second quad run of the same case
measured denoise 85.3 s / total 118.0 s. One run does not establish a direction.

The audio row is **not** a like-for-like comparison and the total inherits that. The 4x8 column was
taken with the audio precision levers off, which was their default when it was measured; they are now
on by default (`split_mode="full"` on `MiniMaxH3AudioDecoder`),
which is an accuracy choice, not a regression -- the same levers cost the same on a single Galaxy.
Denoise, the row the mesh actually changes, is 2.7-3.0x.

At 768P/5s the quad is 41.5 s against 72.7 s, i.e. 1.75x -- both taken before the audio levers
flipped on, so both totals are understated by roughly the same amount. Two things the numbers
say plainly:

* **Tracing is what makes the quad pay at 5s; at 15s it only buys the first step.** Untraced,
  4x32/5s was *slower* than one Galaxy (134.0 s), because at 1184 rows/device a step is
  dispatch-bound; tracing took it to 41.5 s. At 15s each device holds 3424 rows, the loop is
  compute-bound, and the traced steady state (1835 ms/step) matches the untraced one (1745 ms/step) --
  but the first step still collapses 79.0 s -> 5.4 s, since capture allocates the CCL persistent
  buffers once instead of on the first step of every generation.
* **VAE and audio do not scale.** They are data-parallel over a fixed 28-tile work set, so they cost
  the same on 128 chips as on 32 and are now 26 % of the quad's total -- audio most of all, since the
  precision levers above tripled it. Reaching 4x end to end needs denoise at ~50 s *and* something
  done about those 31 s.

### Running on the quad

`tt-run` with `32x4_quad_bh_galaxy_rank_bindings.yaml` and the four hosts in physical ring order
(the wrong order fails as "Graph specified in MGD could not fit in the discovered physical topology").

`MINIMAX_H3_MODEL_PATH` and `TT_DIT_CACHE_DIR` have to be set in the rank binding's `global_env`, not
via `-x` in `--mpi-args`: tt-run builds a fixed per-rank environment, and `-x` lands in the first MPMD
app context only, so it reaches rank 0 and no one else. Anything the model branches on must be
identical on every rank -- a rank that takes a cache hit its peers miss skips collectives they are
still waiting in, and the run deadlocks rather than failing.

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
| Encoder | 0.0 s* | 0.0 s* | *measurement condition: these runs had the prompt embeddings pre-cached, a mechanism this branch does not have. A run of this branch pays the full conditioner encode here — ~2.8 s text-only with the encoder co-resident, plus the vision tower for `fl2va` — on top of the totals below |
| Keyframe encode | — | **0.1 s** | keyframe -> VAE moments -> posterior sample -> fp16 round trip -> normalize -> patchify -> `scale_noise` at t = 0.999 |
| Denoise | 67.0 s | 58.0 s | 49 forwards of the 50-layer DiT over the packed sequence |
| VAE decode | 4.0 s | 4.1 s | 196 work units in 7 waves of 28 across 32 devices |
| Audio decode | 1.7 s* | 1.7 s* | one pass over 207 latents x 2 channels. *Measured on the old fast path; the accurate-mode default is ~3x this stage time |
| **Total (compute)** | **72.7 s** | **63.9 s** | |
| per forward | 1366.5 ms | 1183.2 ms | |
| realtime factor | 14.1x | 12.4x | compute / video seconds |

**Do not read this as `fl2va` being faster than `t2va`.** It is one run of each, and run-to-run variance
at identical shape and seed is **±8 %** (denoise measured between 56.6 and 71.3 s across repeats).
What the numbers do establish is that `fl2va` is **not materially slower** despite a 5.4 % longer packed
sequence. Claiming a direction needs repeats. The co-residency notes in
`pipeline_minimax_h3.py` quote a different run of the same working point (total 69.1 s, VAE decode
6.0 s) — the two tables were measured on different days and sit within the variance band above.

Where the time goes, from Tracy captures:

| stage | device FW | dominated by |
|---|---|---|
| VAE decode, per work unit | 189.3 ms over 940 ops | matmul 36.6 %, SDPA 30.7 % (1.24 ms each). Most matmuls run at 26-52 % of peak with input 0 in DRAM and no `program_config` |
| Audio decode, whole stage | 1506.2 ms over 6449 ops | **layout, not arithmetic** — `Conv2d` is 4.0 % while concat/untilize/slice/tilize/permute together are ~70 % |

Op-to-op gap is 32.1 % of window wall for the video unit and 25.5 % for audio. No tuning has been done
on either.

**The decode stage is device-bound, not dispatch-bound.** Tracing the device-stitched per-chunk graph
(which removes per-op host dispatch entirely) measured 6.887 s against 6.934 s untraced at 768P/15s.
Replay costs 223 ms/chunk and issues no per-op host work, so that is real device execution, and the
~144 ms/chunk of eager dispatch was already hiding underneath it. `MINIMAX_H3_TIME_DISPATCH` reports
enqueue and post-synchronize time separately, but note the second number is only the tail the
synchronize still waits for -- reading it as total device time understates device work and makes the
stage look dispatch-bound when it is not. Do not re-derive this; the tracing experiment is not worth
repeating.

**Always warm up before quoting a number.** A first call reports ~1.4x the total (denoise 104.7 s
against 61.7 s in an earlier measurement), and the mp4 write and every weight load are excluded from the
rows by design. `warmup()` must be given the **real prompt and the real keyframes** — every program in
the 50-block stack is keyed on the padded packed length, so warming a different one warms nothing.
`run_warm_generation` asserts the warm and measured lengths agree; for t2va the hazard is
masked only by luck, since 1 and 39 tokens both round up to 37888.

## Precision

The conditioner's decoder linears run at **HiFi4** instead of the tt_dit-wide HiFi2 default.
`build_minimax_h3_text_encoder` opts in unconditionally (via `Qwen3VlTextEncoder`'s
`high_fidelity_linears`, which threads an explicit compute-kernel config to every qkv/o/gate/up/down
projection); there is no knob. Measured on the `fl2va` conditioner:

| decoder linears | fused conditioner PCC | massive-activation rows (reference has 7) | per forward |
|---|---|---|---|
| HiFi2 (tt_dit default) | 70.89 % | 5 | 1183.2 ms |
| HiFi4 (what runs) | **85.82 %** | **7** | 1184.2 ms |
| HiFi4 + no packer L1 acc | identical to HiFi4 | identical | — |

So it is free at this shape, `packer_l1_acc` has no effect, and the vision tower is unchanged — the gain
is all in the 50-layer decoder. **The shared default in `layers/linear.py` is deliberately left alone**:
one model's measurement is not evidence about LTX, Wan or Ideogram-4, which keep HiFi2. And it changes
the video not at all (40-48 dB PSNR frame-to-frame, identical anchor and CLIP numbers), so this is
conditioner fidelity rather than output quality.

## Audio decode precision

The audio VAE constructs in **accurate mode by default**: `MiniMaxH3AudioDecoder` /
`MiniMaxH3AudioEncoder` take `split_mode="full"` (and `max_c_in_block=128`) as constructor
defaults, and register the H3 conv blockings themselves. The one remaining lever answers a
hardware fact: an fp32 **multiply** through SrcA/SrcB keeps only ~11 significand bits (the FPU
takes ~5 mantissa bits per fidelity pass and HiFi4's 4 passes is the ceiling), so the error is
*flat in reduction depth* and neither `fp32_dest_acc_en` nor a higher fidelity can help.

- **`split_mode`** (`weight` = 2 convs, `full` = 3) splits a conv3d operand into `bf16 hi` plus its
  exact residual, so a second conv carries the mantissa bits the first dropped. A **3-way** split is
  bit-identical to a 2-way one, so 2-way already recovers the whole operand mantissa.

The retired `tap_matmul` lever reformulated stride-1 convs as per-tap matmuls to dodge conv3d's
partial-sum and output-path roundings; those are now fixed in the conv3d kernel itself (fp32
partials reduced on the SFPU, bias/untilize reading `UnpackToDestFp32` CBs — see
`conv3d/device/kernels/compute.cpp`), so conv3d+split matches the old tap+split accuracy in one
op with none of the per-tap weights or layout traffic.

The depthwise resample filters (`depthwise_tap_filter`) need no lever: their `ttnn.conv1d` kernel
accumulates on the SFPU for fp32 operands (`compute_depthwise_conv1d.cpp`) and measures bit-equal to
the exact shift-multiply-add form at every production shape, 2.1–4.1x faster. The retired
`prefer_mac` lever selected that MAC form as a precision workaround; MAC survives only as the
fallback for shapes conv1d cannot configure.

Two things that look like levers and are not: widening `C_in_block` helps an isolated conv (1.48x) but
**not** end to end, because the chain is dominated by the 126 narrow-channel AMP convs where it cannot
widen — and 512 fails outright. And `ttnn.snake_beta` is already fp32-grade at 7.2e-08, so the fused op
is not worth replacing.

The (default) accurate decode graph needs a **450 MB trace region** (375463936 B measured; the
retired fast path fit in 300 MB) and runs long enough to exceed the traced audio decode's 300 s
pytest timeout; pass `--timeout=1200`. The traced output matches eager exactly (PSNR inf).
