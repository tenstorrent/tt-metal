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
└── pipelines/minimax_h3/             # t2va pipeline + host-side packing, scheduler, conditioning
```

There is no `__init__.py`: `tt_dit` uses namespace packages, matching `transformers/wan2_2/` and
`transformers/ltx/`. Import as e.g.
`from ....models.transformers.minimax_h3.attention_minimax_h3 import ...`.

Tests follow the layout of the other models, under `models/tt_dit/tests/models/minimax_h3/`.

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

The gates run at one shape and it is the one the perf log is tuned for:

| | |
|---|---|
| canvas | 1344x768 (16:9, the widest 768P canvas `resolve_canvas_size` yields) |
| frames | 124 @ 24 fps (5.17 s) -> 37 video latent frames, 207 audio latents |
| packed sequence | 37749 rows for a 39-token prompt (38222 at 512 tokens), padded to a multiple of SP x TILE |
| mesh | 4x8 Blackhole Galaxy, TP=4 axis 0, SP=8 axis 1, ring, 2 links |

Note the video VAE tiles this canvas **4x6 = 24** ways, and `test_performance_vae_minimax_h3.py`'s
`WORK_UNITS` table says 28. The projections in that file are slightly optimistic as a result.

## Fully-warm latency

Measured by `pipelines/ltx`'s method so the two are comparable: warmup pass first, prepares and
export excluded, `Total (compute)` = sum of the stage rows.

```bash
scripts/run_safe_pytest.sh models/tt_dit/tests/models/minimax_h3/test_performance_pipeline_minimax_h3.py
```

4x8 Blackhole, TP=4/SP=8, ring, 2 links · 1344x768, 124 frames @ 24 fps · 49 forwards:

| row | seconds |
|---|---|
| Encoder (cache) | 0.0 |
| Denoise | 61.7 |
| VAE decode | 17.6 |
| Audio decode | 1.8 |
| **Total (compute)** | **81.1** |

1259.9 ms per forward; realtime factor 15.7x. No tuning has been done.

**Always warm up before quoting a number.** A first call reports ~1.4x this total (denoise 104.7 s
vs 61.7 s), and the mp4 write and every weight load are excluded from the rows by design.
