# Qwen-Image 2.1 experimental TTNN denoiser

This directory implements a batch-one text-to-image pipeline for the pinned
Qwen-Image 2.1 checkpoint on one Blackhole P150 card. Native mode runs the
Qwen3-VL prompt encoder, seeded Gaussian initial noise, all 32 diffusion
transformer blocks, FlowMatch Euler updates, and VAE decoder on TT. Host CPU
work includes tokenization, checkpoint loading, deterministic scheduler and
rotary metadata construction, and PNG writing. Native inference requires no
CUDA runtime or captured activations. The independent CUDA reference uses the
unmodified upstream pipeline from the same raw prompt.

This remains an experimental implementation. Image dimensions must be positive
multiples of 32 pixels and the configured schedule must contain at least two
steps. Batch size one and text-only generation are supported. Prompt expansion,
conditioning images, guidance, prefix KV caching, and VAE encoding are not
implemented in the TT path. Weights are reloaded from host storage each step.

The checkpoint is [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
at revision `790c92633540aa0cb11d9abf19eb46d861714758`, distributed under
the Qwen Research License. Weights and generated artifacts are not in this
repository. The CUDA oracle uses Diffusers commit
`5ff8e59ff9fe81c6e2df4fb4c6ea0d97a5df5ab2`, since the 0.40.0 wheel
does not include `QwenImage21Pipeline`.

## Layout and dependencies

- `tt/`: TTNN prompt encoder, Gaussian initialization, denoiser, scheduler, and VAE decoder.
- `validation/cuda_reference.py`: independent CUDA pipeline capture.
- `validation/cuda_schedule.py`: sigma schedule reconstruction and timestep check.
- `validation/tt_full_denoise.py`: full TT denoising loop with per-step CUDA comparisons.
- `validation/cuda_decode_latents.py`: CUDA VAE decode of the saved TT latent.
- `tests/test_denoiser.py`: hardware integration test with per-step PCC checks.
- `tests/test_native_pipeline.py`: raw-text native TT smoke/full test with CUDA disabled.

Use a tt-metal Python environment with a matching TTNN runtime, PyTorch,
pytest, pytest-timeout, and the Python reference dependencies in
`requirements-reference.txt`. The timeout plugin bounds the long hardware test.
The card must be explicitly reserved before running a test. Set
`TT_VISIBLE_DEVICES` to its physical PCI BDF before starting pytest so TTNN
sees the same card even if the test environment imports TTNN during collection.
Commands below run from the tt-metal repository root. Keep checkpoints,
reference captures, and outputs on a data disk outside Git.

An independent environment can be installed with the declared reference
requirements and a TTNN wheel matching the host's Metalium build:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python \
  -r models/experimental/qwen_image_2_1/requirements-reference.txt \
  torch==2.8.0 pytest==8.4.2 pytest-timeout==2.4.0 "$TTNN_WHEEL"
```

The fork-path hardware smoke test uses its own Python environment. Existing
compatible package files were reused through hardlinks after shared-cache
permissions prevented ordinary dependency installation; package imports resolve
inside that environment, and the pinned Diffusers Git revision and runtime
versions were checked. This is package-cache reuse, not a shared `PYTHONPATH`.

## Validation

Capture a complete CUDA reference on a CUDA host:

```bash
python -m models.experimental.qwen_image_2_1.validation.cuda_reference \
  --output-dir "$DATA/qwen-image-2-1/cuda" --height 256 --width 384 \
  --steps 20 --capture-steps all --full-capture-steps 0,19 --weight-layers ''
python -m models.experimental.qwen_image_2_1.validation.cuda_schedule \
  --cuda-dir "$DATA/qwen-image-2-1/cuda" \
  --output "$DATA/qwen-image-2-1/cuda/schedule.json"
python -m models.experimental.qwen_image_2_1.validation.select_cuda_capture \
  --cuda-dir "$DATA/qwen-image-2-1/cuda" \
  --output "$DATA/qwen-image-2-1/cuda/files-for-tt.txt"
```

Transfer the files listed in `files-for-tt.txt` to the TT host, preserving
their paths. The pinned checkpoint must also be available on that host. Then:

```bash
export QWEN_IMAGE21_CUDA_CAPTURE="$DATA/qwen-image-2-1/cuda"
export QWEN_IMAGE21_CHECKPOINT="$HF_SNAPSHOT/790c92633540aa0cb11d9abf19eb46d861714758"
export TT_VISIBLE_DEVICES=0000:e1:00.0
export QWEN_IMAGE21_TEST_STEPS=full
mkdir -p "$DATA/qwen-image-2-1/test-artifacts"
pytest -q models/experimental/qwen_image_2_1/tests/test_denoiser.py \
  --basetemp "$DATA/qwen-image-2-1/test-artifacts/full"
```

Without `QWEN_IMAGE21_TEST_STEPS=full`, the integration test checks one fresh
step. It requires a reference capture containing every step even for that
single-step test. The final TT latent can be decoded with
`validation.cuda_decode_latents` on a CUDA host. The decoded image is a TT
denoiser plus CUDA VAE result, not a fully TT-generated image.

## Verified scope

On 2026-09-29, a 20-step 384×256 run and a 40-step 256×256 run completed on a
Blackhole P150 in `f02cs02`, using BF16 and TTNN 0.65.1rc17 built against
tt-metal `e9351c92d66b4673743f9c0ed5a3346d4e644492`. Final latent
relative RMS errors versus the pinned CUDA pipeline were 13.14% and 9.24%,
respectively. The CUDA VAE decoded both final TT latents into recognizable
images, with mean pixel errors of 3.91/255 and 4.14/255 versus CUDA. The
migrated 20-step implementation passed the hardware test in 501.29 seconds,
with minimum velocity and latent PCC of 0.98168 and 0.99150 across all steps.
Its final TT latent file was byte-identical to the earlier 20-step run, and
the CUDA VAE re-decoded the CUDA latent pixel-for-pixel. The validation prompt
was "the quick brown fox jumps over the lazy dog"; this small CUDA reference
image itself does not fully show the dog. These are correctness checks, not
inference speed measurements. No steady-state latency or throughput is
claimed. Other cards, larger resolutions, and long-run stability are
unverified.

## Native integrated generation

Both commands compute all learned activations from raw text. The TT runner also
constructs its request metadata from checkpoint configuration instead of a CUDA
capture. Exported text-only encoder shards and decoder weights can be prepared
with `validation.cuda_encoder_reference` and `validation.cuda_vae_reference`;
these exports contain weights, not runtime inputs. The DiT checkpoint must
include `transformer/config.json` and `scheduler/scheduler_config.json`.

```bash
python -m models.experimental.qwen_image_2_1.validation.cuda_integrated \
  --output-dir "$DATA/qwen-image-2-1/native-cuda" \
  --prompt 'the quick brown fox jumps over the lazy dog' \
  --height 256 --width 384 --steps 20 --seed 42
CUDA_VISIBLE_DEVICES='' python -m models.experimental.qwen_image_2_1.validation.tt_full_denoise \
  --native --checkpoint "$HF_SNAPSHOT/790c92633540aa0cb11d9abf19eb46d861714758" \
  --tt-encoder-checkpoint "$ENCODER_CHECKPOINT/790c92633540aa0cb11d9abf19eb46d861714758" \
  --vae-checkpoint "$VAE_CHECKPOINT/790c92633540aa0cb11d9abf19eb46d861714758" \
  --device-bdf "$TT_VISIBLE_DEVICES" --output-dir "$DATA/qwen-image-2-1/native-tt" \
  --prompt 'the quick brown fox jumps over the lazy dog' \
  --height 256 --width 384 --steps 20 --seed 42
```

The two native RNG implementations do not produce identical tensors for equal
seeds. Independent native outputs therefore establish end-to-end functionality,
not component PCC. The separate module tests use identical inputs for meaningful
accuracy comparisons. The CUDA integrated runner records synchronized module timing automatically.
For TT, add `--timing-breakdown` to enable synchronized module timing. The
resulting `runtime.json` files include measurement scope. Synchronization and
observational capture affect runtime; previously completed uninstrumented runs
provide a separate integrated wall-time measurement.

The 2026-09-30 raw-prompt 384×256, 20-step, seed-42 native TT run completed in
568.45 seconds without CUDA availability or activation injection. The CUDA run
also completed from raw text. Prompt expansion was disabled on both backends;
these small diagnostic images do not establish semantic prompt adherence.
The previously captured same-input comparisons remain available for validation.

The full 36-layer prompt encoder reached PCC 0.999296 against CUDA, and the
same-input VAE decoder reached PCC 0.9999657. The native Gaussian transform
reached PCC 0.9999999956 when supplied the same uniforms as its arithmetic
reference; this comparison does not assert agreement between the independent
CUDA and TT random generators. Encoder, noise, and decoder hardware tests are
in `tests/test_encoder.py`, `tests/test_noise.py`, and `tests/test_vae.py`.

`validation.image_viewer --integrated` displays the two native images.
`validation.monitor_tt_denoise --host <ssh-alias> --remote-root <absolute-path>`
mirrors progress and fetches the final native TT image; it does not start a job
or perform CUDA preview decoding. Captured comparison reports live under
`validation/reports/` when available.

## Module accuracy and measured runtime reports

The [module PCC report](validation/reports/module_pcc_20260930.md) and its
[structured measurements](validation/reports/module_pcc_20260930.json) distinguish
same-input module comparisons from independent native generation. The fresh
native TT final-latent VAE comparison reached PCC 0.999951 against CUDA decoding
of that exact latent. The full prompt encoder reached PCC 0.999296.

The [runtime report](validation/reports/runtime_20260930.md) and
[structured timing data](validation/reports/runtime_20260930.json) record a
568.455-second native TT run and a 36.152-second native CUDA run. The report
spells out differing load/offload, synchronization, and artifact-writing
boundaries; these figures are not an equal-condition steady-state speed ratio.
Native generation uses independent backend RNG inputs, so final-image PCC
between those independently sampled runs is deliberately not reported.

To reproduce the native hardware smoke, configure `QWEN_IMAGE21_CHECKPOINT`,
`QWEN_IMAGE21_ENCODER_CHECKPOINT`, `QWEN_IMAGE21_VAE_CHECKPOINT`, and the reserved
`TT_VISIBLE_DEVICES` BDF, then run:

```bash
CUDA_VISIBLE_DEVICES='' QWEN_IMAGE21_TEST_NATIVE=1 QWEN_IMAGE21_TEST_STEPS=1 \
  .venv/bin/python -m pytest -q models/experimental/qwen_image_2_1/tests/test_native_pipeline.py
```

This decodes after the first step of a valid 20-step schedule to exercise every
pipeline module. Set `QWEN_IMAGE21_TEST_STEPS=full` for the complete generation.

The migrated fork package passed 12 tests (one optional oracle case skipped) on
2026-09-30, including native raw-prompt encoding/noise/one-step/TT-VAE with CUDA
unavailable, the paired first-step denoiser PCC regression, and metadata tests.
The native smoke image was byte-identical to the workspace timing diagnostic.
The complete 20-step native reference is reported separately; the smoke test
does not establish a full-generation accuracy result.
