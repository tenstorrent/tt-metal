# Qwen-Image 2.1 experimental TTNN denoiser

This directory ports the 32-block Qwen-Image 2.1 diffusion transformer to one
Blackhole P150 card. Its TTNN path includes text and image input projections,
timestep embedding and modulation, block-causal attention, the output head,
and FlowMatch Euler updates. The denoising schedule accepts image height and
width in multiples of 32 pixels and two or more steps. Qwen3-VL prompt
encoding, initial noise, and VAE decoding currently use the pinned CUDA
reference. This is a **hybrid validation prototype**, not a complete TT
text-to-image service. It does not yet implement the prefix KV cache, TT
encoder or TT VAE. Weights are reloaded from host storage each step.

The checkpoint is [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
at revision `790c92633540aa0cb11d9abf19eb46d861714758`, distributed under
the Qwen Research License. Weights and generated artifacts are not in this
repository. The CUDA oracle uses Diffusers commit
`5ff8e59ff9fe81c6e2df4fb4c6ea0d97a5df5ab2`, since the 0.40.0 wheel
does not include `QwenImage21Pipeline`.

## Layout and dependencies

- `tt/`: TTNN denoiser components.
- `validation/cuda_reference.py`: independent CUDA pipeline capture.
- `validation/cuda_schedule.py`: sigma schedule reconstruction and timestep check.
- `validation/tt_full_denoise.py`: full TT denoising loop with per-step CUDA comparisons.
- `validation/cuda_decode_latents.py`: CUDA VAE decode of the saved TT latent.
- `tests/test_denoiser.py`: hardware integration test with per-step PCC checks.

Use a tt-metal Python environment with a matching TTNN runtime, PyTorch,
pytest, and the Python reference dependencies in `requirements-reference.txt`.
The card must be explicitly reserved before running a test. The test runner
uses `TT_VISIBLE_DEVICES` to restrict TTNN to the supplied physical PCI BDF.
Commands below run from the tt-metal repository root. Keep checkpoints,
reference captures, and outputs on a data disk outside Git.

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
export QWEN_IMAGE21_DEVICE_BDF=0000:e1:00.0
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
