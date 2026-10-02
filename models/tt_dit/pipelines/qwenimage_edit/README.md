# Qwen-Image-Edit (WH Galaxy, 32 chips)

Single-image [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) on a 32-chip
Wormhole Galaxy, built on the shared `tt_dit` infra. The denoise transformer and the VAE (encode +
decode) run on device; the Qwen2.5-VL image+text encode runs on host.

Two denoise layouts on the (4, 8) mesh:

- **CFG-parallel (default, `cfg_parallel=True`)**: two 4x4 submeshes, **TP=4 x SP=4** each. The cond
  and uncond forwards of every step run concurrently, one per submesh -- the same layout the base
  `pipelines/qwenimage` uses at (4, 8).
- **Sequential (`cfg_parallel=False`)**: **TP=8 x SP=4** on all 32 chips; cond and uncond run back to back.

## What runs where

| Stage | Where | Notes |
| --- | --- | --- |
| VL image+text encode | host (torch) | device encoder is text-only (no vision tower yet); vision features computed once and reused for the negative prompt |
| Denoise transformer | device | `QwenImageTransformer`, traced, one trace per CFG branch |
| VAE encode + decode | device | WAN-architecture, sharded over (H, W); on submesh 0 with CFG-parallel |
| true-CFG, scheduler | host | same math as the reference `diffusers.QwenImageEditPipeline` loop |

## How to run

The device transformer and VAE weights are cached under `TT_DIT_CACHE_DIR` (first run populates it).

```bash
export TT_DIT_CACHE_DIR=/path/to/.ttdit_cache
python_env/bin/python -m pytest \
  models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage_edit.py::test_qwenimage_edit_pipeline \
  -k cfgpar -s --timeout=0
```

Programmatic use:

```python
from models.tt_dit.pipelines.qwenimage_edit import QwenImageEditPipeline
from PIL import Image

pipe = QwenImageEditPipeline.create_pipeline(mesh_device=mesh_device)  # CFG-parallel, traced, device VAE
out = pipe(
    image=Image.open("input.png"),
    prompt="Give the cat a blue wizard hat.",
    num_inference_steps=50,  # 20-30 is usually enough
    true_cfg_scale=4.0,
    side=1024,  # square canvas; letterbox=True preserves aspect ratio
)
out[0].save("edit.png")
```

Notes:
- `side` must be square and SP-aligned (1024 works): the SP ring-attention kernel requires the
  combined (noise + condition) token sequence to divide evenly across SP and be tile-aligned.
  `letterbox=True` (default) pads the input to square without distortion.
- Traced inputs: a device tensor allocated after a trace is captured can be overwritten when that
  trace replays. Each CFG branch therefore allocates all of its trace inputs before any capture
  and writes them in place every step (`_DenoiseBranch`). Without this, the sequential layout's
  uncond trace read clobbered prompt/RoPE buffers and produced noise images.

## How we tested

`tests/models/qwenimage/test_pipeline_qwenimage_edit.py` runs the full pipeline on the Galaxy at
1024^2, 50 steps, prompt "Give the cat a blue wizard hat." on `models/sample_data/huggingface_cat_image.jpg`:

- `4x8_cfgpar_devvae_full_50steps` -- CFG-parallel, full device VAE
- `4x8_devvae_full_50steps` -- sequential TP=8, full device VAE
- `4x8_devdecode_hostencode_50steps` -- sequential TP=8, device VAE decode, host VAE encode

Each saves `edit_pipeline_output_*.png` for visual inspection. Both layouts produce the same edit
(PSNR ~22 dB between them, from the different TP split), and repeated calls with the same seed are
bit-identical.

If outputs are non-deterministic across identical runs, check the hardware first: a chip returning
wrong results intermittently was found on a test Galaxy, and `tt-smi -glx_reset_auto` cleared it.

## Performance (current)

Measured on WH Galaxy, 1024^2, 50 steps, full device VAE:

| Metric | CFG-parallel (first call / warm) | Sequential TP=8 |
| --- | --- | --- |
| Per step (cond + uncond) | 725 ms | 938 ms |
| Denoise | 41.3 s / 36.3 s | 55.9 s |
| VL encode (host) | 12.4 s / 8.5 s | 12.0 s |
| Wall (end to end) | 55.2 s / 45.7 s | 70.2 s |

Optimizations: trace replay per CFG branch, CFG-parallel submeshes (both forwards in flight
before either is read), readback of only the TP-rank-0 shards that hold the noise tokens
(~78 ms -> ~2 ms per step), vision features shared between the prompt and negative-prompt encodes,
and on-device VAE.

Known headroom: the host VL encode (~8.5 s warm) -- the Qwen2.5-VL language model could run on
the tt_dit device encoder with host-computed image embeddings; per-forward kernel time (706 ms at
TP=4 x SP=4); fewer steps (20-30).
