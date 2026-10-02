# Qwen-Image-Edit (WH Galaxy, TP=8 x SP=4)

Single-image [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) on a 32-chip
Wormhole Galaxy, built on the shared `tt_dit` infra. The denoise transformer runs on **all 32
chips** as **TP=8 (heads) x SP=4 (sequence)** -- the same sequence-parallel layout Flux uses on
WH Galaxy -- and the VAE (encode + decode) also runs on device. Text/VL encoding, the scheduler,
and true-CFG are driven by the reference `diffusers.QwenImageEditPipeline` on host.

## What runs where

| Stage | Where | Notes |
| --- | --- | --- |
| VL image+text encode | host (torch) | device encoder is text-only (no vision tower yet) |
| Denoise transformer | device | `QwenImageTransformer`, TP=8 x SP=4, traced |
| VAE encode + decode | device | WAN-architecture, sharded over (H, W) |
| true-CFG, scheduler | host | reference pipeline, bit-faithful |

## How to run

The device transformer and VAE weights are cached under `TT_DIT_CACHE_DIR` (first run populates it).

```bash
export TT_DIT_CACHE_DIR=/home/tt-admin/teja/.ttdit_cache
python_env/bin/python -m pytest \
  models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage_edit.py::test_qwenimage_edit_pipeline \
  -k devvae_full -s
```

Programmatic use:

```python
from models.tt_dit.pipelines.qwenimage_edit import QwenImageEditPipeline
from PIL import Image

pipe = QwenImageEditPipeline.create_pipeline(mesh_device=mesh_device)  # trace + device VAE on
out = pipe(
    image=Image.open("input.png"),
    prompt="Give the cat a blue wizard hat.",
    num_inference_steps=50,   # 20-30 is usually enough
    true_cfg_scale=4.0,
    side=1024,                # square canvas; letterbox=True preserves aspect ratio
)
out[0].save("edit.png")
```

Notes:
- `side` must be square and SP-aligned (1024 works): the SP ring-attention kernel requires the
  combined (noise + condition) token sequence to divide evenly across SP and be tile-aligned.
  `letterbox=True` (default) pads the input to square without distortion.
- `batch_cfg` is **off by default**: a batch-2 cond/uncond forward was measured ~3.1x a batch-1
  forward on this SP=4 layout (net regression), so sequential true-CFG is kept.
- `prompt_bucket` (default 128): the denoise trace is captured per prompt length, so the prompt
  (VL text + image tokens) is zero-padded up to a multiple of 128 and prompts of nearby lengths
  replay one trace instead of re-capturing it. The transformer has no key mask, so the padding is
  attended to, as in the base qwenimage pipeline; `prompt_bucket=None` keeps exact lengths. The
  prompt embeddings are re-uploaded once per call, so a pipeline object can serve many edits.

## How we tested

`tests/models/qwenimage/test_pipeline_qwenimage_edit.py` runs the full pipeline on the Galaxy at
1024^2, 50 steps, prompt "Give the cat a blue wizard hat." on `models/sample_data/huggingface_cat_image.jpg`:

- `4x8_devdecode_hostencode_50steps` -- device VAE decode, host VAE encode
- `4x8_devvae_full_50steps` -- full device VAE (encode + decode)

Each saves `edit_pipeline_output_{devdecode,devvae_full}.png` for visual inspection. Outputs are
correct 1024^2 edits (the edit is applied; background/aspect preserved).

## Performance (current)

Measured on WH Galaxy, 1024^2, 50 steps (100 transformer forwards), full device VAE:

| Metric | Value |
| --- | --- |
| Warm per-forward (traced) | ~460 ms |
| Denoise (device) | ~55.6 s |
| Wall (end to end) | ~88 s |

Optimizations enabled: trace replay of the denoise, on-device VAE, and caching the step-invariant
RoPE/prompt tensors (built once, reused across steps) to cut per-step host overhead
(~98 s -> ~88 s wall).

Known headroom: the ~32 s non-denoise tail is dominated by the host VL encode (vision tower not yet
on device). Fewer steps (20-30) and CFG device-parallel (TP=4, two 16-chip submeshes) are the main
levers to approach Flux-class latency.
