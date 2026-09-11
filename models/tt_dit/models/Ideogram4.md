# Ideogram 4.0

## Introduction

[Ideogram 4.0](https://about.ideogram.ai/4.0) is a text-to-image generation model from Ideogram, notable for
best-in-class typography / text rendering. It was trained on structured-JSON captions, so quoting the exact text
strings (and optionally placing them with bounding boxes) is the reliable way to render legible lettering.

## Details

The pipeline composes four components:

- **Text encoder** — a Qwen3-VL-8B language model. Ideogram concatenates the raw hidden states after 13 selected
  decoder layers (a multi-layer "feature tap", no final norm) into a `13 × 4096 = 53248`-dim conditioning vector,
  assembled feature-major (`out[..., f*13 + l]`).
- **Denoiser** — a single-stream flow-matching DiT (34 layers, `emb_dim` 4608, 18 heads @ `head_dim` 256, SwiGLU MLP,
  4-branch tanh-gated AdaLN with no shift, per-head QK-RMSNorm, interleaved MRoPE). Text + image tokens share one
  sequence with full attention.
- **VAE decoder** — a diffusers `AutoencoderKL` (block_out `[128,256,512,512]`, 32 latent channels), reusing the SD3.5
  VAE decoder.
- **Sampler** — a logit-normal Euler flow-matching schedule with a per-step guidance schedule. Presets: `V4_TURBO_12`
  (12 steps), `V4_DEFAULT_20` (20), `V4_QUALITY_48` (48).

**Asymmetric CFG:** the conditional and unconditional passes are *distinct distilled networks* (not the same weights
run twice), so — unlike the cfg-parallel Flux/Qwen-Image pipelines — CFG is not batched or split across submeshes.
Both nets stay resident on the full mesh and run sequentially, and the velocity is blended on-device
(`v = gw·v_cond + (1-gw)·v_uncond` via `ttnn.lerp`).

## Performance

Measured on the Blackhole LoudBox (4×2), 512×512px, `V4_TURBO_12` (12 steps), program-cache warm, traced. Numbers are
emitted by `test_perf_ideogram4.py` (encode / denoise / decode section timings via the standard event → BenchmarkProfiler path).

| System  | CFG          | SP | TP | Resolution | Preset      |
|---------|--------------|----|----|------------|-------------|
| LoudBox | asym. (seq.) | 4  | 2  | 512²–2048² | V4_TURBO_12 |
| Galaxy  | asym. (seq.) | 8  | 4  | 512²–2048² | V4_TURBO_12 |

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- The gated Ideogram 4.0 fp8 checkpoint. Point `IDEOGRAM4_WEIGHTS` at the local checkpoint directory
  (containing `text_encoder/`, `transformer/`, `unconditional_transformer/`, `vae/`, `tokenizer/`).

## How to Run

```bash
# Gated fp8 checkpoint directory (required for real-weight generation / perf).
export IDEOGRAM4_WEIGHTS=/your/ideogram-4-fp8

# End-to-end generation on the LoudBox (4x2 mesh), SP4 x TP2. Saves a PNG to $IDEOGRAM4_OUT_DIR (default: generated/).
pytest models/tt_dit/tests/models/ideogram4/test_pipeline_class_ideogram4.py::test_pipeline_class -k "sp4tp2 and 512px and V4_TURBO_12"

# End-to-end generation on Galaxy (4x8 mesh), TP4 x SP8 (Ring).
pytest models/tt_dit/tests/models/ideogram4/test_pipeline_class_ideogram4.py::test_pipeline_class -k "bh_galaxy and 512px and V4_TURBO_12"

# Performance (encode / denoise / decode timings).
pytest models/tt_dit/tests/models/ideogram4/test_perf_ideogram4.py -k "sp4tp2"
```

## Scalability

Ideogram 4.0 runs on Blackhole: the 8-chip LoudBox (4×2 mesh, SP4 × TP2) and the 32-chip Galaxy (4×8 mesh,
TP4 × SP8, 2D-torus Ring). The parallel config is discovered from the mesh shape via `Ideogram4Pipeline`'s presets.

The DiT is parallelized on two axes:
1. `sp` (sequence parallel) — the single-stream sequence is fractured across a mesh axis; attention uses ring
   (joint) SDPA. This is the high-resolution win, where activations dominate.
2. `tp` (tensor parallel) — weights are fractured across a mesh axis (18 heads are padded up to a multiple of the
   TP factor); AllGather / ReduceScatter gather and scatter activations.

There is no `cfg` parallel axis: the two distilled nets are asymmetric, so each pass gets the whole mesh
sequentially (see **Asymmetric CFG** above). The text encoder (Qwen3-VL) and VAE decoder are tensor-parallel; the
encoder is additionally FSDP-sharded on the non-TP axis to free resident DRAM during denoise.
