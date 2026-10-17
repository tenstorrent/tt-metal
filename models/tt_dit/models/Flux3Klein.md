# FLUX 3 Image Klein

## Introduction

[FLUX 3 Image Klein](https://github.com/black-forest-labs/flux-3-image-klein) is Black Forest Labs' distilled
flow-map DiT for text-to-image generation and reference-image editing (up to 10 references, up to 4K). It ships
in two checkpoints, `flux-3-image-klein` (distilled, 16 steps, no guidance) and `flux-3-image-klein-base`
(50 steps, CFG 3.5), plus an Apache-2.0 prompt upsampler (`flux-3-image-klein-pup`).

This port targets the Blackhole Galaxy (4x8 mesh). Both checkpoints are selectable. The base checkpoint is
brought up first (50 Euler steps, CFG 3.5 through a two-sample pack); the distilled checkpoint shares the
architecture apart from its flow-map `ref_time_in` head and is added second.

**Status: bringup.** The sections below describe the layout and the contract; every "how to run" command and
every performance cell is a placeholder until the corresponding milestone lands.

## Folder structure

Bringup is split by component. Each component has its own folder for team members to slot work into:

```
models/tt_dit/
├── models/
│   ├── Flux3Klein.md                      # this file
│   ├── transformers/flux3_klein/          # Flux3KleinTransformer, the block, attention, modulation table, RoPE, matmul tables
│   └── vae/                               # vae_flux2.py reused unchanged for decode; the reference encoder is new
├── encoders/qwen3vl/                      # text encoder — shared; loader_flux3_klein.py wires Klein's 8 layer taps
├── pipelines/flux3_klein/                 # pipeline, prompt encoder, upsampler, host packing, scheduler, references, weights resolver
└── reference/flux3_klein/                 # vendored torch reference (BFL src/flux3, Apache-2.0), CPU-capable
```

There is no `__init__.py`: `tt_dit` uses namespace packages, matching `transformers/minimax_h3/` and
`transformers/wan2_2/`. Import as e.g.
`from ....models.transformers.flux3_klein.transformer_flux3_klein import ...`. Every Klein-specific file
carries the `_flux3_klein` suffix; the directory is `flux3_klein` rather than `flux3` because BFL's package is
named `flux3` and the base checkpoint shares the architecture.

Tests live under `models/tt_dit/tests/models/flux3_klein/`, split per component (the `flux1` / `motif` style)
so the risky pieces gate individually:

```
tests/models/flux3_klein/
├── device_params.py                       # line / ring / 8 KB-packet / req-exact device params (flux2 style)
├── common.py                              # checkpoint and goldens resolution, reference loaders, mesh rows
├── test_reference_flux3_klein.py          # CPU: vendored torch reference vs GPU goldens
├── test_packing_flux3_klein.py            # host: layout / bounds / ids digests vs the vendored reference
├── test_scheduler_flux3_klein.py          # host: 17-point schedule digest, flow-map reference time
├── test_rope_flux3_klein.py               # tables, interleave expansion, image id scale, negative reference axis
├── test_attention_flux3_klein.py          # one-sample full attention and windowed attention vs packed_attention
├── test_transformer_block_flux3_klein.py  # one block: broadcast modulation (early) and per-token modulation (single)
├── test_modulation_flux3_klein.py         # modulation table rows vs the reference heads; per-token expansion
├── test_transformer_flux3_klein.py        # whole DiT (mode_blocks / single_blocks / all_blocks ids), Tracy profile
├── test_vae_flux3_klein.py                # decode on vae/, ae.safetensors == vae/ tensor equality, encode PCC
├── test_text_encoder_flux3_klein.py       # 8-tap concat vs the HF Qwen3-VL text path, padded rows included
├── test_upsampler_flux3_klein.py          # prompt-upsampler smoke, system-prompt vendoring equality
├── test_pipeline_flux3_klein.py           # e2e T2I and 1-/2-reference edit; opt-in PSNR gate against goldens
├── test_performance_flux3_klein.py        # BenchmarkProfiler wall-clock, PERF_THRESHOLDS
├── run_flux3_klein_tests.sh               # local sweep runner
└── tools/                                 # not tests: Tracy harness, perf projection, sweep drivers
```

## Components

| component | reference (`reference/flux3_klein/`) | device implementation |
|---|---|---|
| DiT | `dit.py`: 3 per-mode stacks x 5 blocks (`image`, `image_cond`, `text`), then 28 shared single-stream blocks over the concatenated sequence; hidden 3072, 24 heads x 128; 5.55B parameters | `transformers/flux3_klein/`: every Klein block is tt_dit's `Flux2SingleTransformerBlock` after a load-time weight re-layout (the checkpoint stores q/k/v/attn_out/mlp_in/mlp_out as six separate linears; they are fused into `to_qkv`, `proj_mlp` and one `proj_out` at load) |
| token order | `MODES = ("image", "image_cond", "text")`: the order tokens take inside a sample, and the order of the single-stream sequence | `[image \| image_cond \| text]`, each segment tile-aligned; image rows lead so the final layer reads a prefix, as Flux2's `[spatial \| prompt]` does |
| modulation | per-mode `early_stream_modulations` / `single_stream_modulations` heads on `silu(vec)`, per token in the single stage; `vec` depends only on `(t, t_next)` | one merged chunked linear evaluated on two rows (`vec_t` for image, `vec_0` for cond/text) with `+1` folded into the scale bias; for the fixed 16-step schedule the whole table is built once per pipeline and the step reads a slot |
| RoPE | 4 axes x 32 dims, theta 10000; image ids scaled by 0.5, references on a negative axis, text on axis 3 | host cos/sin tables `(1, 1, seq, 128)`, SP-sharded, consumed by the fused RMSNorm+RoPE op |
| text encoder | `text_encoder.py`: Qwen3-VL-4B, hidden states of 8 layers concatenated layer-major to a 20480-wide token, length padded to a multiple of 80 (max 4096) | `encoders/qwen3vl/` with `loader_flux3_klein.py` (32 of 36 decoder layers, taps at layers 3, 7, ..., 31) and `pipelines/flux3_klein/prompt_encoder.py` |
| VAE | `autoencoder.py`: the FLUX.2 KL autoencoder (BFL `ae.safetensors` layout), 2x2 patchify to 128-channel tokens | decode: `vae/vae_flux2.py::Flux2VaeDecoder` unchanged on the snapshot's `vae/` (diffusers layout; `ae.safetensors == vae/` tensor by tensor) through `pipelines/flux3_klein/vae_flux3_klein.py` (SP gather of the fp32 state, alignment rows sliced, latent rows re-partitioned over the VAE's H axis; 4x8: channels TP4 over axis 0, H8 over axis 1); encode for references (M6b): device encoder specified in `docs/M6_M7_DESIGN.md` section 4, built after M7 |
| pipeline | `pipeline.py::Pipeline.generate` | `pipelines/flux3_klein/pipeline_flux3_klein.py`: `Flux3KleinRequest` (BFL field names) -> `policy.admit` (host) -> device text encoder -> traced denoise (`Flux3KleinDenoiser.denoise_on_device`) -> device VAE decode -> the reference `to_pil`; both DiTs, the text encoder and the VAE resident on the 4x8 ring |
| sampler | `sampling.py`: `t = alpha*t / (1 + (alpha-1)*t)` over `linspace(1, 0, steps+1)`; flow-map step with `timesteps_ref = schedule[i+1]`; fp32 latent state | `pipelines/flux3_klein/scheduler.py`; the fp32 Euler update runs on device inside the traced step |
| prompt upsampler | `flux-3-image-klein-pup` (Qwen3.5-9B fine-tune, Apache-2.0); greedy with presence penalty | `pipelines/flux3_klein/upsampler.py` wrapping `models/demos/blackhole/qwen36` on a Galaxy submesh |

## Precision contract

What the device port must match, and how closely. Thresholds are Flux2's; the "worst plausible bug" each
threshold would still catch is noted next to it in the test.

| stage | reference precision | device precision | gate |
|---|---|---|---|
| DiT linears | bf16 autocast | bf16 weights and activations, HiFi2 matmuls with fp32 accumulation (the `tt_dit` default) | per block `pcc >= 0.999, relative_rmse <= 0.032`; whole transformer `pcc >= 0.996, relative_rmse <= 0.09` (`utils/check.py::assert_quality`) |
| QK RMSNorm + RoPE | fp32 (autocast disabled), float64-derived tables | fused RMSNorm+RoPE op with fp32 internals | covered by the block and transformer gates; 16-step drift measured at M4 |
| timestep embedding and modulation | fp32 MLPs on bf16-rounded timesteps | fp32 path (`Timesteps` + `TimestepEmbedding`), the bf16 timestep rounding replicated on host | `test_modulation_flux3_klein.py` rows vs the reference heads |
| latent state | fp32 state updated by bf16-valued velocities | fp32 state tensor updated on device inside the trace | per-step latent PCC vs the GPU golden, `>= 0.99` at step 16 with recorded text embeddings (measured 0.9930 / rel_rmse 0.118 on the CPU golden); traced == eager. With the DEVICE text the distilled trajectory's proper reference is the CPU sampler run on the same text (`test_pipeline_flux3_klein.py::cached_reference_sampling`): measured 0.9855 / 0.170 / 29.99 dB for the 1024 man_dog case, below the 0.99 / 0.15 band although the per-step velocity floor is identical (0.99862 vs 0.99864), so that row is carried as a strict xfail until the B300 render decides it |
| text encoder | bf16 Qwen3-VL, 8 taps | `Qwen3VlTextEncoder`, 32 layers | per concatenated embedding `pcc 0.9997 / rmse 0.04`, per tap `>= 0.99` |
| VAE | fp32 (`ae.safetensors` fp32 weights, fp32 BatchNorm statistics, fp32 tokens) | `Flux2VaeDecoder` as shipped (`models/vae/vae.py`, unchanged): bf16 weights and activations; the fp32 state is typecast to bf16 on device BEFORE the inverse BatchNorm, whose `running_mean` / `running_var` are held in bf16 and applied in bf16 (`vae_flux2.py:94-118`; the one-hot probe lands one bf16 ulp off the fp32 formula; inverse-normalizing in fp32 first measured +3e-6 pcc: 0.999955 vs 0.999952, `logs/m67_designer_vae_decode_4x8.log:865,871`); convolutions and GroupNorm HiFi4 with fp32 accumulation on Blackhole (`vae.py:224`, `layers/normalization.py:100-102,175-177`); mid-block attention: SDPA HiFi2 fp32-acc `exp_approx_mode=False`, the QKV / out matmuls HiFi2 fp32-acc with `packer_l1_acc=True` and `math_approx_mode=True` (`vae.py:479-492,557,583`; approx mode changes nothing for a matmul: no SFPU transcendental in the op; it is the only approx flag not set False on the Klein path). No bf8 / fp16 anywhere | decode `pcc 0.9978 / rmse 0.034` (measured 0.999952 / 0.0113 base, 0.999954 / 0.0110 distilled on the golden tokens, PSNR 51.3 / 52.0 dB vs the CPU decode; `PSNR_DEC_FLOOR` 50.0); traced == eager bit-equal and a captured trace replayed on other tokens == their eager decode; encode `pcc > 0.999` (M6b, not built) |
| PIL conversion | `to_pil`: `((x.clamp(-1, 1) + 1) * 127.5).to(uint8)`, truncation toward zero | the same expression on the host (not `utils/tensor.float_to_uint8`, a different rounding path) | PSNR compares like with like |
| rendered image | GPU bf16 render of the same seed and recorded prompt | full device pipeline | PSNR against our own GPU render (proposed `>= 30 dB`, calibrated by the GPU-render-vs-shipped-PNG band); the shipped `examples/` PNGs are the calibration band, not the gate |

Everything in the baseline is exact with respect to this contract. Changes that are faster but not bit-identical
(bf8 operands with fp32 accumulation, the strided all-gather matmul, streaming SDPA) sit in a decision bucket and
are gated by the golden set before they become defaults; see the status board in the bringup project.

## Getting the weights

Both DiT checkpoints and the prompt upsampler are Hugging Face repositories:

| repo | contents | license |
|---|---|---|
| `black-forest-labs/flux-3-image-klein` | `model.safetensors` (DiT, bf16, 359 tensors), `ae.safetensors`, `text_encoder/`, `transformer/`, `vae/`, `variants/` | FLUX Kommunity License |
| `black-forest-labs/flux-3-image-klein-base` | as above, 357 DiT tensors (no `ref_time_in`) | FLUX Kommunity License |
| `black-forest-labs/flux-3-image-klein-pup` | Qwen3.5-9B fine-tune, `system_prompts.json` | Apache-2.0 |

`resolve_weights_dir` (`pipelines/flux3_klein/weights_flux3_klein.py`) resolves a snapshot in this order, the same
order and the same opt-in as the MiniMax-H3 and lightx2v loaders:

1. `FLUX3_KLEIN_MODEL_PATH`, when set, wins. It may point at the snapshot directory itself (the one holding
   `model.safetensors`, `ae.safetensors`, `text_encoder/`, `vae/`) or at a root holding it as a
   `flux-3-image-klein/` (repository name) or `distilled/` (variant name) child; `-base` and `-pup` resolve the same
   way.
2. Otherwise the Hugging Face cache, `<hub>/models--black-forest-labs--flux-3-image-klein/snapshots/<rev>/`, where
   `<hub>` is `$HF_HUB_CACHE` if set, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub` (the `refs/main` revision
   first, then the newest complete snapshot).
3. Otherwise a download, only with `TT_DIT_ALLOW_HF_DOWNLOAD=1`. With the variable unset a missing snapshot skips
   the test rather than pulling the repository mid-run.

```bash
export FLUX3_KLEIN_MODEL_PATH=/path/to/flux-3-image-klein    # or HF_HUB_CACHE=/path/to/hub (HF_HOME=/path/to/hf), or TT_DIT_ALLOW_HF_DOWNLOAD=1
export TT_DIT_CACHE_DIR=/your/cache/path                     # prepared device weights; set it, the fallback re-reads every run
export FLUX3_KLEIN_GOLDENS=/path/to/gpu_traces/flux3_klein   # optional: enables the PSNR / per-step gates
```

The weights and the prepared `.tensorbin` caches derived from them never enter git.

## How to Run

```bash
# [Install tt-metal](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

export FLUX3_KLEIN_MODEL_PATH=/path/to/flux-3-image-klein   # or let the HuggingFace cache resolve base + distilled
export TT_DIT_CACHE_DIR=/your/cache/path
export FLUX3_KLEIN_GOLDENS=/path/to/goldens                 # record.json + golden.png per case (common.py layout)
export FLUX3_KLEIN_OUT=/path/for/renders                    # PNGs land here, never inside the tree
export FLUX3_KLEIN_VAE_ORACLE_CACHE=/path/for/cpu/decodes     # optional: CPU reference decodes (~11 s each on a node CPU)
export FLUX3_KLEIN_SAMPLING_ORACLE_CACHE=/path/for/cpu/samples  # optional: CPU reference sampling on device text (~9 min)

# Text-to-image with the device text encoder, both checkpoints resident, Galaxy 4x8 (SP4 x TP8, ring).
# Expected: base passes; distilled is a strict XFAIL against the CPU reference on the same device text (precision
# contract, latent-state row) until the B300 render exists.
pytest models/tt_dit/tests/models/flux3_klein/test_pipeline_flux3_klein.py -k "t2i_device_text and bh_4x8_ring" --timeout=4800

# The same request with the recorded text (bit-identical to the M4 traced sampler)
pytest models/tt_dit/tests/models/flux3_klein/test_pipeline_flux3_klein.py -k "t2i_golden_text and bh_4x8_ring" --timeout=4800

# VAE decode on device against the CPU reference decode (test_decode_padded_size_full is a strict XFAIL:
# the shipped decoder's group-norm core grid rejects the 1344x768 per-device shape; non-1024 sizes are M8/M10)
pytest models/tt_dit/tests/models/flux3_klein/test_vae_flux3_klein.py -k bh_4x8_ring --timeout=2400

# Edit with one reference image
pytest models/tt_dit/tests/models/flux3_klein/test_pipeline_flux3_klein.py -k "<TBD: 1ref selector>"

# Whole transformer against the torch reference
pytest models/tt_dit/tests/models/flux3_klein/test_transformer_flux3_klein.py -k "<TBD>"

# Host-only gates (no device): packing, scheduler, vendored reference
pytest models/tt_dit/tests/models/flux3_klein/test_packing_flux3_klein.py \
       models/tt_dit/tests/models/flux3_klein/test_scheduler_flux3_klein.py
```

Rendered images are saved under `$FLUX3_KLEIN_OUT` as `m7_<mesh>_<model>_<w>x<h>_seed<seed>_<label>.png`.

## Performance

Blackhole Galaxy 4x8, SP4 (axis 0) x TP8 (axis 1), ring topology, 2 links, 8 KB fabric packets, traced, warm.
Seconds per image, best of 3 / median; batch 1; base checkpoint (50 steps, CFG 3.5) unless the row says otherwise. The
GPU column is the vendored reference in eager bf16 on the GPU box, measured by the same stage split
(`text_encode`, `vae_encode`, `denoise`, `vae_decode`, `device_to_host`). Image tokens are 2x2 latent patches of a
16x-downsampled canvas, so a 1024x1024 image is 4096 tokens and each 1024x1024 reference adds 4096.

| config | image tokens | text tokens | Galaxy 4x8 total | denoise | GPU reference total | image vs reference |
|---|---|---|---|---|---|---|
| 1024x1024 T2I | 2 x 4096 (CFG pair) | per prompt (multiple of 80) | TBD | TBD | TBD | TBD |
| 1024x1024 + 1 reference (1024x1024) | 4096 + 4096 | | TBD | TBD | TBD | TBD |
| 1024x1024 + 2 references (1024x1024 each) | 4096 + 8192 | | TBD | TBD | TBD | TBD |
| 2048x2048 T2I (4 MP) | 16384 | | TBD | TBD | TBD | TBD |
| 4096x2304 T2I | 36864 | | TBD | TBD | TBD | TBD |
| 4096x4096 T2I | 65536 | | TBD | TBD | TBD | TBD |
| 1024x1024 T2I, distilled checkpoint (16 steps, no CFG) | 4096 | | TBD | TBD | TBD | TBD |

Vendor anchor for the GPU column: about 3 s per 1 MP image on one H200 (upstream README).

Per-op Tracy breakdown of one forward step: TBD (lands with the perf milestone).

## Scalability

The DiT is parallelized on the two `tt_dit` axes, `sp` (sequence parallel; ring attention) and `tp` (tensor
parallel), configured as `((sp_factor, sp_axis), (tp_factor, tp_axis))`. Every gate is passed on the full 4x8 at
`((4, 0), (8, 1))`; unit tests may run on Galaxy submeshes (for example `(1, 8)` or `(2, 2)`, line topology) when
that is faster to iterate. At TP8 each device holds 3 of the 24 heads and a 384-wide hidden shard. Data-parallel
groups on submeshes are the throughput path once latency is settled. The text encoder and the VAE run with tensor
parallelism. No Wormhole target.

## License

- **Code.** The upstream inference code is Apache-2.0 (`Copyright 2026 Black Forest Labs, Inc.`); the vendored
  reference under `models/tt_dit/reference/flux3_klein/` keeps that notice and marks every modified file. The port
  itself is Apache-2.0 like the rest of tt-metal.
- **DiT weights.** `flux-3-image-klein` and `flux-3-image-klein-base` are released under the
  [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md). They are
  never checked in and are not redistributed from this repository.
- **Prompt upsampler weights.** `flux-3-image-klein-pup` is Apache-2.0.

See `models/tt_dit/reference/flux3_klein/README.md` for the per-file provenance of the vendored reference.
