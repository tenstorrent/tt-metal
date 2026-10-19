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
│   └── vae/                               # vae_flux2.py reused unchanged for decode; encoder_flux2.py (M8) is the reference encoder
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
| VAE | `autoencoder.py`: the FLUX.2 KL autoencoder (BFL `ae.safetensors` layout), 2x2 patchify to 128-channel tokens | decode: `vae/vae_flux2.py::Flux2VaeDecoder` unchanged on the snapshot's `vae/` (diffusers layout; `ae.safetensors == vae/` tensor by tensor) through `pipelines/flux3_klein/vae_flux3_klein.py` (SP gather of the fp32 state, alignment rows sliced, latent rows re-partitioned over the VAE's H axis; 4x8: channels TP4 over axis 0, H8 over axis 1; a canvas whose latent height is not a multiple of 8, e.g. 896x1120, takes the replicated-rows TP4 decoder); encode for references: `vae/encoder_flux2.py::Flux2VaeEncoder` (the decoder's blocks plus the strided `VaeDownsample2d`; channels TP4 over axis 0, spatial replicated) through `pipelines/flux3_klein/vae_encode_flux3_klein.py`, eager, one call per prepared reference |
| pipeline | `pipeline.py::Pipeline.generate`, `generate.py:65-77` | `pipelines/flux3_klein/pipeline_flux3_klein.py`: `Flux3KleinRequest` (BFL field names, `images` up to 10) -> `references.prepare_references` (host sizing) -> `policy.admit` (host) -> device text encoder -> device VAE encode of the references -> traced denoise (`Flux3KleinDenoiser.denoise_on_device`) -> device VAE decode -> the reference `to_pil` and the single-reference crop-back; both DiTs, the text encoder, the VAE decoder and the VAE encoder resident on the 4x8 ring |
| references | `pipeline.py:129-180`: `reference_canvas` / `prepare_reference` / `prepare_multi_reference` / `default_size`; `generate.py:65-77` | `pipelines/flux3_klein/references.py`, the reference's expressions verbatim (see "References (edit)" below); bit-exact against the vendored functions in `test_references_flux3_klein.py` |
| sampler | `sampling.py`: `t = alpha*t / (1 + (alpha-1)*t)` over `linspace(1, 0, steps+1)`; flow-map step with `timesteps_ref = schedule[i+1]`; fp32 latent state | `pipelines/flux3_klein/scheduler.py`; the fp32 Euler update runs on device inside the traced step |
| prompt upsampler | `flux-3-image-klein-pup` (Qwen3.5-9B fine-tune, Apache-2.0); greedy with presence penalty | `pipelines/flux3_klein/upsampler.py` wrapping `models/demos/blackhole/qwen36` on a Galaxy submesh |

## References (edit)

The reference sizing is BFL's, expression for expression (`pipelines/flux3_klein/references.py`; every function is
held bit-exact against the vendored copy on the example inputs and ten synthetic sizes):

* **One reference** (`reference_canvas` / `prepare_reference`): fitted under 4 MiP without upscaling (`scale = min(1,
  sqrt(4 MiP / (w h)), 4096 / max(w, h))`, retried at `x 0.995` until the 16-px canvas fits), resized with LANCZOS when
  the fit differs, then centred on the canvas (the fitted size rounded up to 16, clamped to [64, 4096]) with a reflect
  pad (`symmetric` for a side <= 16 px). The fitted box is the **crop-back**: when the request gives neither `width`
  nor `height`, the rendered canvas is cropped to it (`generate.py:76-77`), so a 1016x1016 reference renders on a
  1024x1024 canvas and returns a 1016x1016 image.
* **Several references** (`prepare_multi_reference`): each fitted under 1 MiP (sides floored), centre-cropped to the
  16-px grid; no padding, no crop-back.
* **Canvas**: `width or default_w, height or default_h` with `default_size` = the first PREPARED reference's size
  (1024x1024 for text-to-image). The example cases render at 1024x1024 (bandana, mugprint) and 896x1120 (figurine,
  latent 56x70: the replicated-rows decoder).
* **Tokens**: each reference is encoded on device (`Flux2VaeEncoder`) to `(H/16)(W/16)` rows of 128 channels; the ids
  are `packing.reference_ids(h, w, i)` (image ids scaled by 0.5, `l = -(1 + i)`); the rows are shared by the two CFG
  samples of the base model.
* **Admission** (`policy.py`, host, before any device work): sides multiples of 16 in [64, 4096]; at most 10
  references; every VAE level of each prepared reference and of the canvas tile-aligned (`H W` a multiple of 32 at the
  full, /2, /4 and /8 resolutions: the plain `ttnn.group_norm` reduces over whole tiles; a 1008x1008 canvas, /8 level
  15876 rows, is refused with `budget="reference_size"` / `"canvas_size"`); text <= 4096 tokens. There is no token
  budget (removed 2026-10-19): every structurally valid request goes to the device, up to the worst case 4096x4096 +
  10 references + 4096 text = 110,592 tokens per sample, and a device allocation failure surfaces as the ttnn
  exception. A refused request has allocated nothing on the device and makes zero calls into any device entry point
  (`test_admission_before_device`, `DeviceCallCounter`). Footprints measured so far: T2I 1024 3.41 GB/chip resident,
  1 ref 1024 (8512 tokens) 4.11 GB/chip, 2 refs 1024 (12608 tokens) +9 MB/bank over that (`policy.CAPACITY` comment).
* **Which inputs are accepted**: the tile rule admits a prepared reference or canvas only when `(H/16)(W/16)` is a
  multiple of 8, i.e. the /8 VAE level has a whole number of 32-row tiles. It accepts 1024x1024, 896x1120, 1344x768,
  768x1344, 512x512, 2048x2048 and 64x64, but REFUSES many sizes BFL's sizing produces for ordinary photos: a single
  1000x1000 reference pads to a 1008x1008 canvas (refused: 126 x 126 = 15876 rows), and the multi-reference crops of a
  1001x999 or 2000x1200 photo (992x992, 1312x784) have /8 levels of 15376 / 16072 rows. Resize such references to an
  admitted size first (1024x1024, 1344x768 ...) or pass an admitted `width` / `height`; lifting the rule (a GroupNorm
  that masks the padding rows of a partial tile) is an M10 item. `width=0` / `height=0` mean "not given", as in BFL's
  API (`generate.py:72`).

## Precision contract

What the device port must match, and how closely. Thresholds are Flux2's; the "worst plausible bug" each
threshold would still catch is noted next to it in the test.

| stage | reference precision | device precision | gate |
|---|---|---|---|
| DiT linears | bf16 autocast | bf16 weights and activations, HiFi2 matmuls with fp32 accumulation (the `tt_dit` default) | per block `pcc >= 0.999, relative_rmse <= 0.032`; whole transformer `pcc >= 0.996, relative_rmse <= 0.09` (`utils/check.py::assert_quality`) |
| QK RMSNorm + RoPE | fp32 (autocast disabled), float64-derived tables | fused RMSNorm+RoPE op with fp32 internals | covered by the block and transformer gates; 16-step drift measured at M4 |
| timestep embedding and modulation | fp32 MLPs on bf16-rounded timesteps | fp32 path (`Timesteps` + `TimestepEmbedding`), the bf16 timestep rounding replicated on host | `test_modulation_flux3_klein.py` rows vs the reference heads |
| latent state | fp32 state updated by bf16-valued velocities | fp32 state tensor updated on device inside the trace | per-step latent PCC vs the GPU golden, `>= 0.99` at step 16 with recorded text embeddings (measured 0.9930 / rel_rmse 0.118 on the CPU golden); traced == eager. Reference (edit) rows use the same method (`test_pipeline_flux3_klein.py::test_edit_golden_inputs`): recorded text + recorded reference tokens -> bit-equal to the M4 traced sampler, teacher-forced per-step velocity floor 0.998, final band 0.99 / 0.15 (distilled) and 0.97 / 0.25 (base 50). With the DEVICE text the distilled trajectory's proper reference is the CPU sampler run on the same text (`test_pipeline_flux3_klein.py::cached_reference_sampling`): measured 0.9855 / 0.170 / 29.99 dB for the 1024 man_dog case, below the 0.99 / 0.15 band although the per-step velocity floor is identical (0.99862 vs 0.99864), so that row is carried as a strict xfail until the B300 render decides it |
| text encoder | bf16 Qwen3-VL, 8 taps | `Qwen3VlTextEncoder`, 32 layers | per concatenated embedding `pcc 0.9997 / rmse 0.04`, per tap `>= 0.99` |
| VAE decode | fp32 (`ae.safetensors` fp32 weights, fp32 BatchNorm statistics, fp32 tokens) | `Flux2VaeDecoder` as shipped (`models/vae/vae.py`, unchanged): bf16 weights and activations; the fp32 state is typecast to bf16 on device BEFORE the inverse BatchNorm, whose `running_mean` / `running_var` are held in bf16 and applied in bf16 (`vae_flux2.py:94-118`; the one-hot probe lands one bf16 ulp off the fp32 formula; inverse-normalizing in fp32 first measured +3e-6 pcc: 0.999955 vs 0.999952, `logs/m67_designer_vae_decode_4x8.log:865,871`); convolutions and GroupNorm HiFi4 with fp32 accumulation on Blackhole (`vae.py:224`, `layers/normalization.py:100-102,175-177`); mid-block attention: SDPA HiFi2 fp32-acc `exp_approx_mode=False`, the QKV / out matmuls HiFi2 fp32-acc with `packer_l1_acc=True` and `math_approx_mode=True` (`vae.py:479-492,557,583`; approx mode changes nothing for a matmul: no SFPU transcendental in the op; it is the only approx flag not set False on the Klein path). No bf8 / fp16 anywhere | decode `pcc 0.9978 / rmse 0.034` (measured 0.999952 / 0.0113 base, 0.999954 / 0.0110 distilled on the golden tokens, PSNR 51.3 / 52.0 dB vs the CPU decode; `PSNR_DEC_FLOOR` 50.0); traced == eager bit-equal and a captured trace replayed on other tokens == their eager decode; 896x1120 (replicated decoder + the GroupNorm grid fallback) and 1344x768 (H8 decoder, the attention norm on the fallback grid) at the same pcc / rmse gate with their own PSNR floors (896x1120: `PSNR_DEC_FLOOR_896x1120`, frozen at the first measurement) |
| VAE encode (references) | fp32 encoder, fp32 BatchNorm statistics, tokens cast to bf16 (`pipeline.py:93`) | `Flux2VaeEncoder` (`models/vae/encoder_flux2.py`, built from the shipped `vae.py` blocks): bf16 weights and activations, the input image quantized to bf16 (`uint8 / 127.5 - 1` rounded to 8 mantissa bits), convolutions (incl. the stride-2 `padding=(0, 1, 0, 1)` downsample = `F.pad(x, (0, 1, 0, 1))` + stride-2 conv) and GroupNorms HiFi4 fp32-accumulated on Blackhole, the mid-block attention as the decoder's; the BatchNorm normalize `(z - mean) * inv_std` in fp32 on device with `inv_std = 1 / sqrt(var + 1e-4)` computed in fp32 on the host, then one bf16 cast of the tokens (the reference's rounding structure). No bf8 / fp16 | encode `pcc > 0.999` vs the CPU reference `AutoEncoder.encode` of the same prepared image (PLAN M6), `rel_rmse <= ENCODE_RMSE` (frozen at the first measurement, `test_vae_flux3_klein.py`); ids exact; second encode bit-equal; traced == eager and a trace replayed on another image == its eager encode. Device measurement pending: the first device session hung at its first op (wedged b03u02); on a healthy mesh (b04u02) the shipped conv slice table then threw at the level-2 downsample (`TT_FATAL ... slices (8) exceeds the maximum allowed (4)`, fixed by capping every Klein slice entry at ttnn's `ceil(output_width / 32)`, `vae_flux3_klein.klein_conv_slices` / `VaeDownsample2d`), and b04u02 wedged (PCIe 25) before the capped encoder could be measured; `ENCODE_RMSE` is `None` = log only until the first measurement. The downsample conv alone measured pcc 0.999995 / rel_rmse 0.0034 vs torch (`DOWNSAMPLE_PCC` 0.9999) |
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

# Command-line demo (flags as upstream scripts/cli.py). The prompt must already be an upsampled caption until the
# prompt upsampler is on device; --record replays a shipped example (prompt, model, seed, references from its json).
python models/tt_dit/pipelines/flux3_klein/demo_flux3_klein.py "<upsampled caption>" --model base --seed 7 --out out/t2i.png
python models/tt_dit/pipelines/flux3_klein/demo_flux3_klein.py "<upsampled caption>" --model distilled \
    --image dog.png --image mug.png --out out/edit.png                           # up to 10 references
python models/tt_dit/pipelines/flux3_klein/demo_flux3_klein.py --record reference/examples/distilled/3_bandana.json --out out/bandana.png
# A structurally invalid request (side not a multiple of 16 or over 4096, > 10 references, a size whose VAE levels are
# not tile-aligned) is refused on the host (exit 3, "REFUSED: ... budget ...") before any device call; no token budget.

# Text-to-image with the device text encoder, both checkpoints resident, Galaxy 4x8 (SP4 x TP8, ring).
# Expected: base passes; distilled is a strict XFAIL against the CPU reference on the same device text (precision
# contract, latent-state row) until the B300 render exists.
pytest models/tt_dit/tests/models/flux3_klein/test_pipeline_flux3_klein.py -k "t2i_device_text and bh_4x8_ring" --timeout=4800

# The same request with the recorded text (bit-identical to the M4 traced sampler)
pytest models/tt_dit/tests/models/flux3_klein/test_pipeline_flux3_klein.py -k "t2i_golden_text and bh_4x8_ring" --timeout=4800

# VAE decode on device against the CPU reference decode, incl. the 1344x768 canvas (the GroupNorm grid fallback,
# measured pcc 0.999954 / 53.36 dB) and the 896x1120 figurine canvas on the replicated-rows decoder
pytest models/tt_dit/tests/models/flux3_klein/test_vae_flux3_klein.py -k bh_4x8_ring --timeout=3600

# The trace-allocation contract: two generates (capture, then replay) under the tracker must log 0 "still alive" lines
# (the VAE decode trace of a canvas size is captured BEFORE the DiT trace of a layout; pipeline docstring)
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 pytest \
    models/tt_dit/tests/models/flux3_klein/test_pipeline_flux3_klein.py -k "trace_alloc_tracking and bh_4x8_ring" --timeout=4800

# Edit with references (the goldens' 1- and 2-reference cases and the 896x1120 figurine): recorded text + recorded
# reference tokens (bit-comparable to the golden path), then the full device path (device text + device VAE encode);
# a request over the measured capacity is refused on the host before any device allocation. Expected: the 1024
# cases pass (final pcc 0.999747 / 0.999552 vs the goldens through the M4 denoise method); distilled-1_figurine is a
# strict XFAIL at the unchanged band (0.976069 / 0.2189 with the per-step floor met: the first padded image_cond
# segment, open numerics item, test_denoise_flux3_klein.FINAL_BAND_XFAIL_CASES)
pytest models/tt_dit/tests/models/flux3_klein/test_pipeline_flux3_klein.py -k "edit_golden_inputs and bh_4x8_ring" --timeout=9000
pytest models/tt_dit/tests/models/flux3_klein/test_pipeline_flux3_klein.py -k "edit_device_path and bh_4x8_ring" --timeout=9000
pytest models/tt_dit/tests/models/flux3_klein/test_pipeline_flux3_klein.py -k "(edit_crop_back or admission_before_device) and bh_4x8_ring" --timeout=4800

# The device VAE encoder against the CPU reference encode of the example inputs, and the 896x1120 decode
pytest models/tt_dit/tests/models/flux3_klein/test_vae_flux3_klein.py -k "encode_reference and bh_4x8_ring" --timeout=3600
pytest models/tt_dit/tests/models/flux3_klein/test_vae_flux3_klein.py -k "decode_896x1120 and bh_4x8_ring" --timeout=3600

# Whole transformer against the torch reference (T2I, one and two references, the CFG pair)
pytest models/tt_dit/tests/models/flux3_klein/test_transformer_flux3_klein.py -k "base_real and bh_4x8_ring" --timeout=6000

# Host-only gates (no device): packing, scheduler, vendored reference, reference sizing + admission policy, the
# Klein conv DRAM slice table against ttnn's ceil(output_width / 32) bound, the GroupNorm grid rule
pytest models/tt_dit/tests/models/flux3_klein/test_packing_flux3_klein.py \
       models/tt_dit/tests/models/flux3_klein/test_scheduler_flux3_klein.py \
       models/tt_dit/tests/models/flux3_klein/test_references_flux3_klein.py \
       models/tt_dit/tests/models/flux3_klein/test_vae_flux3_klein.py -k "not bh_4x8_ring and not 1x1"
```

The reference images of the example cases (`examples/inputs/{1,2,3}.png`) are not vendored; the tests find them in
the BFL clone named by `FLUX3_KLEIN_REFERENCE` (`$FLUX3_KLEIN_REFERENCE/examples/inputs/`) or next to a golden case
(the golden generator copies every reference beside `record.json`).

Rendered images are saved under `$FLUX3_KLEIN_OUT` as `m7_<mesh>_<model>_<w>x<h>_seed<seed>_<label>.png` (labels
`golden_text`, `device_text`, `golden_inputs`, `device_path`, `crop_back`).

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
