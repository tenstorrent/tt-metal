# FLUX 3 Image Klein on a Blackhole Galaxy

## The model

FLUX 3 Image Klein is Black Forest Labs' DiT for text-to-image generation and reference-image editing: a canvas of
64 to 4096 px per side (multiples of 16) conditioned on a caption and on up to ten reference images. The port runs
four models on device. The DiT (`model.safetensors`, 5.55B parameters, hidden 3072, 24 heads x 128) has three
per-mode early stacks of 5 blocks each (`image`, `image_cond`, `text`) followed by 28 shared single-stream blocks over
the joint sequence `[image | image_cond | text]`. The text encoder is Qwen3-VL-4B: 32 of its 36 decoder layers, eight
layer taps concatenated into a 20480-wide token, caption length rounded up to a multiple of 80 (at most 4096 tokens).
The FLUX.2 KL autoencoder (8x VAE plus 2x2 patchify = 16x, 128-channel latent tokens) decodes the result and encodes
the references. The prompt upsampler (a Qwen3.5-9B fine-tune) expands a raw prompt into the long caption the DiT was
trained on and runs on one chip in its own process. Two DiT checkpoints share the architecture: `distilled` (a flow
map, 16 Euler steps, no guidance; the schedule `t = 6.93 u / (1 + 5.93 u)` over `u = linspace(1, 0, steps + 1)`) and
`base` (50 steps, classifier-free guidance 3.5, two forwards per step).

The target is the Blackhole Galaxy as one 4x8 mesh (32 chips) on a 1D ring fabric with 8 KB packets. The default DiT
layout is SP8 x TP4: the sequence (token rows) is sharded 8 ways over mesh axis 1 (the eight columns) and every
linear's hidden dimension 4 ways over mesh axis 0 (the four rows). `sp4tp8` is the transposed assignment. The text
encoder is tensor-parallel over the DiT's TP axis and FSDP over the other; the VAE runs channels TP4 over axis 0 and
latent rows H8 over axis 1 regardless of the DiT layout.

## Weights

| variant     | Hugging Face repo                             | files the port reads                                     |
|-------------|-----------------------------------------------|----------------------------------------------------------|
| `distilled` | `black-forest-labs/flux-3-image-klein`        | `model.safetensors` (DiT), `vae/` (decoder + encoder), `text_encoder/` (Qwen3-VL) |
| `base`      | `black-forest-labs/flux-3-image-klein-base`   | the same layout (357 DiT tensors: no `ref_time_in`)      |
| `pup`       | `black-forest-labs/flux-3-image-klein-pup`    | `model.safetensors`, `config.json`, `tokenizer.json`     |

`weights_flux3_klein.resolve_weights_dir(variant)` resolves a snapshot in this order: `$FLUX3_KLEIN_MODEL_PATH`
(the snapshot itself, or a root holding a `flux-3-image-klein[-base|-pup]/` or `distilled/`, `base/`, `pup/` child;
a path that names a different variant is passed over), then the Hugging Face cache
(`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`, `refs/main` first), then a download only
when `TT_DIT_ALLOW_HF_DOWNLOAD=1`. The demo's `--weights-base`, `--weights-distilled` and `--pup-weights` override the
resolver per variant. Prepared device weights are cached under `$TT_DIT_CACHE_DIR` by `utils/cache.py` (the DiT per
variant and layout, the text encoder and the VAE keyed by their Hugging Face blob, so base and distilled share them);
the upsampler's tensor cache goes to `$TT_CACHE_PATH`, defaulting to `$TT_DIT_CACHE_DIR/flux3_klein_pup`.

```bash
export FLUX3_KLEIN_MODEL_PATH=$HF_HUB_CACHE/models--black-forest-labs--flux-3-image-klein/snapshots/<rev>  # optional
export TT_DIT_CACHE_DIR=/path/to/tt_dit_cache
```

## Launch

The in-tree demo opens the 4x8 ring (`demo_flux3_klein.open_mesh`: `FABRIC_1D_RING`, 8 KB packets, L1_SMALL 64 KB,
a 256 MB trace region), makes one DiT variant resident and renders. The prompt is upsampled by default: the
upsampler runs first in a subprocess (`upsampler.Flux3KleinUpsamplerSubprocess`, its own fabric-less 1x1 mesh),
then the ring opens for the DiT. A prompt that is already a finished caption passes through; `--no-pup` skips the
upsampler. Without `--seed` the seed is `sha256(caption) % 2**32`, as upstream's CLI.

```bash
# text-to-image (distilled, 16 steps, 1024x1024)
python models/tt_dit/pipelines/flux3_klein/demo_flux3_klein.py "A man walking with his dog, afternoon sun" \
    --model distilled --seed 7 --out out/t2i.png
# editing with references (repeat --image, up to 10; the canvas defaults to the first prepared reference's size)
python models/tt_dit/pipelines/flux3_klein/demo_flux3_klein.py "put a red bandana around the dog's neck" \
    --model distilled --image dog.png --seed 7 --out out/edit.png
# a finished caption, no upsampler, explicit size and steps, base model with CFG
python models/tt_dit/pipelines/flux3_klein/demo_flux3_klein.py --prompt-file caption.txt --no-pup \
    --model base --width 1344 --height 768 --steps 50 --guidance 3.5 --out out/base.png
# replay a vendored example record (recorded caption, model, seed, references), or re-upsample its input prompt
python models/tt_dit/pipelines/flux3_klein/demo_flux3_klein.py \
    --record models/tt_dit/reference/flux3_klein/examples/distilled/man_dog.json --out out/man_dog.png
python models/tt_dit/pipelines/flux3_klein/demo_flux3_klein.py --record <record.json> --pup --out out/pup.png
```

Flags (`demo_flux3_klein.parse_args`): `prompt` (positional) or `--prompt-file`; `--record`; `--no-pup` / `--pup`;
`--boxes boxes.json` (0-1 layout boxes `[y_min, x_min, y_max, x_max]` with `id` / `desc`); `--pup-weights`;
`--model {base,distilled}`; `--image` (repeatable); `--width`, `--height`, `--steps`, `--guidance` (base only);
`--seed`; `--out` (default `out.png`; `.jpeg` selects JPEG; a sidecar `<out>.json` with the record fields is written
next to it); `--weights-base`, `--weights-distilled`; `--eager` (no tracing); `--cache F8B0:0.08:w8` (DBCache spec,
implies eager); `--layout {sp4tp8,sp8tp4,dual}` (default `sp8tp4`; `dual` keeps both resident and picks SP8 x TP4
from 16384 tokens up); `--repeat N` (the second render replays the captured traces). A structurally invalid request is
refused on the host before any device work and exits with code 3 (`REFUSED: <reason>`); argument errors exit 2.

The upsampler alone, on one chip:

```bash
python -m models.tt_dit.pipelines.flux3_klein.upsampler --prompt "A man walking with his dog" --out out/caption.json
python -m models.tt_dit.pipelines.flux3_klein.upsampler --prompt "put a red bandana on the dog" --image dog.png --out out/edit.json
```

(`--boxes`, `--weights`, `--no-trace`, `--max-new-tokens`, `--teacher-force <record.json>` as in its `parse_args`.)

### Tests

Tests live in `models/tt_dit/tests/models/flux3_klein/`. The host-only rows need no device (only the repository's Python environment):

```bash
pytest models/tt_dit/tests/models/flux3_klein/test_packing_flux3_klein.py \
       models/tt_dit/tests/models/flux3_klein/test_scheduler_flux3_klein.py \
       models/tt_dit/tests/models/flux3_klein/test_references_flux3_klein.py \
       models/tt_dit/tests/models/flux3_klein/test_rope_flux3_klein.py \
       models/tt_dit/tests/models/flux3_klein/test_layout_flux3_klein.py
pytest models/tt_dit/tests/models/flux3_klein/test_upsampler_flux3_klein.py -k "not bh_"
pytest models/tt_dit/tests/models/flux3_klein/test_pipeline_flux3_klein.py -k "upsampler_hook or request_resolve"
pytest models/tt_dit/tests/models/flux3_klein/test_vae_flux3_klein.py -k "not bh_4x8_ring and not 1x1"
```

Device rows parametrize `mesh_device, device_params` from `device_params.py` and self-select by id: `bh_4x8_ring`
(the Galaxy gate), `bh_glx_2x2sub_linear` (block, modulation, stacks, denoise and transformer rows on a 2x2
submesh), `bh_glx_1x4sub_linear` (text encoder), `bh_1x1_nofabric` (upsampler), `1x1` (VAE).

```bash
pytest models/tt_dit/tests/models/flux3_klein/test_pipeline_flux3_klein.py -k "t2i_device_text and bh_4x8_ring" --timeout=4800
pytest models/tt_dit/tests/models/flux3_klein/test_pipeline_flux3_klein.py -k "edit_device_path and bh_4x8_ring" --timeout=9000
pytest models/tt_dit/tests/models/flux3_klein/test_transformer_flux3_klein.py -k "base_real and bh_4x8_ring" --timeout=6000
pytest models/tt_dit/tests/models/flux3_klein/test_vae_flux3_klein.py -k bh_4x8_ring --timeout=3600
pytest models/tt_dit/tests/models/flux3_klein/test_upsampler_flux3_klein.py -k "bh_1x1_nofabric and caption_t2i" --timeout=7200
```

Goldens are produced by an out-of-tree GPU harness (`generate_goldens.py`) that replays the vendored example records
(`models/tt_dit/reference/flux3_klein/examples/<set>/<case>.json`) through the vendored torch reference and writes
one directory per case under `$FLUX3_KLEIN_GOLDENS/<set>/<case>/`: `record.json` (model, size, seed, token counts,
noise sha256), `inputs.safetensors` (text tokens and ids, reference tokens and ids, the noise, the schedule),
`steps.safetensors` (per step `x_t` and `velocity`, `final_tokens`, the decoded image) and `golden.png`; the layout is
documented in `tests/models/flux3_klein/common.py`. The gates compare: the vendored CPU DiT against the golden's per-step velocities
(`test_reference_flux3_klein.py`, PCC floor `FLUX3_KLEIN_GOLDEN_PCC`); the device sampler fed the golden's recorded
text and reference rows against its per-step velocities and final state (`test_denoise_flux3_klein.py`
`teacher_forced_floor` / `final_gate`); the device render against `golden.png` (`PSNR_INTERIM`) and against the CPU
decode of its own final tokens (`PSNR_DEC_FLOOR`, isolates the VAE); and, with the device text encoder, against a
CPU reference run on that same text (`PSNR_PROPER_REFERENCE`, cached under `$FLUX3_KLEIN_SAMPLING_ORACLE_CACHE`).
Renders go to `$FLUX3_KLEIN_OUT`, never into the tree. Test thresholds are never lowered: a row that misses its band
is carried as a strict `xfail` with its measurement.

## Code flow

### Module map

`models/tt_dit/pipelines/flux3_klein/`:

| file | role |
|---|---|
| `demo_flux3_klein.py` | CLI: upsampler subprocess, mesh open, `Flux3KleinPipeline`, sidecar json |
| `pipeline_flux3_klein.py` | `Flux3KleinRequest` / `Flux3KleinResolved` / `Flux3KleinResult`, `Flux3KleinPipeline` (residency, layouts, `generate`), `resolve_caption`, `make_noise`, `pick_layout`, DiT weight loading |
| `policy.py` | `admit`: the host admission checks and `Flux3KleinAdmissionError` |
| `references.py` | BFL's reference sizing (`prepare_references`, crop-back box), expression for expression |
| `prompt_encoder.py` | `Flux3KleinPromptEncoder`: tokenizer recipe, RoPE tables, the device Qwen3-VL call, 8-tap concat |
| `upsampler.py` | `build_pup_inputs`, `Flux3KleinUpsampler` (one chip), `Flux3KleinUpsamplerSubprocess`, the CLI |
| `packing.py` | pure-torch token contract: `latent_shape`, `image_ids`, `reference_ids`, `text_ids`, `padded_length`, `pad_rows` |
| `scheduler.py` | `Flux3KleinScheduler`: the Euler grid and the bf16 timestep roundings the DiT sees |
| `denoise_flux3_klein.py` | `Flux3KleinDenoiser`: slots, plans, the traced step (`_step`), `denoise_on_device` |
| `cache_flux3_klein.py` | `Flux3KleinCacheConfig` / `Flux3KleinCache`: DBCache over the single-block stack |
| `vae_encode_flux3_klein.py` | `Flux3KleinVaeEncode`: references -> `image_cond` rows through `models/vae/encoder_flux2.py` |
| `vae_flux3_klein.py` | `Flux3KleinVaeDecode`: state slot -> image through `models/vae/vae_flux2.py`; conv slice tables, exact GroupNorm |
| `weights_flux3_klein.py` | repo ids, snapshot resolution, `dit_checkpoint` / `autoencoder_checkpoint` |

Model code it imports: `models/tt_dit/models/transformers/flux3_klein/` (`transformer_flux3_klein.py`
`Flux3KleinTransformer` + `Flux3KleinSequencePlan`; `transformer_block_flux3_klein.py` the block composed from
`layers/` pieces; `stacks_flux3_klein.py` the early stacks and `assemble_single_stage`; `modulation_flux3_klein.py`
the time path and the 60-row modulation table; `rope_flux3_klein.py`; `layout_flux3_klein.py`
`Flux3KleinSequenceLayout`), `models/tt_dit/encoders/qwen3vl/` (`loader_flux3_klein.py` builds the 32-layer
tapped `Qwen3VlTextEncoder`), `models/tt_dit/models/vae/` (`vae_flux2.py`, `encoder_flux2.py`, unchanged) and the
vendored torch reference `models/tt_dit/reference/flux3_klein/` (tests only; `bbox.py` for the layout boxes).

### The request path (`Flux3KleinPipeline.generate`)

1. **Request -> resolved -> admitted.** `Flux3KleinRequest.resolve` fills the variant defaults (steps, guidance),
   validates boxes, prepares the references on the host (`references.prepare_references`: one reference fitted
   under 4 MP and reflect-padded to the 16-px grid with a crop-back box, several each under 1 MP centre-cropped) and
   picks the canvas. `resolve_caption` applies the upstream rule: a finished caption passes, else the attached
   upsampler runs (a refusal falls back to the prompt), else the box table is appended. The text length comes from
   `Flux3KleinPromptEncoder.tokenize`, the seed from `caption_seed`. `policy.admit` then checks sides, reference
   count, the VAE levels' tile alignment and the text length; a rejection raises `Flux3KleinAdmissionError` before
   anything touches the device.
2. **Prompt encoding.** `Flux3KleinPromptEncoder.encode_host(caption)` (the layout's wrapper of the resident
   encoder) returns `(L, 20480)` bf16 rows and `text_ids(L)`; the base model also uses the empty prompt's encoding,
   computed once at construction (`_Layout.null_text`).
3. **References -> `image_cond`.** `Flux3KleinVaeEncode.encode_references(prepared)` encodes each prepared image on
   device (`Flux2VaeEncoder`, eager, one call per reference) into bf16 `[h_i * w_i, 128]` tokens, concatenated with
   `packing.reference_ids(h, w, i)`; the two CFG samples share them.
4. **Noise, ids, layout, plan.** `make_noise(seed, h, w)` (CPU generator, bf16-valued fp32) and `image_ids(h, w)`;
   `pick_layout` chooses the DiT layout by row count; `Flux3KleinDenoiser.prepare` builds the
   `Flux3KleinSequenceLayout` of the sample, the `Flux3KleinSequencePlan` (`Flux3KleinTransformer.prepare_plan`:
   RoPE tables per mode and for the joint sequence, modulation gather indices, the assembly and output index maps),
   uploads the text and reference rows and hoists their input projections (`embed_text`, `embed_image_cond`).
5. **Denoise loop.** `Flux3KleinDenoiser.denoise_on_device` runs `step` for every schedule point: the host writes
   the step's scalars into fp32 slots (`t_embed`, `t_ref_embed`, `dt`, `guidance`), then `_step` (one ttnn trace)
   casts the fp32 state to bf16, runs `Flux3KleinTransformer.forward` (modulation table from the time path, early
   stacks per mode, `assemble_single_stage`, the 28 single blocks or `Flux3KleinCache.run` when a DBCache is
   attached, final layer), combines CFG (`combine_cfg`) and applies the Euler update in fp32 (`euler_update`); the
   new state is copied into the state slot.
6. **VAE decode.** `Flux3KleinVaeDecode.decode(state)` typecasts the slot to bf16, all-gathers it over SP, slices the
   alignment rows, re-partitions the latent rows over the VAE's H axis and runs `Flux2VaeDecoder` (traced; a canvas
   whose latent height is not a multiple of 8 takes a replicated-rows fallback decoder).
7. **Host image.** `image_to_uint8_on_device` and one replica read (`readback = "fast"`), the single-reference
   crop-back, and the sidecar fields on `Flux3KleinResult` (`image`, `seed`, `caption`, `timings`, `final_tokens`).

### Layouts, slots and caches

`Flux3KleinSequenceLayout` (`layout_flux3_klein.py`) is the host bookkeeping of one sample at SP factor `sp`: per mode
`L_m` real rows, `padded[m] = padded_length(L_m, sp)` (a multiple of `32 * sp`, zero alignment rows at the end),
`rows[m] = padded[m] / sp` tile-aligned rows per device, and the joint order `G = [image | image_cond | text | pad]`
with `total_padded = sum(rows) * sp`. Activations are `[1, L_pad, 3072]` with `mesh_axes=[None, sp_axis, tp_axis]`:
rows over the SP axis, the hidden dimension over the TP axis; token inputs and the state are `[None, sp_axis, None]`.
`Flux3KleinPipeline` holds one `_Layout` per resident mesh assignment (its own `CCLManager`,
`DiTGParallelConfigNoCFG`, transformers, denoisers, prompt-encoder wrapper and decode helper).

The denoiser keeps one slot set per layout key `(flow_map, sp, cond lengths, uncond lengths)`: the fp32 state `x`,
the scalar slots, the row mask, and per prompt the plan tree, `text_embedded` and `image_cond_embedded`. Slots live at
fixed addresses; a later request with the same key is refreshed by `ttnn.copy`, so the captured trace replays
unchanged (nothing per step is baked into the trace). A new key releases every live trace first, allocates its
slots, runs one eager step so every program is compiled (ttnn's program cache) and warms the copy programs, then the
first traced step captures. At most `max_live_traces` DiT traces stay captured; the VAE decode trace of a canvas size
is captured before the DiT trace of a layout (the trace aliasing rule in the module docstrings). The modulation of a
step is one merged matmul producing a 60-row table (`Flux3KleinModulation.table`); per-token rows are
`ttnn.embedding` gathers with the plan's constant indices, and pad rows get `(0, 1, 0)` so zero rows stay zero.
DBCache (`cache_flux3_klein.py`) is the cache-dit algorithm over the single-block stack: the first `Fn` blocks always
run, their residual is compared with the previous computed step's, and when the relative change is under
`residual_diff_threshold` (after `max_warmup_steps`) the middle blocks are skipped and the cached middle residual
is re-added (TaylorSeer extrapolation optional). `Flux3KleinCacheConfig.from_spec("F8B0:0.08:w8")` parses
`F<n>B<m>`, the threshold, `w<warmup>`, `c<max cached>`, `k<max continuous>`, `ts` / `ts2`, `all`. It is eager-only and
an approximation by construction; it is off unless `Flux3KleinRequest.cache` is set.

### Exact-fidelity rules

Keep these when touching any op in the path. Operands are bf16 with fp32 accumulation everywhere (`layers/linear.py`
`fp32_dest_acc_en=True`; the block replaces the attention's shared SDPA config with HiFi2, `math_approx_mode=False`,
`fp32_dest_acc_en=True`). The time-embedding MLP runs fp32 operands at HiFi4; the VAE convolutions and GroupNorms run
HiFi4 with fp32 accumulation, the exact GroupNorm path reduces fp32 statistics. No bfp8, bfp4 or fp16 anywhere
(`FLUX3_KLEIN_DIT_WEIGHTS` is a measurement override, never a default). The one approved approximation is the SFPU
approximate exp inside the ring SDPA softmax (`transformer_block_flux3_klein.py`, `ring_sdpa_exp_approx = True`).
Host-side values are the reference's own: bf16 timestep rounding (`scheduler.py`), bf16-valued noise, the
`to_pil` truncation. In the Euler chain no binary op mixes dtypes and `dt` / `guidance` are fp32 `[1, 1]` slot
tensors, never Python scalars (`denoise_flux3_klein.py` lists the forms that were measured wrong).

## Knobs

| name | where | effect |
|---|---|---|
| `FLUX3_KLEIN_MODEL_PATH` | `weights_flux3_klein.py` | explicit checkpoint snapshot or root (wins over the HF cache) |
| `HF_HUB_CACHE`, `HF_HOME` | `weights_flux3_klein.py` | the Hugging Face cache searched for snapshots |
| `TT_DIT_ALLOW_HF_DOWNLOAD=1` | `weights_flux3_klein.py` | permit a `snapshot_download` of a missing variant |
| `TT_DIT_CACHE_DIR` | `utils/cache.py`, `upsampler.py` | prepared device weights; unset = re-read every run |
| `TT_CACHE_PATH` | `upsampler.configure_pup_env` | the upsampler's tensor cache (default `$TT_DIT_CACHE_DIR/flux3_klein_pup`) |
| `FLUX3_KLEIN_DIT_WEIGHTS=bfp8[,lofi]` | `pipeline_flux3_klein._dit_weight_override` | measurement-only reduced-precision DiT linears |
| `FLUX3_KLEIN_VAE_SLICE_MODE` | `vae_flux3_klein._slice_mode` | decoder conv DRAM slicing: `auto` (default), `aligned`, `explicit` |
| `FLUX3_KLEIN_VAE_CONFIG_TENSORS_DRAM` | `vae_flux3_klein._config_tensors_in_dram` | conv config tensors in DRAM (default `1`) |
| `FLUX3_KLEIN_VAE_GN` | `vae_flux3_klein._exact_group_norm_mode` | decoder GroupNorm: `fused` (default) or `exact` fp32 statistics |
| `FLUX3_KLEIN_OUT` | `upsampler.py`, tests | upsampler scratch dir and test renders |
| `QWEN36_EXACT_PROFILE`, `QWEN36_MTP`, `HF_MODEL` | `upsampler.configure_pup_env` | set by the upsampler for the qwen36 package (bf16 profile, no drafter, the snapshot) |
| `PUP_DEVICE_SAMPLING=1`, `PUP_DIAG_SYNC=1` | `upsampler.py` | on-device argmax with presence penalty; per-step sync diagnostics |
| `FLUX3_KLEIN_GOLDENS`, `FLUX3_KLEIN_REFERENCE` | `tests/models/flux3_klein/common.py` | golden cases root; the BFL clone holding the example PNGs |
| `FLUX3_KLEIN_GOLDEN_PCC`, `FLUX3_KLEIN_REFERENCE_DEPTH`, `FLUX3_KLEIN_GOLDEN_FREE_RUN`, `FLUX3_KLEIN_GOLDEN_STEPS` | `tests/models/flux3_klein/common.py`, `test_reference_flux3_klein.py` | CPU golden test floor, smoke depth, free-run mode, step cap |
| `FLUX3_KLEIN_VAE_ORACLE_CACHE`, `FLUX3_KLEIN_SAMPLING_ORACLE_CACHE`, `FLUX3_KLEIN_PUP_ORACLE_CACHE`, `FLUX3_KLEIN_BASE_PROPER_REFERENCE=1` | tests | CPU oracle caches; opt-in base-model proper reference |
| `TT_METAL_TRACE_ALLOC_TRACKING=1` | `test_pipeline_flux3_klein.py` | trace allocation tracker (`test_trace_alloc_tracking`) |
| `TT_METAL_CACHE` | tt-metal | kernel JIT cache; a cold one compiles the VAE kernels for minutes |
| `MESH_SHAPE`, `L1_SMALL_SIZE`, `TRACE_REGION_SIZE`, `FABRIC_PACKET_BYTES` | `demo_flux3_klein.py` | mesh open parameters (mirrored by `tests/models/flux3_klein/device_params.py`) |
| `LARGE_LAYOUT_MIN_TOKENS` | `pipeline_flux3_klein.py` | row count from which `dual` switches to the large layout (16384) |
| `FLUX3_KLEIN_STEPS`, `FLUX3_KLEIN_GUIDANCE`, `FLUX3_KLEIN_ALPHA` | `scheduler.py` | per-variant defaults (16 / None, 50 / 3.5) and the schedule shift 6.93 |
| `FLUX3_KLEIN_MAX_REFERENCES`, `FLUX3_KLEIN_MAX_TEXT_LENGTH`, `FLUX3_KLEIN_TEXT_PAD_MULTIPLE` | `packing.py` | 10 references, 4096 text tokens, text padded to multiples of 80 |
| `FLUX3_KLEIN_MIN_SIDE`, `FLUX3_KLEIN_MAX_SIDE`, `FLUX3_KLEIN_MATCH_MAX_PIXELS`, `FLUX3_KLEIN_MULTI_REFERENCE_MAX_PIXELS` | `references.py` | canvas side range 64..4096, 4 MP single / 1 MP multi reference budgets |
| `FLUX3_KLEIN_SDPA_Q_CHUNK_MAX` | `transformer_block_flux3_klein.py` | ring SDPA q-chunk cap (128) that keeps the fp32-accumulating kernel in L1 |
| `AUTO_SLICE_MAX_AREA`, `DECODER_SLICE_BUDGET_BYTES` | `vae_flux3_klein.py` | decoder conv slicing thresholds |
| `Flux3KleinCacheConfig` defaults | `cache_flux3_klein.py` | `F8B0`, threshold 0.08, 8 warmup steps, unlimited cached steps, content rows only |
| `PUP_MAX_SEQ_LEN`, `PUP_PREFILL_CHUNK`, `PUP_DEVICE_PARAMS` | `upsampler.py` | upsampler context (8192), prefill chunk (2048), its 1-chip device parameters |
