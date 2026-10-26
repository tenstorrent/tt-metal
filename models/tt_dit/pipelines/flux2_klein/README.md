# FLUX.2 Klein (4B / 9B) on tt_dit

## What the model is

FLUX.2 Klein is Black Forest Labs' small FLUX.2 family: a text-to-image and reference-image editing model made of a
Qwen3 text encoder, a FLUX.2 diffusion transformer (DiT) and the FLUX.2 KL autoencoder (32 latent channels, 2x2
patch, 128 token channels). Two sizes and two variants share one code path. The 4B DiT is 3072 wide, 24 heads,
5 dual-stream blocks + 20 single-stream blocks, with Qwen3-4B (hidden 2560, taps 9 / 18 / 27 concatenated to a
7680-wide prompt row); the 9B DiT is 4096 wide, 32 heads, 8 dual + 24 single, with Qwen3-8B (hidden 4096, joint
12288). The `distilled` variant renders in 4 Euler steps without guidance; the `base` variant in 50 steps with
classifier-free guidance 4.0 (`scheduler.py`, `FLUX2_KLEIN_STEPS` / `FLUX2_KLEIN_GUIDANCE`). The port runs every
stage on a Blackhole Galaxy 4x8 mesh. The default DiT layout is `sp8tp4`: sequence-parallel over mesh axis 1 (the 8
columns, each device holds 1/8 of the token rows) and tensor-parallel over mesh axis 0 (the 4 rows, each device
holds 1/4 of the heads and hidden width). The alternatives are `sp4tp8` (axes swapped) and `dual` (both resident,
the SP8 x TP4 layout taking requests of 16384 DiT tokens and more). The text encoder is TP over the DiT's TP axis
and FSDP over the other axis; the VAE decoder is channels-TP4 over axis 0 and latent-rows-H8 over axis 1; the VAE
encoder is channels-TP4 with the spatial axes replicated.

## Weights

The four checkpoints are the diffusers-layout Hugging Face repositories (`weights_flux2_klein.py`,
`FLUX2_KLEIN_REPO_IDS`); they are gated, so accept the licence and log in before any download:

| `--model` | `--size` | repository |
|---|---|---|
| `distilled` | `4b` | `black-forest-labs/FLUX.2-klein-4B` |
| `distilled` | `9b` | `black-forest-labs/FLUX.2-klein-9B` |
| `base` | `4b` | `black-forest-labs/FLUX.2-klein-base-4B` |
| `base` | `9b` | `black-forest-labs/FLUX.2-klein-base-9B` |

`resolve_weights_dir(variant, size)` finds a snapshot in this order: the `--weights` argument, then
`$FLUX2_KLEIN_MODEL_PATH` (a snapshot directory, or a root holding `FLUX.2-klein-4B/` or `distilled-4b/` children;
ignored when the path identifiably names another checkpoint), then the Hugging Face cache (`$HF_HUB_CACHE`, else
`$HF_HOME/hub`, else `~/.cache/huggingface/hub`: the `refs/main` revision first, then the newest complete
snapshot), then a download only when `TT_DIT_ALLOW_HF_DOWNLOAD=1`. A complete snapshot holds `model_index.json`,
`transformer/config.json`, `transformer/diffusion_pytorch_model.safetensors`, `text_encoder/`, `tokenizer/`,
`vae/` and `scheduler/` (`REQUIRED`). The converted device weights are cached under `$TT_DIT_CACHE_DIR` by
`utils/cache.py`: the DiT under `flux2_klein_<variant>_<size>_<snapshot>/transformer_D<dual>_S<single>/`, the text
encoder under `flux2_klein_text_encoder_<blob>/text_encoder_L27/` (the 27-layer build; shared by base and
distilled of a size), the VAE under `flux2_klein_vae_<blob>/vae/` and `/vae_encoder/`.

## Launch

The demo (`demo_flux2_klein.py`) opens the 4x8 mesh with the ring fabric, builds one `Flux2KleinPipeline` and
renders. From the repository root with the tt-metal environment active:

```bash
# text-to-image, distilled 4B, 1024x1024, 4 steps
python models/tt_dit/pipelines/flux2_klein/demo_flux2_klein.py "a lighthouse on a basalt shore at dusk" \
    --model distilled --size 4b --width 1024 --height 1024 --seed 7 --out out/t2i.png

# editing with references: every --image is one reference (at most 10); the canvas defaults to the first one's size
python models/tt_dit/pipelines/flux2_klein/demo_flux2_klein.py "put a red bandana around the dog's neck" \
    --image dog.png --seed 7 --out out/edit.png

# base 9B: 50 steps, CFG 4.0, a non-square canvas, an explicit snapshot
python models/tt_dit/pipelines/flux2_klein/demo_flux2_klein.py "a watercolour harbour at dawn" \
    --model base --size 9b --guidance 4.0 --width 1536 --height 1024 --weights /path/to/snapshot --out out/base.png

# eager (untraced) denoise and decode, the two-layout pipeline, three renders in one process
python models/tt_dit/pipelines/flux2_klein/demo_flux2_klein.py --prompt-file prompt.txt \
    --eager --layout dual --repeat 3 --out out/run.png
```

Flags (`parse_args`): the positional `prompt` or `--prompt-file`; `--model {distilled,base}` (default
`distilled`); `--size {4b,9b}` (default `4b`); `--image PATH` (repeatable); `--width` / `--height` (multiples of 16,
default the first reference's size, else 1024); `--steps` (default 4 / 50); `--guidance` (base only, default 4.0);
`--seed` (default a fresh 32-bit seed); `--out PATH` (default `out.png`; `--repeat N` writes `<stem>_<i>.png`);
`--weights DIR`; `--eager`; `--layout {sp8tp4,sp4tp8,dual}` (default `sp8tp4`); `--repeat N` (default 1). Every
render writes a `<out>.json` sidecar with the model, size, prompt, canvas, steps, guidance, seed and reference
paths. Exit code 2 is a bad input, 3 a canvas the admission policy refused.

Programmatic use:

```python
from models.tt_dit.pipelines.flux2_klein.pipeline_flux2_klein import Flux2KleinPipeline, Flux2KleinRequest

pipeline = Flux2KleinPipeline.create(mesh_device, size="4b", resident=("distilled",))  # resolves the snapshot
result = pipeline.generate(Flux2KleinRequest(prompt="...", seed=7, images=("dog.png",)), traced=True)
result.image.save("out.png")   # result.timings, result.seed, result.layout, result.final_tokens
pipeline.close()
```

### Tests

The suite is `models/tt_dit/tests/models/flux2_klein/`; `common.py` resolves the snapshot and the goldens from the
environment (`FLUX2_KLEIN_MODEL_PATH` or the HF cache, `FLUX2_KLEIN_GOLDENS`, `FLUX2_KLEIN_ORACLE_CACHE` for the CPU
oracles between runs, `FLUX2_KLEIN_OUT` for saved renders, `TT_DIT_CACHE_DIR`); `device_params.py` holds the mesh
dictionaries (ring fabric, 8 KB packets, `l1_small_size` 65536, `trace_region_size` 256 MB). Host-only rows need
no device (they import diffusers / transformers and compare `torch.equal`):

```bash
pytest models/tt_dit/tests/unit/test_flux2_klein_layout_pick.py
pytest models/tt_dit/tests/models/flux2_klein/test_host_reference_flux2_klein.py
pytest models/tt_dit/tests/models/flux2_klein/test_pipeline_flux2_klein.py -k test_request_resolve
pytest models/tt_dit/tests/models/flux2_klein/test_prompt_encoder_flux2_klein.py \
    -k "test_tokenize_matches_reference or test_config_and_sizes or test_reduced_depth_config"
pytest models/tt_dit/tests/models/flux2_klein/test_vae_flux2_klein.py -k "test_unpack_matches_pipeline or test_plain_grid_rule"
```

Device rows run on the 4x8 ring (`bh_4x8_ring`), parametrized over `size` and `layout`:

```bash
pytest models/tt_dit/tests/models/flux2_klein/test_prompt_encoder_flux2_klein.py -k "test_device_taps or test_dit_handoff"
pytest models/tt_dit/tests/models/flux2_klein/test_transformer_flux2_klein.py
pytest models/tt_dit/tests/models/flux2_klein/test_vae_flux2_klein.py -k "test_decode_golden_tokens or test_encode_golden_png"
pytest models/tt_dit/tests/models/flux2_klein/test_pipeline_flux2_klein.py -k "test_distilled_render or test_base_cfg_smoke"
```

Goldens are CPU renders of the diffusers `Flux2KleinPipeline` in bf16, observed and never altered, written by
`generate_goldens_flux2_klein.py` (no `ttnn` import; minutes per case on a many-core CPU):

```bash
python models/tt_dit/tests/models/flux2_klein/generate_goldens_flux2_klein.py \
    --model distilled --size 4b --prompt man_dog --canvas 1024 1024 --seed 7 --threads 40 --out /path/to/goldens
python models/tt_dit/tests/models/flux2_klein/generate_goldens_flux2_klein.py \
    --model base --size 4b --steps 4 --guidance 4.0 --out /path/to/goldens      # the base CFG smoke case
```

Each case directory (`<out>/<model>/<case>/`) holds `record.json`, `inputs.safetensors` (the packed noise, ids, the
recorded text encodings, the sigmas), `steps.safetensors` (the state entering every step, the velocities, the final
tokens, the fp32 image) and `golden.png`. What the gates compare against: the transformer test runs one step on
random inputs against `Flux2Transformer2DModel` on CPU (velocity pcc >= 0.995, rel_rmse <= 0.10); the prompt-encoder
test runs the device Qwen3 against `Qwen3ForCausalLM` on CPU (per-tap pcc >= 0.99, concatenated rows pcc >= 0.9997,
pad block pcc >= 0.997); the VAE test decodes the golden tokens and encodes `golden.png` against the fp32
`AutoencoderKLFlux2` (decode pcc >= 0.9978 and rel_rmse <= 0.034, 50 dB PSNR floor, encode pcc >= 0.999); the
pipeline test renders with the golden's recorded text (`text_override`) and compares the PNG to `golden.png`
(PSNR >= 30 dB for distilled), the final state to the golden's (pcc / rel_rmse band 0.99 / 0.15 distilled, 0.97 /
0.25 base) and every teacher-forced step velocity (pcc >= 0.995). Host arithmetic (noise, ids, schedule, Euler
update, packing) is compared bit for bit. The thresholds are only ever tightened.

## Code flow

### Module map

`models/tt_dit/pipelines/flux2_klein/` (this directory):

| file | what |
|---|---|
| `demo_flux2_klein.py` | the command line: mesh + fabric setup (`open_mesh`), `LAYOUTS`, the render loop, the sidecar |
| `pipeline_flux2_klein.py` | `Flux2KleinPipeline`, `Flux2KleinRequest` / `Flux2KleinResolved` / `Flux2KleinResult`, the `_Layout` record, `pick_layout`, the env-var readers, the uint8 readback |
| `policy.py` | `admit`: the host-side admission checks and `Flux2KleinAdmissionError` |
| `references.py` | reference sizing to the reference recipe (1 MP cap, 16-px floor, centre crop), `PreparedReferences`, `ReferenceCache` |
| `prompt_encoder.py` | `Flux2KleinPromptEncoder` (chat template, 512-token tokenization, the 27-layer Qwen3, device-resident RoPE tables and attention biases), `Flux2KleinTextEncoding` (`encode_device` / `encode_host`), `build_flux2_klein_text_encoder` |
| `vae_encode_flux2_klein.py` | `Flux2KleinVaeEncode.encode_references`: image upload (one row band per mesh column, gathered on device), encoder forward, bf16 tokens + ids |
| `packing.py` | latent geometry (`latent_shape`), the 4-axis ids (`latent_ids`, `reference_ids`, `text_ids`), `make_noise` / `make_noise_cached`, 2x2 patchify, BatchNorm helpers, tile padding (`padded_length`) |
| `scheduler.py` | `compute_empirical_mu`, the shifted `linspace(1, 1/N, N)` sigmas, `Flux2KleinScheduler` / `Flux2KleinStep` (the bf16 timestep chain, `dt_rounded`) |
| `denoise_flux2_klein.py` | `Flux2KleinDenoiser`: slots per layout key, `prepare`, the traced pure `_step`, `denoise_on_device`, the Euler / CFG elementwise chains |
| `vae_flux2_klein.py` | `Flux2KleinVaeDecode`: SP gather of the state, H re-partition, inverse BatchNorm + unpatchify, the decoder, conv slice tables |
| `weights_flux2_klein.py` | `resolve_weights_dir`, the repo ids, `WeightsNotFoundError` |
| `weights_dit_flux2_klein.py` | the DiT state dict through the weight cache, the host time-path weights |

Model code it imports:

| directory | what |
|---|---|
| `models/tt_dit/models/transformers/flux2_klein/` | `config_flux2_klein.py` (`Flux2KleinConfig`, `FLUX2_KLEIN_SIZES`), `rope_flux2_klein.py` (4-axis RoPE tables, the single-stream interleave), `time_flux2_klein.py` (`Flux2KleinHostTime`: the bit-exact host modulation; `Flux2KleinTimeModulation`: the device fallback), `block_flux2_klein.py` (`Flux2KleinSingleBlock`, `apply_exact_attention_profile`), `transformer_flux2_klein.py` (`Flux2KleinTransformer`, `Flux2KleinSequenceLayout`, `Flux2KleinPlan`, `Flux2KleinPlanCache`) |
| `models/tt_dit/encoders/qwen3/` | `Qwen3Encoder` (the generic `encoders/transformer.py` decoder with GQA and q/k RMSNorm) and `Qwen3Checkpoint` (config + weights of a layer prefix) |
| `models/tt_dit/models/vae/` | `vae_flux2.py::Flux2VaeDecoder` (shared, unchanged) and `encoder_flux2.py::Flux2VaeEncoder` (down blocks, mid block, `quant_conv`, the fp32 BatchNorm normalize) |

Shared tt_dit pieces used as they are: `blocks/transformer_block_opt.py` (the dual block), `blocks/attention_opt.py`
(ring joint attention), `layers/linear.py` / `layers/normalization.py` / `layers/conv2d.py`, `parallel/manager.py`
(`CCLManager`), `utils/cache.py` (the weight cache), `utils/tracing.py` (`StateTensor`, `traced_function`).

### The request path

1. Request: `Flux2KleinRequest` (prompt, model, size, width / height, seed, steps, guidance, `images`) is turned into
   a `Flux2KleinResolved` by `Flux2KleinRequest.resolve` (`pipeline_flux2_klein.py`): the variant defaults, the
   distilled-guidance check, then `references.prepare_references` sizes every reference on the host (path sources
   through the pipeline's `ReferenceCache`) and fixes the canvas.
2. Layout pick and admission: `Flux2KleinPipeline._layout_of` picks the resident DiT layout by the row count
   (`pick_layout`, `LARGE_LAYOUT_MIN_TOKENS`); `policy.admit` checks the sides (multiples of 16 in [16, 4096]), the
   reference count (at most 10), that every VAE level of the canvas is tile-aligned and that the spatial stream's
   per-device rows differ from the prompt stream's (the two would alias their CCL ping-pong buffers). A refusal
   raises `Flux2KleinAdmissionError` before anything touches the device.
3. Noise and ids: `packing.make_noise(seed, h, w)` (the reference's bf16 `randn` packed row-major) is drawn on a
   worker thread while the encodes run; `latent_ids(h, w)` gives the spatial ids.
4. Prompt encoding: `Flux2KleinTextEncoding.encode_device` (`prompt_encoder.py`) tokenizes with the Qwen3 chat
   template to 512 rows, runs the device `Qwen3Encoder` (layers 0-26, no final norm) and concatenates hidden states
   9 / 18 / 27; the `[512, joint]` rows stay on the device, `ttnn.mesh_partition`ed over the layout's SP axis. The
   base model's empty prompt is encoded once per layout at start-up (`encode_host`).
5. References: `Flux2KleinVaeEncode.encode_references` (`vae_encode_flux2_klein.py`) uploads each prepared image,
   runs `Flux2VaeEncoder.encode` (posterior mean, 2x2 patchify, BatchNorm normalize) and returns bf16 `[N, 128]`
   tokens with `reference_ids_for` ids (`t = 10 + 10 i`).
6. Plan and slots: `Flux2KleinDenoiser.prepare` (`denoise_flux2_klein.py`) builds the `Flux2KleinSequenceLayout`
   via `Flux2KleinTransformer.sequence_layout`, takes the RoPE plan from `Flux2KleinTransformer.plan_for` (cached),
   runs `embed_prompt` once (the hoisted `context_embedder`), writes `[noise | reference tokens | zero pad]` into the
   fp32 state slot and uploads the schedule's step constants the caches do not hold yet.
7. Denoise: `denoise_on_device` runs `Flux2KleinScheduler` steps; each `step` refreshes the `t_embed` / `dt` /
   modulation slots and calls the pure `_step` (one `Flux2KleinTransformer.forward` per sample, `combine_cfg` for
   the base model, `euler_update` + `round_state_to_bf16`), as one captured trace per layout key when `traced`.
8. Decode: `Flux2KleinVaeDecode.decode` (`vae_flux2_klein.py`) all-gathers the state over SP, slices the image rows,
   re-partitions them over the VAE's H axis, applies the inverse BatchNorm and unpatchify and runs
   `Flux2VaeDecoder`; its trace is captured before the DiT trace of the same layout.
9. Host image: `image_to_uint8_on_device` does `(x / 2 + 0.5).clamp(0, 1) * 255` on the device (the sum rounded to
   bf16 as the reference's `denormalize` does, then `round` and a uint8 typecast) and one replica is read; `Flux2KleinResult` carries the PIL image, the seed, the timings and optionally the final
   tokens and the text rows.

### The layout object

`Flux2KleinSequenceLayout` (`transformer_flux2_klein.py`) is the row contract of one request: `image` latent rows,
`cond` reference rows, `real = image + cond`, `padded = padded_length(real, sp_factor)` (a multiple of `32 * SP`),
`prompt` (512) and `sp_factor`. The DiT's parallel config is a `DiTGParallelConfigNoCFG` with
`sequence_parallel = ParallelFactor(factor=mesh.shape[sp_axis], mesh_axis=sp_axis)` and `tensor_parallel` on the
other axis (`Flux2KleinPipeline._build_layout`). The spatial stream is `[1, L_pad / SP, 128]` with
`mesh_axes=[None, sp, None]`; the prompt stream is 512 rows SP-sharded the same way (`shard_prompt=True`, required
from SP 4 up); weights are TP-fractured over the TP axis, so activations inside a block are `[rows / SP, D / TP]`
with `mesh_axes=[None, sp, tp]`. Pad rows are the global tail and are masked as keys through the plan's logical
lengths. The single blocks run on one per-device sequence `[prompt_d | spatial_d]` (one `ttnn.concat` per step).
The denoiser's layout key is `(sp, S_img, S_cond, L_pad, cfg)`.

### The caches

- Plan cache (`Flux2KleinPlanCache`, `FLUX2_KLEIN_PLAN_CACHE_SIZE` = 16, LRU): the uploaded RoPE tables of a
  `(layout, spatial ids, text ids)`; the prompt pair is shared by every plan. The plan belongs to the model; a miss
  releases every live trace before it allocates.
- Modulation slots and step-constant caches (`Flux2KleinDenoiser._mod_cache`, `_scalar_cache`): the 17 + 1 bf16
  modulation chunks per distinct timestep (computed on the host by `Flux2KleinHostTime`, bit-equal to the reference)
  and the fp32 `t_embed` / `dt` scalars per distinct value, uploaded once per process and `ttnn.copy`'d into the
  traced step's slots.
- Slots per layout key (`_LayoutSlots`): the fp32 state, the scalars, the modulation, the row mask and the hoisted
  prompt embedding, at fixed addresses so a captured trace can be replayed; refreshed in place by `ttnn.copy`.
- Program cache: ttnn's per-device cache of compiled programs. `prepare` of a new key runs one eager step so every
  program of the layout is compiled (and every copy program warmed) before a trace is captured; `__init__` warms the
  VAE decoder and encoder at `warmup_size` for the same reason.
- Host caches: `make_noise_cached` (last 8 `(seed, h, w)`), `ReferenceCache` (last 16 path references keyed by path
  + mtime + size), and the prompt encoder's device-resident RoPE tables and attention biases (32-entry LRU).

### Exact-fidelity rules

The port's rule is bf16 operands with fp32 accumulation everywhere and no bfp8 / fp16 / quantisation:

- every DiT linear runs `exact_matmul_config` (`block_flux2_klein.py`): `FLUX2_KLEIN_DIT_MATH_FIDELITY_DEFAULT` =
  HiFi4, `math_approx_mode=False`, `fp32_dest_acc_en=True`; HiFi2 is the only admitted alternative;
- the ring SDPA of every `Attention` (dual and single blocks) takes `apply_exact_attention_profile`: HiFi4 compute
  (`FLUX2_KLEIN_SDPA_MATH_FIDELITY`, not under the linear knob), fp32 accumulation, q chunks clamped to
  `FLUX2_KLEIN_SDPA_Q_CHUNK_MAX` = 128 so the fp32 kernel fits L1;
- the one approved deviation is the approximate SFPU exp inside that ring SDPA
  (`FLUX2_KLEIN_SDPA_EXP_APPROX_DEFAULT` = True); everything else in the softmax accumulates in fp32;
- the time path runs on the host (`Flux2KleinHostTime`), op for op as the bf16 reference; the Euler and CFG chains
  never mix dtypes or bake a Python scalar into a traced op (`denoise_flux2_klein.py` module docstring lists the
  forms that are wrong); the state is bf16-valued between steps;
- the text encoder runs bf16 linears (`Qwen3Encoder`: `linear_dtype` / `attn_qkv_dtype` bf16, not the generic bfp8)
  with fp32 accumulation; the VAE convolutions and GroupNorms are HiFi4 fp32-accumulated and its attention uses the
  precise exp;
- the host modules (`packing.py`, `scheduler.py`, `references.py`) are bit-exact against the reference and must not
  be reassociated or moved to the device.

## Knobs

| variable / constant | where | effect |
|---|---|---|
| `FLUX2_KLEIN_MODEL_PATH` | `weights_flux2_klein.py` | explicit snapshot (or root of snapshots) |
| `HF_HUB_CACHE`, `HF_HOME` | `weights_flux2_klein.py` | the Hugging Face cache searched next |
| `TT_DIT_ALLOW_HF_DOWNLOAD=1` | `weights_flux2_klein.py` | allow a missing snapshot to be downloaded |
| `TT_DIT_CACHE_DIR` | `utils/cache.py` | the converted-weight cache root |
| `FLUX2_KLEIN_SDPA_EXP_APPROX` (`1` / `0`) | `pipeline_flux2_klein.py` | ring SDPA exp: approximate (default) or precise; the environment wins over the `sdpa_exp_approx=` keyword |
| `FLUX2_KLEIN_DIT_FIDELITY` (`hifi4` / `hifi2` / `default`) | `pipeline_flux2_klein.py` | matmul fidelity of every DiT linear; the environment wins over `dit_math_fidelity=` |
| `FLUX2_KLEIN_TEXT_ENCODER_FIDELITY` (`default` / `hifi4` / ...) | `pipeline_flux2_klein.py` | Qwen3 linear fidelity (default HiFi2); the environment wins over the `text_encoder_math_fidelity=` keyword |
| `FLUX2_KLEIN_TEXT_ENCODER_FSDP` (`1` / `0`) | `pipeline_flux2_klein.py` | text encoder weights sharded over the non-TP axis (default) or replicated there; the environment wins over `text_encoder_fsdp=` |
| `FLUX2_KLEIN_RECORD_MEMORY` (`1` / `0`) | `pipeline_flux2_klein.py` | append the mesh memory view to every per-request log line (default off) |
| `FLUX2_KLEIN_VAE_SLICE_MODE` (`auto` / `aligned` / `explicit`) | `vae_flux2_klein.py` | how the VAE conv slice tables are chosen |
| `FLUX2_KLEIN_VAE_CONFIG_TENSORS_DRAM` (`1` / `0`) | `vae_flux2_klein.py` | keep the conv halo config tensors in DRAM (default) or L1_SMALL |
| `FLUX2_KLEIN_CONV_PREP_CACHE` (`per_shape` / `single`) | `layers/conv2d.py` | one prepared conv weight copy per input shape (default) or one per module |
| `FLUX2_KLEIN_GOLDENS`, `FLUX2_KLEIN_ORACLE_CACHE`, `FLUX2_KLEIN_OUT` | `tests/models/flux2_klein/common.py` | goldens root, CPU oracle cache, saved test renders |
| `FLUX2_KLEIN_PLAN_CACHE_SIZE` = 16 | `transformer_flux2_klein.py` | uploaded RoPE plans kept per model |
| `FLUX2_KLEIN_SDPA_Q_CHUNK_MAX` = 128 | `config_flux2_klein.py` | ring SDPA q chunk cap under fp32 accumulation |
| `FLUX2_KLEIN_DIT_MATH_FIDELITY_DEFAULT`, `FLUX2_KLEIN_SDPA_MATH_FIDELITY`, `FLUX2_KLEIN_SDPA_EXP_APPROX_DEFAULT` | `block_flux2_klein.py` | the exact-profile defaults (HiFi4, HiFi4, approximate exp) |
| `FLUX2_KLEIN_TEXT_ENCODER_LAYERS` = 27, `FLUX2_KLEIN_ENCODER_BIAS_CACHE_SIZE` = 32 | `prompt_encoder.py` | Qwen3 depth built; attention-bias LRU |
| `FLUX2_KLEIN_MAX_REFERENCES` = 10, `FLUX2_KLEIN_TEXT_LENGTH` = 512, `FLUX2_KLEIN_NOISE_CACHE_SIZE` = 8 | `packing.py` | reference limit, prompt rows, memoised noise draws |
