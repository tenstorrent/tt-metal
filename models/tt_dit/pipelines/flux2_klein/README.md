# FLUX.2 Klein on Tenstorrent (tt_dit)

A port of Black Forest Labs' FLUX.2 Klein text-to-image / image-editing models to a Tenstorrent Galaxy mesh. The
reference is the diffusers `Flux2KleinPipeline` (`pipeline_flux2_klein.py`, cited as `P` in the docstrings) with
`Flux2Transformer2DModel` (`T`), `FlowMatchEulerDiscreteScheduler` (`S`) and `AutoencoderKLFlux2` (`AE`); the BFL
`flux2` sources (`model.py` `M`, `sampling.py` `SA`, `util.py` `U`) are the second citation. Every module's
docstring names the reference lines it mirrors.

Four checkpoints are served by one code path:

| `model`     | `size` | repository                              | DiT                                   | text encoder | steps | CFG  |
|-------------|--------|-----------------------------------------|---------------------------------------|--------------|-------|------|
| `distilled` | `4b`   | `black-forest-labs/FLUX.2-klein-4B`      | 3072 wide, 24 heads, 5 dual + 20 single | Qwen3-4B (2560 hidden, joint 7680)  | 4  | none |
| `distilled` | `9b`   | `black-forest-labs/FLUX.2-klein-9B`      | 4096 wide, 32 heads, 8 dual + 24 single | Qwen3-8B (4096 hidden, joint 12288) | 4  | none |
| `base`      | `4b`   | `black-forest-labs/FLUX.2-klein-base-4B` | as 4B                                 | as 4B        | 50    | 4.0  |
| `base`      | `9b`   | `black-forest-labs/FLUX.2-klein-base-9B` | as 9B                                 | as 9B        | 50    | 4.0  |

The 4B row is the snapshot's `transformer/config.json`; the 9B row is `Klein9BParams` (`M:24-35`) and is validated
against `transformer/config.json` when that snapshot is on disk (`Flux2KleinConfig.from_snapshot`). The VAE is the
FLUX.2 autoencoder (32 latent channels, 2x2 patch, BatchNorm statistics) for every row.

## File map

`models/tt_dit/models/transformers/flux2_klein/` (the DiT):

| file | what |
|---|---|
| `config_flux2_klein.py` | `Flux2KleinConfig`, `FLUX2_KLEIN_SIZES` (the table above), `FLUX2_KLEIN_SDPA_Q_CHUNK_MAX`, per-mesh shape checks |
| `rope_flux2_klein.py` | 4-axis RoPE tables, theta 2000, axes (32, 32, 32, 32), fp32 (`T:971-998`, `E:885-949`) |
| `time_flux2_klein.py` | fp32 sinusoid -> bf16 timestep MLP -> the merged 17-chunk modulation linear (`T:1011-1056`, `N:346-351`) |
| `block_flux2_klein.py` | `Flux2KleinSingleBlock` (`T:723-873`) and `apply_exact_attention_profile` (the exact profile below) |
| `transformer_flux2_klein.py` | `Flux2KleinTransformer` (`T:1226-1434`): dual blocks from `blocks/transformer_block_opt.py`, single blocks, `norm_out`, `proj_out`, `sequence_layout`, `plan_for` (the plan cache: uploaded RoPE tables per (layout, ids), `FLUX2_KLEIN_PLAN_CACHE_SIZE` 16 LRU entries, the prompt pair shared; `prepare_plan` is the fresh caller-owned form), `embed_prompt` |

`models/tt_dit/pipelines/flux2_klein/` (this directory):

| file | what |
|---|---|
| `packing.py` | latent geometry, 4-axis ids, `make_noise` (the reference's bf16 `randn` + `_pack_latents`; `make_noise_cached` memoises it per `(seed, h, w)`, last 8), 2x2 patchify, BatchNorm helpers, tile padding (`P:287-397`, `P:464-477`) |
| `scheduler.py` | `compute_empirical_mu`, the `linspace(1, 1/N, N)` sigmas with exponential time shift, `Flux2KleinScheduler` and the bf16 timestep chain (`P:63-78`, `P:814-825`, `S:283-391`, `S:432-530`) |
| `references.py` | condition-image preparation: 1 MP cap, 16-px floor, centre crop, `[-1, 1]` (`P:764-787`); `ReferenceCache` keeps the prepared image + fp32 pixels of the last 16 PATH references (key path + mtime + size) so a file served again skips its PNG decode (40-100 ms at 1 MiP) and conversion (~10 ms); PIL images are prepared per request |
| `vae_flux2_klein.py` | `Flux2VaeDecoder` wiring: conv slicing, weight cache, `Flux2KleinVaeDecode` (BatchNorm inverse, unpatchify, decode) |
| `vae_encode_flux2_klein.py` | `Flux2VaeEncoder` wiring, `Flux2KleinVaeEncode.encode_references` (posterior mean, BatchNorm, patchify, reference ids); the reference image reaches the encoder as ONE row band per mesh column (`sharded_upload_view`: 0.8 MB per chip instead of 6.3 MB x 32) and is rebuilt replicated on the device (tilize, CCL all-gather over the replicated axis, untilize, reshape: bf16 data movement, tokens bit-equal); sizes without a tile-aligned band (880x1168, 400x736, ...) keep the host replication; `replicate_on_device=False` forces it (A/B) |
| `weights_dit_flux2_klein.py` | the diffusers DiT state dict and its `utils.cache.load_model` key |
| `weights_flux2_klein.py` | `resolve_weights_dir(variant, size)`: explicit path / HF cache / opt-in download, `WeightsNotFoundError` |
| `prompt_encoder.py` | Qwen3 chat template, 512-token tokenization, taps 9 / 18 / 27 concatenated (`P:208-263`); the device encoder runs decoder layers 0-26 only (`FLUX2_KLEIN_TEXT_ENCODER_LAYERS` = 27 = max(taps), no final norm: hidden_states[27] is the output of layer 26, layers 27-35 and the norm only feed the discarded logits; the FLUX 3 Klein text encoder does the same with 32 of 36), `num_layers` / `final_norm` keywords for the full-depth reference build; `build_flux2_klein_text_encoder`, `Flux2KleinTextEncoding` (`encode_device`: the rows stay on the device, `ttnn.mesh_partition`ed over the layout's SP axis into the denoiser's `embed_prompt` placement, no host round trip; `encode_host`: the `[512, joint]` host rows, one replica read); the per-request constants stay on the device: the `arange(512)` RoPE tables once per sequence length, the causal + padding attention bias built on the host (`encoders/transformer.py` `attention_bias`, exactly `{0, MASK_VALUE}` bf16) once per real length R in a 32-entry LRU (512 KB per chip and entry), so a request uploads only its tokens (rows bit-equal to the mask path, D4) |
| `denoise_flux2_klein.py` | `Flux2KleinDenoiser`: slots (the state, scalars, modulation, hoisted prompt; the plan is a reference into the transformer's plan cache, so a request of a seen canvas uploads no RoPE table), the traced pure `_step`, exact Euler / CFG elementwise chains |
| `policy.py` | `admit`: canvas sides multiple of 16 in [16, 4096], at most 10 references, VAE levels tile-aligned, per-device spatial rows != prompt rows (the two streams' CCL ping-pong buffers would alias: `L_pad == 512` under the SP-sharded prompt) |
| `pipeline_flux2_klein.py` | `Flux2KleinPipeline`, `Flux2KleinRequest` / `Flux2KleinResult`, layouts, warm-ups, the uint8 readback; the initial latents are drawn on the pipeline's worker thread while the encodes run (`timings["noise_wait_s"]` = what the join still waited); `request.resolve(reference_cache=pipeline.reference_cache)` is timed as `timings["resolve_s"]` with the cache hits / misses (`reference_cache=False` prepares every reference per request) |
| `demo_flux2_klein.py` | command line: mesh + fabric setup, one or more renders, `<out>.json` sidecar |

Shared tt_dit code this port reuses unchanged: `blocks/transformer_block_opt.py` (the dual block), `blocks/attention_opt.py`
(ring joint attention), `layers/linear.py` / `layers/normalization.py` / `layers/embeddings.py`, `models/vae/vae_flux2.py`
(the decoder), `utils/cache.py` (the weight cache), `encoders/transformer.py` (the generic decoder-only encoder).

Shared files this port touches (each owned by one package, behaviour of every other model unchanged):

| file | change |
|---|---|
| `encoders/qwen3/__init__.py`, `encoders/qwen3/model_qwen3.py` | NEW: `Qwen3Encoder` (the generic `TransformerEncoder` with GQA and q/k RMSNorm, HF key renames) and `Qwen3Checkpoint` (config from `text_encoder/config.json`, weights through the cache with a caller-supplied cache name and the `_FSDP` key when sharded; `num_layers` / `final_norm` build a prefix of the decoder, `route_qwen3_state_keys` drops the upper layers' and the final norm's keys explicitly and raises on any unrouted key, the reduced entry caches under `<subfolder>_L<num_layers>`) |
| `encoders/transformer.py` | `TransformerEncoderConfig.linear_dtype` threaded into the attention output projection, the feed-forward and the final linear (bf16 for the exact profile; the generic default stays bfp8); `linear_math_fidelity` (None = the layers' HiFi2 default) threaded into qkv / o_proj / feed-forward / final linear as their `compute_kernel_config` |
| `blocks/attention_opt.py` | ring SDPA program config cached per `(sequence length, worker grid)`, the worker grid resolved on every forward, `exp_approx_mode` read from the instance flag `ring_sdpa_exp_approx` (default on for this port) |
| `utils/matmul.py` | the L1-fit of heuristic all-gather-matmul blockings (`_fit_blocking_to_l1`; the 4096-px SwiGLU under SP8 x TP4 overflows L1 without it) |
| `layers/conv2d.py` | `conv_config_overrides` so the VAE can keep its halo config tensors in DRAM on large canvases; one prepared weight / bias copy per input-shape key (`ttnn.prepare_conv_weights` lays the weight out for one conv program: the decoder is warmed at 1024x1024 in `__init__` and then decodes 2048x2048 on the same instance; FLUX 3 Klein's striped 2048+ renders came from reusing the 1024 copy) |
| `models/vae/encoder_flux2.py` | NEW: the FLUX.2 VAE encoder (down blocks, mid block, `quant_conv`) on `models/vae/vae.py` |

## Exact profile

The rule for the whole model: bf16 operands, fp32 accumulation, no bfp8 / fp16 / quantisation. The configs are
the shared tt_dit defaults where those already satisfy it and explicit overrides where they do not.

| op class | config | where |
|---|---|---|
| every DiT matmul with bf16 weights (`x_embedder`, `context_embedder`, `to_qkv` / `add_qkv_proj`, `to_out` / `to_add_out`, the dual block's feed-forward, `proj_mlp`, `proj_out`, the final `proj_out`) | **HiFi4** by default (`FLUX2_KLEIN_DIT_MATH_FIDELITY_DEFAULT`; all mantissa bits of both bf16 operands, 4 FPU passes; the shared HiFi2 default, 2 passes, drops the low bits of one operand per product), `fp32_dest_acc_en=True`, `math_approx_mode=False`; `packer_l1_acc` kept where the shared layer had it (qkv, dual feed-forward). **HiFi2 is the opt-in** (`FLUX2_KLEIN_DIT_FIDELITY=hifi2` / `Flux2KleinPipeline(dit_math_fidelity=ttnn.MathFidelity.HiFi2)`; the fidelity FLUX 3's exact profile runs on). A/B 2026-10-24 (job 147350, one box, one process, warm eager sp8tp4, `Flux2KleinTransformer.apply_math_fidelity` swapped in place): DITFID_AB_NUMBERS | `block_flux2_klein.exact_matmul_config(math_fidelity=)`, applied in the constructors and by `apply_exact_attention_profile` / `apply_linear_math_fidelity` (2026-10-23: whole-DiT step-0 velocity vs diffusers bf16 CPU, 4B 1024x1024, rel_rmse 0.0433 -> 0.0328 together with the SDPA row below) |
| DiT ring SDPA compute (QK^T, PV) | **HiFi4** always (`FLUX2_KLEIN_SDPA_MATH_FIDELITY`, not under the DiT linear knob: HiFi4 took the dual-block SDPA output from rel_rmse 0.0193 to 0.0144 vs the CPU reference, job 146747), `math_approx_mode=False`, `fp32_dest_acc_en=True` (the shared default is HiFi2 and accumulates in bf16); the ring SDPA is bound by the exp and the fp32 accumulation, not by matmul fidelity | `apply_exact_attention_profile` |
| device time module (`Flux2KleinTimeModulation`, the fallback when a transformer has no host path) | fp32 sinusoid -> bf16 timestep MLP and the merged 17-chunk head on the `layers/linear.py` defaults (HiFi2) | `time_flux2_klein.py` |
| DiT ring SDPA exp | **approximate** (`ring_sdpa_exp_approx = FLUX2_KLEIN_SDPA_EXP_APPROX_DEFAULT = True`; user decision 2026-10-23 23:3x 'the approx looks good enough'); the precise exp is the opt-in `sdpa_exp_approx=False` / `FLUX2_KLEIN_SDPA_EXP_APPROX=0`. A/B 2026-10-23 (4B man_dog 1024 seed 7 4 steps eager sp8tp4, jobs 146815 approx / 146817 precise, b05u08): teacher-forced velocity rel_rmse 0.033/0.079/0.052/0.042 -> 0.032/0.054/0.046/0.037, golden-text e2e 28.84 -> 31.02 dB, device-text e2e 25.26 -> 26.79 dB; warm denoise 1024x1024 0.524 -> 0.520 s, 2048x2048 0.590 -> 0.713 s (+21 %) | `apply_exact_attention_profile` |
| DiT ring SDPA chunks | per-instance chunk tables with every q chunk `min(q, 128)` (fp32 accumulation makes the (256, 512) default exceed L1) | `apply_exact_attention_profile` |
| `DistributedLayerNorm` (adaLN with dynamic affine), `DistributedRMSNorm` (q/k norm), `LayerNorm` | HiFi4, fp32 accumulation, `math_approx_mode=False`, `packer_l1_acc=False` | `layers/normalization.py` default |
| time path (the pipeline's) | **host, bit-equal to the reference**: `Flux2KleinHostTime` runs the fp32 sinusoid, the bf16 timestep MLP, the four modulation heads and the consumers' `1 + scale` in bf16 torch, op for op as the diffusers bf16 model (verified bit-equal on `temb` and all 17 chunks against forward hooks, 2026-10-23); the denoiser uploads the 17 + 1 bf16 `[1, 1, D]` chunk tensors ONCE per distinct timestep per process (`_mod_cache`, shared by every layout key) and the fp32 `[1, 1]` scalars `t_embed` / `dt` ONCE per distinct value (`_scalar_cache`), both in `prepare`; an eager step uploads and copies NOTHING (FLUX 3 uploads its step constants per request and its two scalars per step); a traced step refreshes the `t_embed` / `dt` / 17 + 1 `mod` slots by `ttnn.copy` (`Flux2KleinTransformer.forward(mod=...)`). ~0.1 s per distinct timestep per process, memoized | `time_flux2_klein.py`, `weights_dit_flux2_klein.load_time_torch_state`, `denoise_flux2_klein.py` |
| Euler / CFG | fp32 / bf16 typecast chains reproducing `S:493-526` and `P:877` op for op; the state is bf16-valued between steps; the `dt` slot holds `bf16(dt)` because torch casts the scheduler's 0-dim fp32 `dt` to bf16 before `dt * v` (`Flux2KleinStep.dt_rounded`; measured bit-equal to the host update, 2026-10-23) | `denoise_flux2_klein.py`, `scheduler.py` |
| VAE conv2d / GroupNorm / attention | Conv2d HiFi4 fp32; GroupNorm HiFi4 fp32; SDPA HiFi2 fp32 with `exp_approx_mode=False` | `layers/conv2d.py`, `models/vae/vae.py` defaults |
| text encoder (`Qwen3Encoder`) | bf16 linears (`linear_dtype` / `attn_qkv_dtype` = bf16), fp32 accumulation, linear fidelity HiFi2 (`TransformerEncoderConfig.linear_math_fidelity = None`); `FLUX2_KLEIN_TEXT_ENCODER_FIDELITY=hifi4` is the opt-in: 2026-10-23 lever L1 (jobs 146815 -> 146816), device text vs the recorded diffusers text on man_dog (R = 32 real rows): REAL rows rel_rmse 0.0085 -> 0.0007 (pcc 0.999998 -> 1.000000), PAD rows [32, 512) 0.1645 -> 0.1702 (pcc 0.9864 -> 0.9856; the reference's own bf16 sdpa-vs-eager band on the pad rows is rel_rmse 0.11 / pcc 0.994), device-text e2e PSNR 25.26 -> 24.56 dB (with precise exp 26.79 -> 24.59 dB), encode wall 0.182 -> 0.181 s. Not the default: better on the real rows, not neutral-or-better end to end (the pad rows set the e2e number) | `encoders/qwen3/model_qwen3.py`, `encoders/transformer.py` |

`apply_exact_attention_profile(attn, device)` is applied to every `Attention` instance of the DiT (5 | 8 dual and
20 | 24 single blocks) after construction.

### Deviations from the reference

1. **Approximate exp inside the DiT ring attention** (the default; `sdpa_exp_approx=False` /
   `FLUX2_KLEIN_SDPA_EXP_APPROX=0` selects the precise exp; user decision 2026-10-23 23:3x). When enabled it is the only
   approved deviation: the difference between the SFPU's approximate and precise exponential in the softmax.
   Everything else in the attention (QK^T, the softmax statistics, PV) accumulates in fp32 either way.
2. **One bf16 rounding of `1 + scale` in the device time module only.** The device fallback fuses the `+1` of every
   scale chunk into the modulation linear's bias (one rounding instead of the reference's two). The pipeline does not
   take that path: its modulation comes from the host (`Flux2KleinHostTime`), which reproduces the reference's
   `bf16(1 + bf16(acc))` exactly.
3. **fp32 BatchNorm normalisation in the device VAE encoder.** The reference normalises the posterior mean with
   the VAE's own dtype; the device encoder normalises in fp32 and casts the tokens to bf16 (the decode side's
   inverse is applied the reference's way on the host rows).

The exactness rule constrains operand and accumulation dtypes; it does not claim bit equality with a GPU run
(kernel reduction orders differ). The Tests phase pins each stage against a CPU reference.

## Mesh layout

Default: the 4x8 Galaxy with the DiT sequence-parallel over axis 1 (SP8) and tensor-parallel over axis 0 (TP4),
`sp_axis=1, tp_axis=0` (`--layout sp8tp4`), Ring topology, 2 fabric links, 8 KB fabric packets, `FABRIC_1D_RING`
with `STRICT_INIT`, `l1_small_size=65536`, `trace_region_size=256_000_000`, WORKER dispatch cores. Per device at
1024 x 1024: 512 image rows, 64 prompt rows, 6 (4B) | 8 (9B) attention heads.

- The text encoder shares the mesh: TP over the DiT's TP axis, FSDP over the other axis (`text_encoder_fsdp=True`;
  the 27 of 36 Qwen3-4B layers the taps need are 6 GB bf16, 190 MB per chip; its RoPE tables and causal + padding
  biases stay cached on the device, `Flux2KleinPromptEncoder.constants_info`). Its `[1, 512, joint]` bf16 output is
  replicated on every chip and handed to the DiT on the device: `ttnn.mesh_partition(dim=1, cluster_axis=sp_axis)`
  keeps each device's `512 / SP` rows (a per-device slice, exact), the placement the denoiser's hoisted
  `embed_prompt` reads once per request (no 7.9 MB replica read, host tilize or 32 shard writes in the request).
  `Flux2KleinRequest.return_text` (default True; the bench and the demo pass False) adds the one-replica host read for
  `Flux2KleinResult.text`; the empty prompt of the base CFG stays on the host (`encode_host`, cached) and is uploaded
  per request as before.
- The VAE decoder: channels TP4 over axis 0, latent rows H8 over axis 1 (`vae_tp_axis=0, vae_h_axis=1`); the VAE
  encoder: channels TP over the same axis, spatial axes replicated; its image input is uploaded as one row band per
  column of axis 1 and all-gathered on the device (the host used to write all 32 chips: ~70 ms of the 0.28 s per
  1024x1024 reference was host time; measured warm eager on the 4x8 ring: upload wall 68 ms -> 5 ms, encode wall
  276 -> 213 ms per 1024x1024 reference, the encode kernel 0.207 s unchanged, tokens bit-equal; matrix 2048x2048 r1
  encode_refs 0.278 -> 0.216 s, 1024x1024 r3 e2e 1.570 -> 1.365 s).
- `--layout sp4tp8` (SP over axis 0, TP over axis 1) and `--layout dual` (sp4tp8 as the primary layout with sp8tp4
  resident next to it for canvases of `LARGE_LAYOUT_MIN_TOKENS` = 16384 tokens and more) are the alternatives;
  `shard_prompt=True` is required at SP >= 4.

Sequence contract of the DiT (the denoiser's slots): the spatial stream `[image rows | reference rows | zero pad]`
with `L_pad = padded_length(S_img + S_cond, SP)` (a multiple of `32 * SP`), the ring attention masking the pad
rows through `logical_n`; the prompt is a separate stream of exactly 512 rows, SP-sharded under `shard_prompt`, for
the dual blocks. The single blocks run on ONE sequence per device, the per-device concatenation `[prompt_d |
spatial_d]` (one `ttnn.concat` per step, one `ttnn.slice` after the last block), so the pad rows stay the global
tail of the ring-gathered order and `logical_n = 512 + S_img + S_cond`; its RoPE table is the matching interleave
(`rope_flux2_klein.single_stream_rope`, `plan.single_rope`). `shard_prompt=True` (the default) is required for it at
SP > 1. The three tables of a canvas (the plan) are built on the host and uploaded ONCE per distinct (layout, spatial ids,
text ids) and kept by the transformer (`Flux2KleinTransformer.plan_for`, `FLUX2_KLEIN_PLAN_CACHE_SIZE` = 16 LRU
entries, at most ~7.5 MB per device each at SP 8; the prompt pair is shared by every plan); the denoiser's slots
hold references, a miss releases the live traces before it allocates, and `prepare` records the cache's
`plan_cache_hits` / `plan_cache_misses` in the timings. Before: ~2.5 ms (1K) .. 8.5 ms (2K, three references) of
host table building plus two fp32 `[1, 1, L_pad, 128]` uploads per request.

## Request API

```python
@dataclass(frozen=True)
class Flux2KleinRequest:
    prompt: str
    model: str = "distilled"          # "distilled" | "base" (base = CFG 4.0 / 50 steps)
    size: str = "4b"                  # "4b" | "9b"
    width: int | None = None          # None -> the first prepared reference's width, else 1024
    height: int | None = None
    seed: int | None = None           # None -> a fresh 32-bit seed (reported on the result)
    steps: int | None = None          # None -> 4 (distilled) | 50 (base)
    guidance: float | None = None     # distilled accepts None / 1.0 only; base None -> 4.0
    images: tuple = ()                # references: PIL images or paths, at most 10
    output_format: str = "png"
    text_override: tuple | None = None            # tests: ((tokens [512, joint] bf16, ids), (null_tokens, null_ids) | None)
    image_cond_override: tuple | None = None      # tests: (cond_tokens [N, 128] bf16, cond_ids [N, 4] int64)
    return_final_tokens: bool = True
    return_text: bool = True          # read the prompt encoding back (Flux2KleinResult.text); False for serving

result = pipeline.generate(request, traced=True)   # -> Flux2KleinResult
```

`Flux2KleinResult` carries `image` (PIL), `final_tokens` (fp32 `[S_img, 128]`, bf16-valued), `timings`
(`resolve_s` with `reference_cache_hits` / `reference_cache_misses`, `encode_s`, `encode_refs_s`, `prepare_s` with `plan_cache_hits` / `plan_cache_misses`, `vae_capture_s`, `denoise_s`, `decode_s`, `to_host_s`, `total_s`,
`trace_captured`, `vae_trace_captured`), `layout_key`, `model`, `size`, `seed`, `steps`, `guidance`, `admission`,
`text` (None under `return_text=False`), `image_cond`, `references`, `input_prompt` and `layout`. `request.resolve()` applies the defaults and
prepares the references; `admit(resolved, mesh_shape)` runs on the host before any device work and raises
`Flux2KleinAdmissionError` on a refused canvas.

Numerical contract reproduced on the way (each a docstring citation in the module that owns it):

- noise: `torch.randn((1, 128, H/16, W/16), generator=Generator().manual_seed(seed), dtype=bfloat16)` packed row-major;
- ids: latents `(0, h, w, 0)`, reference `i` `(10 + 10 i, h, w, 0)`, text `(0, 0, 0, l)`;
- schedule: `sigmas = linspace(1, 1/N, N)` shifted by `exp(mu) / (exp(mu) + (1/sigma - 1))` with
  `mu = compute_empirical_mu(S_img, N)`, `timesteps = sigmas * 1000`, terminal 0 appended;
- timestep seen by the DiT: `bf16(bf16(bf16(sigma * 1000) / 1000) * 1000)`;
- Euler: `prev = fp32(sample) + dt * v` with `dt * v` in bf16, then `prev.to(bf16)`;
- CFG (base): `neg + g * (pos - neg)` with the `""` prompt encoded through the same text path;
- readback: `(x / 2 + 0.5).clamp(0, 1) * 255` rounded (not truncated) to uint8.

## Weights and environment variables

| variable | meaning |
|---|---|
| `FLUX2_KLEIN_MODEL_PATH` | an explicit snapshot directory, or a root holding `FLUX.2-klein-4B/` (repository name) or `distilled-4b/` (`<variant>-<size>`) children; ignored for a model it identifiably does not name |
| `HF_HUB_CACHE`, `HF_HOME` | the HuggingFace cache searched next (`refs/main` first, then the newest complete snapshot; a snapshot holding only `LICENSE.md` / `README.md` is skipped) |
| `TT_DIT_ALLOW_HF_DOWNLOAD=1` | opt into downloading a missing snapshot (the BFL-layout `flux-2-klein-*.safetensors` twin and the sample images are skipped) |
| `TT_DIT_CACHE_DIR` | the converted-weight cache: `<root>/<model_name>/<subfolder>/<config_id>mesh4x8_bf16[_FSDP]/`; the DiT under `flux2_klein_<variant>_<size>_<snapshot>/transformer_D<dual>_S<single>/TP4_0_SP8_1_mesh4x8_bf16/`, the text encoder under `flux2_klein_text_encoder_<blob[:16]>/text_encoder_L27/` (the 27-layer build the pipeline runs; a full-depth build uses `/text_encoder/`; one entry per mesh layout serves base and distilled of a size), the VAE under `flux2_klein_vae_<blob[:16]>/vae/` and `/vae_encoder/` |
| `TT_DIT_CACHE_VERIFY=1` | re-read a freshly written cache before marking it complete |
| `FLUX2_KLEIN_VAE_SLICE_MODE` | `auto` (default; ttnn's conv auto slicer), `aligned` or `explicit` slice tables |
| `FLUX2_KLEIN_VAE_CONFIG_TENSORS_DRAM` | `1` (default): keep the conv halo / sliding-window config tensors in DRAM instead of L1_SMALL |
| `FLUX2_KLEIN_CONV_PREP_CACHE` | `per_shape` (default): `layers/conv2d.py` keeps one prepared weight / bias copy per input-shape key (batch, h, w, memory config, layout, dtype, slice config, config overrides); `single`: the old one-copy-per-module path (prepared on the first forward, reused for every later shape), kept for the regression probe `test_vae_flux2_klein.py::test_decode_prepared_per_shape` only |
| `FLUX2_KLEIN_SDPA_EXP_APPROX` | `1` (default, user decision 2026-10-23) the approximate SFPU exp in the DiT ring SDPA, `0` the precise exp (free at 1K, -17 % denoise at 2K, see the exact-profile table); the `Flux2KleinPipeline(sdpa_exp_approx=...)` keyword wins over it |
| `FLUX2_KLEIN_TEXT_ENCODER_FIDELITY` | matmul fidelity of every Qwen3 linear: `default` (the generic HiFi2) or `hifi4`; the `Flux2KleinPipeline(text_encoder_math_fidelity=...)` keyword wins over it |
| `FLUX2_KLEIN_DIT_FIDELITY` | matmul fidelity of every DiT linear (the ring SDPA compute stays HiFi4): `hifi4` (= `default`, the block module's `FLUX2_KLEIN_DIT_MATH_FIDELITY_DEFAULT`) or `hifi2` (the layers/linear.py default FLUX 3 runs on; the A/B numbers are in the exact-profile table); anything else is refused; the environment wins over the `Flux2KleinPipeline(dit_math_fidelity=...)` keyword, which wins over the default |
| `FLUX2_KLEIN_RECORD_MEMORY` | `1` ends every per-request `RECORD generate` line with the mesh memory view (`dram_per_bank`: two `ttnn.get_memory_view` calls inside `total_s`); `0` (default, `RECORD_MEMORY_DEFAULT`) leaves it to the init-time RECORD lines; the `Flux2KleinPipeline(record_memory=...)` keyword is the fallback below the environment |
| `FLUX2_KLEIN_TEXT_ENCODER_FSDP` | `1` the text encoder's weights FSDP-sharded over the mesh axis its TP does not use (every linear all-gathers its weight shard on every encode: 5 collectives per layer, 135 per encode), `0` replicated over that axis (no weight gathers, +1.2 GB per chip on TP4, bit-equal encodings); unset = `TEXT_ENCODER_FSDP_DEFAULT`; the `Flux2KleinPipeline(text_encoder_fsdp=...)` keyword is the fallback below the environment |

The weights are gated on HuggingFace: accept the licence on the model page and log in (`huggingface-cli login`)
before a download. The port reads the diffusers layout only: `model_index.json`, `transformer/config.json` +
`transformer/diffusion_pytorch_model.safetensors`, `text_encoder/`, `tokenizer/`, `vae/`, `scheduler/`
(`weights_flux2_klein.REQUIRED`).

## Running

From the repository root with the tt-metal environment active:

```bash
# distilled 4B, 1024 x 1024, 4 steps, no CFG (the default)
python models/tt_dit/pipelines/flux2_klein/demo_flux2_klein.py "a lighthouse on a basalt shore at dusk" \
    --model distilled --size 4b --width 1024 --height 1024 --seed 0 --out out.png

# base 4B: 50 steps, CFG 4.0
python models/tt_dit/pipelines/flux2_klein/demo_flux2_klein.py "a lighthouse on a basalt shore at dusk" \
    --model base --size 4b --steps 50 --guidance 4.0 --seed 0 --out out_base.png

# image editing: every --image is a reference (at most 10); the canvas defaults to the first reference's size
python models/tt_dit/pipelines/flux2_klein/demo_flux2_klein.py "make it night, keep the composition" \
    --image photo.png --seed 0 --out edited.png

# eager (untraced) run, the large-canvas layout, repeated renders, an explicit snapshot
python models/tt_dit/pipelines/flux2_klein/demo_flux2_klein.py "..." --eager --layout dual --repeat 3 \
    --weights /path/to/snapshot
```

Flags: positional `prompt` or `--prompt-file`, `--model {distilled,base}`, `--size {4b,9b}`, `--image` (append),
`--width`, `--height`, `--steps`, `--guidance`, `--seed`, `--out` (default `out.png`), `--weights`, `--eager`,
`--layout {sp4tp8,sp8tp4,dual}`, `--repeat`. Exit status 2 on bad input, 3 on a refused canvas. Every render writes
`<out>.json` with the model, size, prompt, canvas, steps, guidance, seed and reference sizes.

Programmatic use:

```python
from models.tt_dit.pipelines.flux2_klein.pipeline_flux2_klein import Flux2KleinPipeline, Flux2KleinRequest

pipeline = Flux2KleinPipeline.create(mesh_device, size="4b", resident=("distilled",))   # resolves the snapshot
result = pipeline.generate(Flux2KleinRequest(prompt="...", seed=0))
result.image.save("out.png")
pipeline.close()
```

## Tests

`tests/models/flux2_klein/` (the Tests phase): one set of CPU-reference tests per stage against the diffusers
modules on the 4B snapshot, with the device parameters of `tests/models/flux2_klein/device_params.py`
(`L1_SMALL_SIZE` 65536, `TRACE_REGION_SIZE` 256_000_000, the 8K ring fabric parameters).
