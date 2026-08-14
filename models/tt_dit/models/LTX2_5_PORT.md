# LTX-2.5 Port Plan

Status and plan for bringing [LTX-2.5](https://github.com/Lightricks/LTX-2) (upstream
`v1.2.0`) into TT-DiT alongside the existing LTX-2.3 port. For running LTX-2.3, see
[LTX2.md](LTX2.md).

## Status

**Distilled T2V is the bring-up target.** Text path + distilled pipeline + stage-1 ancestral
Euler are in, and the DiffVAE decoder now matches upstream on device though it is not yet wired
in; I2V and DFR are out of scope until T2V is solid.

| Component | State |
|---|---|
| Gemma-4 encoder (`encoders/gemma4/`) | Done — 48 layers on shipped weights, PCC-verified |
| Encoder pair, tokenizer, projection | Done — prompt to video/audio embeddings end to end |
| Feature extractor, connectors | Reused from `gemma3/` unchanged, verified against 2.5 weights |
| Distilled T2V pipeline | `pipeline_ltx25_distilled.py` + `test_pipeline_ltx25_distilled.py` — Gemma-4, split DiT/audio/upsampler, stage-1 ancestral |
| Conv video VAE | Loader accepts monolith `vae.decoder.*` and bare `decoder.*` / PCS; prefer `*-conv-bf16` (gated HF — falls back to 2.3 monolith) |
| Transformer config flags | `ff_bias` auto-detected from checkpoint; keyframes abs-pos dropped on load (DFR later) |
| Audio VAE | Split `*-audio-vae-bf16` wired (decoder + vocoder prefixes match loader) |
| Duration head | Not started, optional |
| DiffVAE video decoder | Decodes latent to pixels on device, PCC 99.99 % vs upstream on shipped weights; **not yet wired into the pipeline**, and untested above 768×512 |
| I2V / DFR | Deferred |

## Structure

**Rule:** share a file only when it is identical; otherwise give the new version its own
file (even temporarily) so later versions do not grow into `if version` forests inside
2.3 code. Deduplicate into helpers later once a second/third version is real.

| Piece | Layout |
|---|---|
| Gemma encoders | Split — `encoders/gemma3/` (2.3), `encoders/gemma4/` (2.5) |
| Pipelines + checkpoint wiring | Split — `pipeline_ltx*.py` (2.3), `pipeline_ltx25_*.py` (2.5) |
| Conv video VAE / audio VAE **classes** | Shared — arch/dispatch identical; do not fork the hand-tuned conv3d/halo tables |
| Transformer | Soft — same `LTXModel` class for now; version flags / thin wrappers as needed |

An earlier full `ltx2_3` rename covering pipelines, VAE, audio, upsampler and patchifiers
was abandoned; it is kept on `backup/ltx-full-rename-2026-08-12` if ever needed.

Distilled 2.5 scaffold: `pipelines/ltx/pipeline_ltx25_distilled.py` +
`tests/models/ltx/test_pipeline_ltx25_distilled.py`.

## Gemma-4 encoder

Verified against `models/demos/gemma4/` and the actual checkpoint tensors, not inferred.
The backbone is shared with Gemma-3-12B (hidden 3840, 48 layers, 16 heads,
`intermediate_size` 15360, `gelu_pytorch_tanh`, the 4-norm sandwich, the 6-layer
interleave), as is the `ColParallelLinear` / `nlp_create_qkv_heads` /
`rotary_embedding_hf` scaffolding.

| Concern | Gemma-3 | Gemma-4 |
|---|---|---|
| RMSNorm weight | `(1 + weight)` fold | raw `weight`, no fold |
| Attention scale | `1/sqrt(head_dim)` | 1.0 |
| `head_dim` | 256 everywhere | 256 sliding / 512 global |
| KV heads | 8 | 8 sliding / 1 global (MQA) |
| V projection | `v_proj` | global layers have no `v_proj`; V = K |
| V norm | none | per-head RMS, no learnable scale |
| Global RoPE | theta 1e6, linear scaling 8.0 | theta 1e6, proportional, no scaling |
| Sliding RoPE | theta 1e4, no scaling | identical — reused as-is |
| Per-layer scalar | none | `layer_scalar`, multiplies the layer output |
| Key prefix | `language_model.model.` | `model.` (flat) |

Only the 8 `full_attention` layers (indices 5, 11, …, 47) diverge structurally. Two audit
concerns died against the checkpoint: `num_kv_shared_layers=0` (no cross-layer KV sharing)
and `use_double_wide_mlp=false`.

### Proportional RoPE

From `transformers==5.10.1` `_compute_proportional_rope_parameters`. The tt-metal env is
pinned at 4.53.0, which has neither `gemma4` nor the `proportional` rope type, so this was
read from a downloaded copy rather than an installed one:

```
head_dim    = global_head_dim = 512      # head_dim_key override on full_attention
rope_angles = int(0.25 * 512 // 2) = 64
inv_freq    = 1/(1e6 ** (arange(0, 128, 2) / 512))   # 64 real frequencies
inv_freq    = cat([inv_freq, zeros(192)])            # pad to head_dim/2 = 256
```

The zero frequencies give `cos=1, sin=0`, so those dimensions pass through unrotated while
the encoding stays full `head_dim`. No new kernel is needed — `rotary_embedding_hf` works
unchanged and only the `inv_freq` table differs. Compute `inv_freq` in float32: float64
crosses bf16 rounding boundaries and is not bit-identical to the reference.

### Weight loading

K=V tying and the single global KV head are both load-time concerns; the runtime split
stays plain `nlp_create_qkv_heads`. On global layers `v_w = k_w`, taken from the
*pre-`k_norm`* k_proj output. When `num_key_value_heads < tp`, each device keeps a full KV
head chosen GQA-aware via `kv_idx = (i * q_per_device) * num_kv_heads // num_q_heads`; with
1 global KV head every device gets index 0. Order inside the layer is project, split heads,
norm Q/K/V, RoPE on Q and K only, SDPA at scale 1.0, `layer_scalar` last.

### Sliding-window masking

The port never builds sliding masks — it runs causal-plus-padding for all layers and only
varies the RoPE table. With `sliding_window=1024` that stays correct for any prompt at or
under 1024 tokens, which is guarded explicitly rather than left to chance.

### Parity method

No Gemma-4 exists in transformers 4.53, so the reference is generated in a throwaway venv
on 5.10.1 and handed over as safetensors (`gen_gemma4_reference.py`), sidestepping a torch
version gap that a pickle would not survive.

**bf16 has a floor, and it is far below 0.99.** The first parity run read 93.17 % at the
output, decaying smoothly from 99.9999 % at the embedding. Running HF's *own*
implementation in bf16 against its own fp32 output decays to 92.99 % — slightly worse than
ours at every stage, because ttnn accumulates matmuls in fp32 where torch's CPU bf16 path
does not. A fixed 0.99 target is unreachable at this depth and would send someone hunting a
phantom, so the test asserts against the measured bf16 floor instead. A structural bug
falls far below it while dtype noise tracks it.

Random weights are the pessimal case: two *trained* layers at full width give 99.98 %, so
the decay is a property of the random init's conditioning rather than of depth.

## Text path reuse

Both claims tested, not assumed:

- **Feature extractor** — 2.3's `GemmaFeatureExtractor` is already FeatureExtractorV2, and
  Gemma-3-12B and Gemma-4-12B share hidden 3840 × 49 states, so `input_dim` is 188160 for
  both. Against the real 2.5 `text_embedding_projection.*` weights: 99.999 % on both axes.
- **Connectors** — the 2.5 transformer checkpoint's 258 connector tensors are structurally
  identical to 2.3's: same keys, same shapes, values aside. `EmbeddingsConnector` carries
  over verbatim.
- Hidden-state selection is unchanged: LTX wants HF's 49 (`num_hidden_layers + 1`), which is
  our 50 minus index -2, exactly as the 2.3 pair already does.

The encoder pair exists only because the weights moved:

| what | 2.3 | 2.5 |
|---|---|---|
| gemma weights | HF directory | packed text-encoder file |
| `text_embedding_projection.*` | monolithic LTX checkpoint | packed text-encoder file |
| connectors | monolithic LTX checkpoint | transformer checkpoint |
| tokenizer | HF directory | `tokenizer_json` blob inside the packed file |

The bundled `gemma4-12b-with-proj` safetensors is self-sufficient — HF config in file
metadata, tokenizer and sidecars as embedded tensors, projection weights included — so 2.5
needs no separate gated Gemma download, removing a blocker 2.3 had.

Two tokenizer traps:

- **Gemma-4's tokenizer has no BOS post-processor**, where Gemma-3's does. Upstream prepends
  BOS by hand and so must we; a plain `tokenizer(prompt)` call silently drops it and shifts
  every hidden state. Against the shipped tokenizer, a bare encode starts at 236746, not BOS.
- The packed `tokenizer_config.json` carries transformers-5 spellings
  (`extra_special_tokens` as a list, `model_specific_special_tokens`) that the pinned 4.53
  rejects as kwargs. Both are vocab tokens regardless, so skipping the keys costs only an
  attribute alias.

## Remaining work (distilled T2V)

| | State | Notes |
|---|---|---|
| Pipelines + split loading | Landed | Paths, Gemma-4, generate; smoke on 4×8 (`4x8sp1tp0nl2_ring_is_fsdp0`). |
| Stage-1 ancestral Euler | Landed | On-device ttnn math; host only draws seeded noise (`seed+10000`) and uploads. |
| Video decode is 2.3-era | Decoder built, not yet wired | The pipeline still calls the 2.3 monolith's conv decoder. Safe for correctness — the shipped 2.5 VAE's 84 encoder tensors and both `per_channel_statistics` are **byte-identical** to 2.3's, so the latent space is unchanged — but it forgoes 2.5's decode quality. The distilled recipe on the model card passes `--video-vae-path .../ltx-2.5-video-vae-bf16.safetensors`, the **DiffVAE**, and never lists `*-conv-bf16` (which is why the local mirror lacks it): the gated conv file was never the target. That decoder now exists on device (see [DiffVAE](#diffvae-video-decoder)); what remains is pipeline wiring and resolutions above 768×512. Do not load DiffVAE weights into the conv decoder: different architectures. |
| Transformer `ff_bias` | Landed | Auto from checkpoint metadata. |
| Audio VAE | Landed | Split file loads on device; all 1329 tensors (`audio_vae.*` + `vocoder.*`) byte-identical to 2.3. |
| Spatial upsampler | Landed | 2.5 spatial-x2 path; all 72 tensors byte-identical to 2.3. |
| Duration head | Optional | Empty in shipped config. |
| I2V CRF 18 / DiffVAE / DFR / keyframes | Deferred | Not needed for distilled T2V. |

Weight-level diff of the shipped files: the only genuinely new 2.5 weights on the distilled T2V
path are the **DiT** and the **Gemma-4 text encoder + projection**. Video-VAE encoder, latent
statistics, audio VAE, vocoder and spatial upsampler are all byte-identical to 2.3 (the new
DiffVAE decoder and temporal upscaler are unused here).

Checkpoint headers settled the expensive questions in the cheap direction:

- **Conv VAE config is byte-identical to 2.3** — `patch_size=4`, `latent_channels=128`,
  `spatial_padding_mode="zeros"`, `timestep_conditioning=false`, no `attn` blocks. No
  `AttnBlock3D`, no timestep conditioning, no reflect padding, and the port's hardcoded
  32×/8× and `base_channels * 8` are correct for 2.5.
- **Transformer dimensions are unchanged** — 48 layers, 32 heads, `attention_head_dim=128`,
  `cross_attention_dim=4096`, `in_channels=128`. No re-tuning of SDPA chunks, AG-matmul
  configs, caches or traces. `cross_attention_adaln` and `apply_gated_attention` are already
  auto-detected by our loader.
- **Audio VAE config is identical to 2.3** — same `ddconfig`, same STFT, same 16 kHz.

`gemma_source_checkpoint = {"gemma_version": "gemma4-12b-ltx-v1", "ltx_version": "2.5.0"}`,
so the compatibility check will hard-require the Gemma-4 encoder.

### Deferred

The DFR pipeline and its temporal upsampler.

DiffVAE was deferred here on a 2-3 month estimate; that estimate was wrong by a wide margin
and the decoder is built. Two assumptions behind it are worth correcting, because they are the
reason it looked expensive: the plan expected to ship on the conv decoder, but the distilled
recipe does not use it and the gated file was never obtainable; and NATTEN's fused kernel
looked irreplaceable, when in fact upstream's own natten-free fallback shows the window is
expressible as masked attention over grouped query tiles, which ttnn already has. The
observation that the shipped config is *heavier* than the class defaults still stands
(`stage_channels=[2048, 1024, 512, 512, 256]`, and `model_output_type="x0"` rather than the
`"v"` default) — it just did not turn out to be the deciding cost.

## DiffVAE video decoder

2.5's video decoder is not a convnet. Every one of its 24 blocks is 3D neighborhood attention
over a local window, so the port hinges on one primitive rather than on conv3d halo tuning:
`layers/na3d.py`. Files are `models/vae/diffvae_ltx.py` (deterministic stages 1-4, plus the
composed `DiffVAEDecoder`) and `models/vae/diffvae_ltx_stage5.py` (the diffusion stage).

Shape of the thing, read from the checkpoint's own metadata rather than hardcoded
(`decoder_config`): `stage_channels=[2048, 1024, 512, 512, 256]`, `stage_depths=[4, 6, 4, 2, 8]`,
`stage_kernels=[(3,7,7), (3,7,7), (3,5,5), (3,5,5), (11,11,11)]`, `head_dim=64`, `patch_size=4`.
The first four entries are the deterministic stages (16 blocks, 97.7 % of the 417 M parameters);
the trailing 256-channel 8-block entry is the diffusion stage, which owns almost none of the
weights and almost all of the compute — it runs on the largest grid with a 1331-element window.

### Neighborhood attention as masked attention

NATTEN keeps the window *size* constant and shifts it inward at boundaries, so a query at
index 0 attends to `[0, K)`, not to a truncated `[0, K//2]`. A truncating window looks
plausible and is wrong everywhere near an edge. Because the rule is that regular, an axis has
only three regimes and a whole volume collapses to at most 27 distinct masks in 3D no matter
how many tiles it has — so `plan_na3d` groups query tiles by window geometry and each group
becomes one batched SDPA call sharing one additive mask.

- **bfloat16 only.** The gathers are `ttnn.embedding`, which validates that its table is
  bfloat16. Lifting this means replacing the gather, not casting — a quiet downcast would let
  an fp32 caller believe it had fp32 attention. Asserted at the top of the primitive.
- **Plans are cached** (`cached_device_plan`) on grid, kernel and dtype. They depend on no
  weights, but they upload index tables and masks, so rebuilding per block would dominate.
- **Waste is the tuning knob.** The dense formulation evaluates more scores than an exact
  kernel because a tile's key span covers the union of its queries' windows. Current tile
  search grows tiles until they hit a score budget, which minimises call count and *maximises*
  waste: 2.9x on stage 5's big kernel but **10-26x on the deterministic stages**, where a
  length-10 tile reads 14 keys to serve a 5-wide window. Harmless at 768×512, worth fixing
  before full res.

### RoPE: reordered, not reimplemented

Upstream rotates adjacent dim pairs within each axis chunk (for `head_dim` 64 the split is
`(16, 24, 24)`: T rotates dims 0-15, H 16-39, W 40-63). Interleaved pairs would need a stride-2
gather per rotation. Attention only sees `q·k`, so permuting `head_dim` identically in q and k
is invisible in the output — reorder to `[all first-of-pair, all second-of-pair]` and RoPE
becomes the contiguous `(x1*cos - x2*sin, x1*sin + x2*cos)` with a single width-32 table. The
permutation folds into the q/k projection rows and the `q_norm`/`k_norm` weights at load, so it
is free at runtime, and it is **bit-identical** to upstream (verified, not approximated).
RMSNorm tolerates it because its scale is over all dims, hence permutation-invariant, provided
its learned weight is permuted the same way.

`rope_num_tiles=4` is arithmetically a **no-op** (exactly 0 difference): upstream slabs W only
to keep Dynamo from specializing on shape, and carries absolute offsets across slabs.

### Verified against upstream, on shipped weights

Ground truth comes from `capture_stages.py`, which drives upstream's own decoder one stage at a
time with the shipped checkpoint and *injected* noise — stage 5 predicts x0 from that noise in a
single step, so the noise is an input, and matching pixels requires the reference's own draw
rather than a reseed. It bypasses tiling deliberately, so each stage compares against one
contiguous tensor.

| What | PCC |
|---|---|
| NA3D device vs host executor (5 shapes incl. axis-shorter-than-kernel) | > 99.9 % |
| `NABlock`, blocks 0 and 1, real activations | 99.985 %, 99.987 % |
| `conv_in` + 14 blocks + 3 upsamples | 99.976 % |
| Latent → stage-5 context (4 stages + ghost pad + crop) | 99.988 % |
| Stage 5, per block (8) then pixels, synthetic weights | 99.995 % → 99.993 % |
| **Latent → pixels, whole decoder, 320×320** | **99.985 %** |
| **Latent → pixels, whole decoder, 768×512** | **99.994 %** |

bfloat16 error does not compound with depth: one block is 99.985 %, fourteen blocks plus three
upsamples is 99.976 %, and the joined 24-block decoder is 99.99 %. Host-side exactness checks
(`ltx25_diffvae/check_rope.py`, `check_upsample.py`, `check_na3d_plan.py`) cover the pieces where
a silent error is possible: RoPE and pixel shuffle are exact to 0.0, plan geometry to < 1e-5.

### Traps

- **`per_channel_statistics` folds into `conv_in`.** The decoder's first act is
  `conv_in(x * std + mean)`, exactly a Linear with `std` scaled into the weight columns and
  `W @ mean` added to the bias. Free and exact, so the decoder takes the same normalized latent
  the conv decoder takes — but it means `conv_in`'s loaded weights are not the file's weights.
- **The ghost pad and crop are one workaround in two halves.** Before stage 1 the last latent
  frame is replicated `(stage_kernels[0][0] // 2) * 2 = 2` times for NATTEN's trailing border;
  after stage 4 that appendix is cropped back off (`2 * 8 = 16` pixel frames). Apply the pad
  without the crop and the video grows 16 spurious frames. Frame arithmetic for a 4-frame
  latent: 4 → 6 padded → 41 through four upsamples → 25.
- **Pixel shuffle packs channels outermost.** `(c p1 p2 p3)`, so the shuffle is a
  reshape-then-transpose, not a view. Wrong factor order still yields the right shape with
  plausible statistics — it scrambles space against channels and survives to the video as mush.
- **Each temporal upsample emits a duplicate leading frame** that must be dropped, and only for
  the chunk holding the true t=0. A tiled caller decoding a later chunk must pass
  `drop_leading_frame=False`.
- **`model_output_type` lives at the top of the vae config, not under `decoder`**, but the
  constructor takes it — and its default `"v"` silently changes the final step from "return x0"
  to an Euler update. This checkpoint is `"x0"` with one inference step, so a single pass is the
  whole decode.
- **`decoder.type_emb`** is in the checkpoint and referenced nowhere in upstream's source
  (repo-wide search): vestigial, not something we fail to apply.
- **Static gates are pre-folded** into `attn.proj` / `mlp.w_down` / `context_proj` at export, so
  the shipped file has no gate tensors and `AdaLNZero`'s 7 chunks are computed with 3 discarded.
  The loader rejects a checkpoint that *does* carry them rather than silently decoding wrong.

### Remaining

Not wired into the pipeline yet, and untested above 768×512. Full res is 3.26 M stage-5 sites
(vs 614 k at 768×512); memory has scaled without trouble so far, which is worth confirming
before building tiling, because upstream's halos are punishing: tiling happens on the stage-4
input grid — full res `(21, 136, 240)` — with a one-sided halo of **(22, 24, 24)** (stage-4
depth × kernel plus stage-5's 8 × 5, scaled by `upsample3_stride`). The T halo alone exceeds
the 21-frame axis, so a 25-frame video cannot be split temporally at all, and a 2×2 spatial
split would compute ~2.4x the untiled work. Upstream runs stages 1-3 on the full volume and
tiles only stages 4-5, with pixel blending on overlaps. Nothing is timed yet.

Three upstream inconsistencies found while reading, worth filing:
`AdaLNZero`'s docstring claims zero-init identity behaviour that `DiffusionNABlock` undoes (it
discards all three gate chunks and calls `reset_parameters()` on the output projection);
`combined/attn.py` documents a `w_chunks` contract its `full()` path never reads; and
`ops.unpatchify` lacks `patchify`'s rank guard, so a 6D tensor passes through untouched.

## Text path: what is verified

An extended investigation into apparently weak prompt adherence ended in a **stale cache**, not a
port defect. The A/B runs from that hunt are not recorded here: any comparison that hit the cache
is uninterpretable, so treating them as evidence would mislead. What survives is the work that
read weights or ran deterministic parity tests, which is worth keeping so it is not re-derived:

- **The whole text path, numerically.** `test_gemma4_parity::test_gemma4_full_stack_matches_huggingface`
  holds all 48 layers against HF on a padded, masked real prompt (post-norm PCC 99.9997 %), and
  `test_gemma4_text_path` holds the aggregate projection plus both connectors against diffusers'
  `LTX2TextConnectors` on 2.5 weights (video 99.9849 %, audio 99.9899 %). The second one is what
  clears `_weight_to_layer_major`: the checkpoint packs the 49 states D-major (measured directly —
  grouping the 188160 columns strided by 49 recovers a per-layer norm profile, grouping them in
  contiguous 3840-blocks gives a flat one), and a permute that disagreed could not hold 99.98 %.
- **Tokenization.** Upstream `LTX2Pipeline._get_gemma_prompt_embeds` encodes the raw prompt with
  no chat template or system wrapper, left-padded, `add_special_tokens=True`, and passes
  `scale_factor=8` to the connectors — all matching this port. The packed tokenizer's
  post-processor injects no special tokens, so `tokenize` adds BOS by hand; local HF Gemma-3
  tokenizes the same text to identical ids including that BOS, so the hand-added token is right.
- **HF architecture choice.** The packed config says `model_type: gemma4_unified_text` while our
  reference builds `Gemma4TextModel`. On this config the two are bit-identical (PCC 1.000000, max
  abs diff 0, neither with missing or unexpected keys), because the plain model's extra features
  (MoE, per-layer embeddings, shared KV) are all disabled here.
- **`keyframes_abs_pos_embedding`.** A single `[1, 4096]` vector, not a positional table, so it is
  a keyframe marker added at keyframe tokens; dropping it on the T2V path is right.
- **Checkpoint integrity.** All six 2.5 files are structurally complete (header + tensor extents
  equal file size exactly), so nothing is truncated. Upstream's `DistilledPipeline` expects
  exactly the two files we use.

One property of the shipped encoder is worth knowing before probing it: activation scales are
folded into its norms (`model.norm.weight` reaches +600, layer 0's `input_layernorm` spans
-143..+193), so its final residual stream sits at norm ~2963 versus ~132 for Gemma-3, dominated
by a few channels. A tied LM-head probe therefore predicts one constant token (0 % next-token
accuracy where Gemma-3 scores 44 %), and raw cosine between prompts runs high for the same
reason. Neither is a defect — the feature extractor's per-token RMS norm removes the scale.

Still unverified, though no longer suspected: **weight provenance.** We read the projection and
connectors out of the distilled DiT file, while upstream's diffusers layout keeps `connectors/`
as a *shared* component stored apart from `transformer/`. Our parity tests prove the math is
right **given those weights**, not that they are the tensors upstream loads. Settling it needs a
diff against `Lightricks/LTX-2.5-Diffusers` `connectors/` (6.3 GB), gated for the local token.

Debug levers left in the code from that investigation — `LTX25_TEXT_STACK=gemma3`, `LTX25_BOS`,
`LTX25_TEXT_PAD_SIDE` — are now dead weight and a footgun (the first silently swaps in the 2.3
text stack). Remove them unless a use appears.

## Correctness traps

- **Image-conditioning CRF (I2V only).** `_PARAMS_SINCE_VERSION` has no 2.5 row, so a 2.5
  checkpoint inherits the 2.4 row and gets **CRF 18, not 33**. Our port still hardcodes 33 —
  fix when I2V is brought up; irrelevant to distilled T2V.
- **Euler-ancestral sampler (landed for distilled).** Stage 1 only, seeded `seed + 10000`;
  stage 2 deterministic (`use_ancestral_sampler=True` on `LTX25DistilledPipeline`). The step
  math runs on device (`_euler_ancestral_step_tt`); the host only draws the seeded noise and
  uploads it, and the result is `ttnn.copy`'d back into the latent so the address-baked
  `inner_step` trace stays valid. No parity test yet for the tt step vs the host reference
  `_euler_ancestral_step` — worth adding.
- **Ancestral branch ignores frame-0 pinning.** The `ancestral` path in `_denoise_no_guidance`
  does not call `_post_process_latent_tt`, so an image-conditioned stage 1 would silently drop
  its conditioning. T2V is unaffected; wire the pin (or assert) when 2.5 I2V is brought up.
- **`use_prompt_adaln_single`.** Absent from the 2.5 config, so it defaults true — same as
  2.3, no work. Separately, our `None` branch skips the static `prompt_scale_shift_table`,
  which is not upstream's `false` behaviour.
- **Video VAE keys (landed).** Loader accepts `vae.decoder.*` and bare `decoder.*` (+ PCS
  both spellings). Audio/transformer still expect monolith-style prefixes (split files ship
  them).

Two audit-stage worries that turned out not to exist: multishot generation is absent from
2.5 (repo-wide search returns zero hits; the adjacent feature is generated interior keyframe
slots, which is intra-shot), and the changelog's video/audio CFG/STG and guidance-rescaling
items are trainer-only.

## Performance notes

`get_matmul_config` warns for 15 distinct shapes in the 2.5 text path, which reads like a
standing perf gap. It mostly is not. A 36-way blocking sweep on `(1024, 3840, 1920)` — the
FFN gate/up shape, the single largest FLOP contributor at two per layer across 48 layers —
had the default 8×8×8 win outright at 0.120 ms, with nothing beating it.

**That sweep is dangerous on a Galaxy box and should not be repeated as written.** Hammering
many candidate blockings across all 32 chips knocked 8 of them off the PCIe bus; the broker
killed the job and per-tray BMC resets did not recover them (it holds rather than doing a
galaxy reset below the 16-chip floor). If the remaining shapes are ever worth measuring, do
it on a single chip, validate each config's L1 footprint first, and expect little.

## Test and mesh gotchas

- **Use a fresh `TT_DIT_CACHE_DIR` for every run.** Reusing a warm cache is not currently
  safe: entries are keyed on names that do not capture every shape-affecting change, so a
  stale entry is served for a tensor whose layout has moved. It surfaces loudly when the
  shapes disagree (`shape mismatch: expected (4096, 3072), got (4096, 3104)` on
  `transformer_blocks.0.attn1.to_qkv.weight.tensorbin` — a tile-padding delta), and
  silently when they happen to agree, which is the dangerous case: an extended prompt-
  adherence investigation was ultimately chasing exactly that. Pay the reload.
- The device broker reaps a job after 300 s of silence, and piping pytest through `tail`
  buffers everything — including the framework's own 5 s progress heartbeat — so the job
  looks dead and gets killed. Don't pipe.
- On a 32-chip box `require_exact_physical_num_devices` forces the full 4×8; a 2×4 submesh
  fails fabric router sync at mesh open, before any model code runs.
- Full 12B encode is 136 s on 4×8, 45 s of which is loading Gemma (~50 s of a cold run is
  converting 530 device tensors).
