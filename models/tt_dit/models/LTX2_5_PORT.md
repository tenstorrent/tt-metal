# LTX-2.5 Port Plan

Status and plan for bringing [LTX-2.5](https://github.com/Lightricks/LTX-2) (upstream
`v1.2.0`) into TT-DiT alongside the existing LTX-2.3 port. For running LTX-2.3, see
[LTX2.md](LTX2.md).

## Status

**Distilled T2V is the bring-up target.** Text path + distilled pipeline + stage-1 ancestral
Euler are in; I2V / DiffVAE / DFR are out of scope until T2V is solid.

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
| I2V / DiffVAE / DFR | Deferred |

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
| Conv video VAE file | Blocked on HF | Gated `*-conv-bf16`; interim = 2.3 monolith (arch-identical). Do **not** use DiffVAE `video-vae-bf16`. |
| Transformer `ff_bias` | Landed | Auto from checkpoint metadata. |
| Audio VAE | Landed | Split file loads on device. |
| Spatial upsampler | Landed | 2.5 spatial-x2 path. |
| Duration head | Optional | Empty in shipped config. |
| I2V CRF 18 / DiffVAE / DFR / keyframes | Deferred | Not needed for distilled T2V. |

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

DiffVAE (2-3 months) and the DFR pipeline. The `-conv-bf16` checkpoint is a complete decode
path selected automatically from checkpoint metadata, so baseline 2.5 needs neither. Note
the shipped DiffVAE is *heavier* than the audit assumed — `stage_channels=[2048, 1024, 512,
512, 256]`, double the upstream class defaults, and `model_output_type="x0"` rather than the
`"v"` default — which strengthens the case for shipping on the conv decoder first. The
temporal upsampler is DFR-only and deferred with it.

## Correctness traps

- **Image-conditioning CRF (I2V only).** `_PARAMS_SINCE_VERSION` has no 2.5 row, so a 2.5
  checkpoint inherits the 2.4 row and gets **CRF 18, not 33**. Our port still hardcodes 33 —
  fix when I2V is brought up; irrelevant to distilled T2V.
- **Euler-ancestral sampler (landed for distilled).** Stage 1 only when
  `model_version >= (2,5)`, seeded `seed + 10000`; stage 2 deterministic. Implemented as a
  host gather → ancestral step → write-back so changing noise stays outside the address-baked
  `inner_step` trace (`use_ancestral_sampler=True` on `LTX25DistilledPipeline`).
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

- The device broker reaps a job after 300 s of silence, and piping pytest through `tail`
  buffers everything — including the framework's own 5 s progress heartbeat — so the job
  looks dead and gets killed. Don't pipe.
- On a 32-chip box `require_exact_physical_num_devices` forces the full 4×8; a 2×4 submesh
  fails fabric router sync at mesh open, before any model code runs.
- Full 12B encode is 136 s on 4×8, 45 s of which is loading Gemma (~50 s of a cold run is
  converting 530 device tensors).
