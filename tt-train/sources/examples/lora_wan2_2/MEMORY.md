# Working state — Wan2.2 T2V-A14B LoRA on tt-train

Last updated: 2026-08-06. Everything here was verified on the machine, not assumed.

## Goal

Port a working CUDA/diffusers LoRA pipeline (4 stages: preprocess → precompute → train → infer)
to tt-train, keeping stage boundaries, cache format and training semantics identical so the two
runs stay comparable. T2V only, image-only training.

## Hardware

- **Blackhole Galaxy, 32 chips** (`tt-smi` reports Board Type: Blackhole, tt-galaxy-*).
  Also visible in kernel logs as `compute_with_storage_grid_size=12-10` and `-DARCH_BLACKHOLE`.
- `device.mesh_shape: [4, 8]` uses all 32. For precompute the axes are the VAE height/width
  parallel factors; for train they are `[DP, TP]`.
- `num_links=2` — the Blackhole value. 4 is Wormhole 4x8, 1 is WH 2x4.
- `train` needs `TT_MESH_GRAPH_DESC_PATH` set explicitly; auto-selection may pick a descriptor
  that does not match a 4x8 Blackhole Galaxy.

## Where things stand

| Stage | State |
|---|---|
| `preprocess` | **done** — `data/lego/`: 106 images + 106 rows in `metadata.jsonl` |
| `precompute` | **done** — `cache/wan22_14b_lego/`: 106 latents, `embeds.pt`, `metadata.json` |
| `train` | **blocked** — see the model-port section below |
| `infer` | untested (needs a trained adapter) |

Verified cache contents: latent `(16, 1, 64, 64)` fp32, finite, mean −0.09 std 0.58 (the std
matters — it confirms the `latents_mean`/`latents_std` normalization worked). Embedding
`(512, 4096)` bf16 with 81 non-zero rows for a typical caption; the empty-caption entry for
classifier-free guidance dropout is present.

The cache was **rebuilt at 14:34–14:35** after turning on `strip_style_words`, and 0 of 106
captions still contain lego/blocky/minifigure. Captions now read `lg, A young girl, dressed as
a fairy…`. Note the rebuild took ~3 minutes versus ~25 for the first run, because the JIT
kernels are cached under `~/.cache/tt-metal-cache`.

Not re-verified after the rebuild: the number of unique captions. Stripping style words may
have collapsed some captions into duplicates, so `embeds.pt` could hold fewer than the 107
keys counted before the rebuild.

## Config

Single YAML: `tt-train/configs/training_configs/wan2_2_t2v_a14b_lora.yaml`.
All 54 `Config` fields are covered by it. The CLI is only `stage`, `-c/--config`, `--set KEY=VALUE`.

Non-default choices made so far:
- `device.mesh_shape: [4, 8]`
- `data.strip_style_words: true` — so the rare trigger `lg, ` carries the style rather than the
  words "LEGO style" that appear in every raw caption
- `lora.lora_target_set: attn` — reproduces what the CUDA run *actually* trains (see below)

## The model port — `ttml/models/wan2_2/` (written 2026-08-06)

All six files exist and compile. Design page:
https://claude.ai/code/artifact/77deb496-373b-4264-b65b-0717aafabb13

**All files written; the full test suite passes on device (2026-08-06).**

Tests: `tt-train/tests/python/test_wan2_2.py` (15 tests). Run with:

```bash
cd ~/tt-metal
TT_LOGGER_LEVEL=FATAL python_env/bin/python -m pytest tt-train/tests/python/test_wan2_2.py \
    --confcutdir=tt-train/tests/python --tb=native -s
```

`--confcutdir` is required (the repo-root conftest imports `ttnn.device` and fails);
`--tb=native` is required (a long traceback reprs a ttml tensor and **segfaults**);
`python_env/bin/python` is the only interpreter with both ttml and diffusers.

| File | Verified by |
|---|---|
| `rope.py` | 3D tables bit-identical to diffusers (0.00e+00), 3 latent shapes; kernel confirmed to use adjacent-pair rotation (0.999996 vs 0.44 for split-halves) |
| `conditioning.py` | sinusoid matches diffusers; 5.8e-05 residual at t=999 is fp32 large-angle sensitivity, ~70x below bf16 eps |
| `patch_embed.py` | patchify+Linear == real Conv3d (8.9e-16); unpatchify == diffusers exactly |
| `attention.py` | `SplitHeads` fwd+bwd vs torch; block forward PCC vs diffusers; gradients reach all 8 LoRA-wrapped projections |
| `transformer.py` | `GeluTanh` fwd+bwd vs torch tanh gelu; staged block PCC after every sub-step |
| `weights.py` | name mapping on 12 keys; shape mismatch raises |

### The bug that cost the most: ttml's fused SDPA is causal by default

`ops/scaled_dot_product_attention.cpp` sets `mask_type = AttentionMaskType::Causal` and only
switches to `Arbitrary` when a mask is passed. So `sdpa(q, k, v, None)` is **causal**, but Wan
attends bidirectionally over patches. Symptom: block PCC 0.989, with error decaying monotonically
with sequence position (token 0 worst — it could only see itself; token 63 fine). Lowering the
modulation scale made it *worse*, not better, because more of the output then came from the
attention branch.

Fix: `attention.py` uses `scaled_dot_product_attention_composite`, which applies no mask unless
given one, and which also handles cross-attention's unequal q/k lengths.

**Perf debt from this:** the composite path is matmul + softmax + matmul and materialises the
`(B,H,S,S)` attention matrix — 1024x1024 per head at production size. Recovering the fused kernel
means passing an explicit all-permitted `Arbitrary` mask of shape `(1,1,S,S)`. The staged test will
guard that change.

Only a stage-by-stage bisect found this: five of six device tests were green, RoPE was bit-exact,
`SplitHeads` was verified, and gradients reached every projection — the fault was still a default
argument.

### ttml/ttnn sharp edges (each cost a debug round)

- `backward()` takes a **required** `retain_graph` bool — `backward()` alone is a TypeError.
- `Tensor.from_numpy` creates tensors with **`requires_grad=False`** (the C++ ctor default), while
  the module-level `create_tensor` binding defaults it to *true*. `Function.apply` then silently
  skips gradient accumulation for such inputs — no error, just no gradient.
- `get_grad()` returns **nullptr** when uninitialised; `to_numpy()` on that segfaults. Guard with
  `is_grad_initialized()`.
- `get_grad()` returns a **raw ttnn tensor**, not a ttml one; its `to_numpy` takes only
  `mesh_composer`. Discriminate via `hasattr(t, "get_requires_grad")` as `autograd/function.py` does.
- `to_numpy()` only upcasts if the tensor has an fp32 master copy; anything wrapped straight from
  bf16 (gradients, raw ttnn results) needs `to_numpy(new_type=ttnn.float32)`.
- `repr()` on a `_ttml.autograd.Tensor` **segfaults** — any pytest long traceback involving one
  kills the process. This is a genuine ttml bug worth filing.
- `docs/CUSTOM_AUTOGRAD_FUNCTIONS.md`'s example calls `output.backward()` on a non-scalar and reads
  `x.get_grad()`; it cannot work as written.
- reshape+permute on a **tile-layout** tensor is not a supported path (it segfaults). Use
  `nlp_create_qkv_heads` / `nlp_concat_heads` — which is what ttml's own head ops use.
- `ttnn.experimental.gelu_bw`'s `approximate` argument is **keyword-only**.

### Design decisions that differ from the original plan

- **No custom RoPE Function.** `RotaryEmbeddingParams` *is* constructible from Python
  (`nb::init<>()` + `def_rw` on all five caches) — the earlier "no constructor" claim was wrong,
  from grepping the wrong type name. So we build 3D tables on host, drop them into the struct, and
  call `ttml.ops.rope.rope`, inheriting the C++ backward. `neg_cos = cos`, `neg_sin = -sin`.
- **No conv3d at all.** Wan's patch embed has `stride == kernel`, so it is a linear map over
  non-overlapping patches: host rearrange + `LinearLayer`. This also dodges the conv3d
  blocking-table problem seen in the VAE, and stays differentiable in case the patch embed is ever
  trained.
- **Head splitting is per-tensor via `SplitHeads`.** `grouped_heads_creation` asserts
  `input_shape_kv[2] == input_shape[2]` — q and kv must share a sequence length, which is false for
  cross-attention (1024 image tokens vs 512 caption tokens). `SplitHeads` wraps
  `nlp_create_qkv_heads(..., num_kv_heads=0)` forward and `nlp_concat_heads` backward, the same pair
  ttml's own head ops use. This also removed the earlier `ConcatLastDim` entirely, since K and V no
  longer need fusing. Verified against torch in both directions.
- **`GeluTanh` Function** in `transformer.py`: ttml's `gelu` is the exact erf form both ways
  (backward passes `approx_mode="none"`), but Wan trained with the tanh approximation. Routes to
  `ttnn.gelu(..., fast_and_approximate_mode=True)` and `ttnn.experimental.gelu_bw(..., "tanh")`, so
  no derivative is hand-written. Measured gap if left unfixed: 4.7e-4 max, 8.5e-5 mean — small but
  systematic across 40 blocks, and a train/serve mismatch since inference uses tanh.
- **Conditioning is entirely no-grad.** LoRA touches only attention and FFN, so timestep sinusoid,
  time/text embedders and the 6-way chunk run under `GradMode.DISABLED` (restored in a `finally`).
  This is why no differentiable `chunk` is needed.
- **The model returns patch tokens, not a latent.** Unpatchifying on device would need a
  differentiable permute. Instead compare against `patchify_output_order(target)`: MSE is unchanged
  by a consistent permutation. Note the two orders genuinely differ — the conv weight contracts
  `(C, p_t, p_h, p_w)` (channel-major) while `proj_out` emits `(p_t, p_h, p_w, C)` (channel-minor).
- **`mask` is the second forward argument** on the block, because `memory_efficient_runner` calls
  its callable as `(input, mask, *extras)`.
- **The top-level `scale_shift_table` has 2 chunks**, not 6 like the per-block ones — the final norm
  needs shift and scale but no gate.
- **Not registered in `ttml/models/__init__.py`** on purpose: that file is imported by every ttml
  user, so registration waits until the package has run once on device. `train.py` imports from
  `ttml.models.wan2_2` directly, which works regardless.

### Remaining, in order

1. ~~Update `train.py`~~ **done** — takes patchified tokens, `patchify_output_order` target, and
   `rope_params` built once from `latent_shape(cfg)`. It refuses TP>1 (see item 3). Never executed.
2. ~~Small-config PCC test~~ **done and passing** (see above).
3. **Tensor-parallel pass — required before any real expert fits.** Everything is
   `ttml.modules.LinearLayer` (non-parallel), so nothing shards, but one expert is ~28 GB bf16 and
   the config asks for both on `[4, 8]`. Needs `ColumnParallelLinear` for `to_q/to_k/to_v/ff1` and
   `RowParallelLinear` for `to_out/ff2`, plus shard mappers in `weights.py`. The column/row split
   already assumed by `utils/lora_export.py`'s `_gather`/`_scatter` matches this exactly.
4. **Download the 14B experts** — still absent; only `vae`, `text_encoder`, `tokenizer` are cached.

## Open issues, in rough priority order

1. **`grad_clip` must be 0 at TP>1.** `ttml.core.clip_grad_norm` is per-device with no cross-mesh
   reduction, so at `[4, 8]` it would clip against shard-local norms. `train.py` refuses rather
   than clip wrongly. The exact fix is implementable — each LoRA pair has exactly one sharded half
   whose shards *partition* its elements, so summing per-device sums-of-squares and all-reducing
   over the TP axis is exact — but it needs a scalar all-reduce ttml does not expose.
2. **`ff` vs `ffn`.** The CUDA reference's `LORA_TARGETS` contains `"ff.net.0.proj"`, but Wan's
   feedforward module is named `ffn`, so under PEFT's suffix matching it never matched and the FFN
   was silently left un-adapted. Its `assert 0 < trainable < total // 20` still passed on the
   attention targets alone. Worth fixing on the CUDA side too. `ttml.modules.LoraModel` raises when
   a pattern matches nothing, so this class of typo cannot pass silently here.
3. **`dtype=torch.float32` is ignored on the text path.** Latents honour it, embeddings come back
   bf16. Hypothesis (untraced): the replicated `[None, None]` path in `fast_device_to_host`
   short-circuits and skips the fused dtype conversion. Harmless — training casts at the upload
   boundary — but the argument is misleading. Fix with an explicit `.float()`.
4. **`_ENCODER_SPATIAL_MULTIPLIER = 8` is hardcoded** in `utils/tt_encoders.py` while
   `_validate_res` derives the same 8 from `2 ** len(vae_config.temperal_downsample)`. A VAE variant
   with a different downsample count would silently pad too little. Derive it from config.
5. **`prompt_clean` divergence.** tt_dit normalizes caption text (ftfy, HTML unescape, whitespace)
   before tokenizing; the CUDA script does not. A no-op for clean ASCII, but different tokens for
   anything with mojibake or entities.
6. **LoRA A init differs** — PEFT `init_lora_weights="gaussian"` vs ttml's kaiming-uniform. Affects
   the early loss curve, not final quality. `LORA_A_INIT` documents the knob; not implemented.
7. **Mid-block attention is redundant at T=1.** After the all-gather the batch-time split is guarded
   on `T > 1`, so on stills all 32 chips compute the identical 4096-token attention. One block of
   ~30; first place to look if precompute feels slow.
8. **`ConcatLastDim.backward` is the only hand-written derivative in the model.** Silent failure
   mode: mis-sized or swapped gradient halves mean `to_k`/`to_v` adapters never train and nothing
   errors. Test it directly before trusting any loss curve.
9. **Cross-attention mask is plumbed but never built.** Irrelevant at batch 1 with everything padded
   to 512; matters once captions of differing length share a batch.
10. **Dropout after `to_out` is not implemented.** Wan configures it at 0.0, so a no-op today.

## Parity procedure

The strong gate is **step-level replay**: dump `(t, noise, latent, text_embed)` from one CUDA
`flow_matching_step` and feed those exact tensors here — losses should agree to bf16 tolerance,
because timesteps and noise are injected rather than sampled, and `x_t` is built on the host with
torch using the reference's own generators and seeds.

Curve-level comparison is only approximate: the device VAE lands ~0.995 PCC against the torch VAE,
so the two runs fit slightly different targets. Encode a cache with diffusers on a CUDA box if an
exact curve comparison is ever needed.

Adapter portability: files are written with PEFT/diffusers keys (`transformer.blocks.0.attn1.to_q.lora_A.weight`,
2-D, float32), verified to pass through tt_dit's lightx2v→diffusers rename unchanged, so one file
loads in diffusers, in tt_dit's `register_lora`, and here.

## Commands

```bash
export TT_METAL_HOME=/home/bmijanovic/tt-metal
export TT_MESH_GRAPH_DESC_PATH=...   # required for train on 32 devices
cd $TT_METAL_HOME/tt-train/sources/examples/lora_wan2_2

python pipeline.py preprocess
python pipeline.py precompute
python pipeline.py train --set GRAD_CLIP=0
python pipeline.py infer

# check stage 1 / stage 2 without a device
ls data/lego/images | wc -l
ls cache/wan22_14b_lego/samples | wc -l && ls -l cache/wan22_14b_lego/embeds.pt
```

To reset: `rm -rf cache/wan22_14b_lego` (redo precompute, ~3 min now that kernels are cached) or
also `data/lego` (redo the HF download). Do **not** clear `~/.cache/huggingface` — 12 GB, holds
the A14B weights.

## Reference pages

- DiT port design: https://claude.ai/code/artifact/77deb496-373b-4264-b65b-0717aafabb13
  (two claims on it are now stale: RoPE needs no custom Function, and patch embed needs no conv3d)
- Encoder implementation walkthrough: https://claude.ai/code/artifact/0cc2db85-1328-474e-b656-23b51be806cd
- Sharding animation: https://claude.ai/code/artifact/aea01add-0c91-492e-b489-62103216ee7c

## Untracked

Three new, untracked locations:

- `tt-train/sources/examples/lora_wan2_2/` — this directory
- `tt-train/configs/training_configs/wan2_2_t2v_a14b_lora.yaml`
- `tt-train/sources/ttml/ttml/models/wan2_2/` — the model port (6 files)

No existing file has been modified: `models/tt_dit` is untouched, and `ttml/models/__init__.py`
deliberately does not yet register the new model (see the port section).
