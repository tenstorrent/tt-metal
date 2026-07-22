# FIBO SmolLM3 encoder simplification — design

Date: 2026-07-22
Status: approved (pending spec review)
Branch: `fibo-pipeline`

## Summary

Simplify the FIBO SmolLM3 text encoder (`models/tt_dit/encoders/smollm3/`) and its
pipeline wrapper (`models/tt_dit/pipelines/bria_fibo/text_encoder.py`) by adopting the
clean structural patterns from the `origin/friedrich/fibo` branch, and by pruning the
encoder down to exactly the paths the full FIBO pipeline exercises.

The encoder stays **standalone** — we do NOT migrate onto friedrich's shared
`encoders/transformer.py`. We keep our own `model_smollm3.py` and mirror the good
patterns locally.

## Goals

- Adopt friedrich's lazy, cache-backed checkpoint loading.
- Adopt a declarative state-dict key conversion for the top-level renames.
- Slim the encoder config down to a lean dataclass.
- Prune every code path the full pipeline never runs, keeping only the
  sequence-parallel (SP) path.

## Non-goals / preserved behavior (non-negotiable)

The following are **preserved exactly** — same math, same weights, same PCC:

- SP × TP parallelism: SP=8 × TP=4 on the 4×8 Galaxy mesh, SP=2 on the 2×2 Blackhole
  mesh. `sp_factor` and `tp_factor` remain parameterized.
- The per-shard SP causal-bias construction and its cache (`build_sp_causal_bias`,
  `_sp_bias_cache`).
- Fused qkv projection (`qkv_proj`) with the `optimal_groups` head-padding / split-factor
  surgery.
- The fast shard-selective readback (`_read_seq_sharded` via `ttnn.get_device_tensors`).

We also do **not** migrate to the shared `TransformerEncoder` abstraction; the encoder
remains a self-contained SmolLM3 implementation.

## Target meshes

Both 2×2 Blackhole (dev) and 4×8 Galaxy. `sp_factor` stays parameterized to cover
sp=2 and sp=8, but is no longer permitted to be 1.

## Changes

### 1. Declarative top-level state conversion

Replace the imperative `SmolLM3TextEncoder._prepare_torch_state`
(`model_smollm3.py:490-498`, which strips the `model.` prefix and pops
`lm_head`/`rotary_emb`) with a small declarative `StateConversion` helper using regex
`rename`/`remove` rules:

- rename `^model\.` → `` (strip prefix)
- remove `lm_head`
- remove `rotary_emb`

Add a minimal local `StateConversion` (regex `rename` + `remove`, mirroring friedrich's
in `encoders/transformer.py`) in the smollm3 module rather than depending on the shared
encoder file we do not otherwise use.

**Known limitation (accepted):** the qkv **fusion** + group-pad/split surgery in
`SmolLM3Attention._prepare_torch_state` (`model_smollm3.py:273-319`) **cannot** be
expressed as a rename table (a rename cannot fuse three tensors into one). It is exactly
why friedrich keeps q/k/v separate. Because keeping fused qkv is part of the preserved
perf path, the attention conversion **stays imperative**. This change is therefore
largely cosmetic — it tidies the ~6-line top-level conversion and removes the
`pop_substate` usage — but it aligns the loading style with friedrich and is kept for
consistency.

### 2. Lazy `SmolLm3Checkpoint` + `cache.load_model` (primary win)

Introduce a `SmolLm3Checkpoint` class:

- `__init__(name)`: read only `config.json` (via `AutoConfig.from_pretrained(name,
  subfolder="text_encoder")`) and build the `SmolLM3Config`. No torch weights loaded.
- `build(*, device, parallel_config, ccl_manager)`: construct the `SmolLM3TextEncoder`
  and call `cache.load_model(model, get_torch_state_dict=<lazy loader>,
  model_name=name, subfolder="text_encoder", parallel_config=parallel_config,
  mesh_shape=tuple(device.shape))`.
- The lazy `get_torch_state_dict` callback loads the HF weights
  (`SmolLM3ForCausalLM.from_pretrained(...).model.state_dict()`) and applies the
  top-level `StateConversion`. It runs **only on cache miss**.

This replaces the wrapper's eager load (`text_encoder.py:99-106`). Warm runs load from
the per-parallel-config on-device cache and skip the HF load entirely.

Feasibility confirmed: our `models/tt_dit/utils/cache.py::load_model` already exposes the
exact lazy signature (`get_torch_state_dict`, `parallel_config`, `mesh_shape`,
`subfolder`). The cache is keyed per parallel-config/mesh, which is correct because the
fused-qkv weight layout depends on `tp_factor`.

The wrapper (`SmolLM3TextEncoderWrapper.__init__`) is updated to construct a
`SmolLm3Checkpoint` and call `build(...)` instead of instantiating the full torch model
and calling `load_torch_state_dict`.

### 3. Slim `config.py`

Convert `SmolLM3Config` to a lean `@dataclass` exposing a single `from_hf` classmethod.

- Keep the robust `_read_rope_theta` fallback (top-level vs
  `rope_parameters`/`rope_scaling`).
- Keep the `no_rope_layers` derivation.
- Drop the long duplicated positional `__init__` defaults — only `from_hf` is used by the
  pipeline.
- Keep `config.py` as its own file.

### 4. Prune to SP-only

In `model_smollm3.py`, assume `sp_factor >= 2`. Remove:

- **FSDP**: the `is_fsdp` parameter, `fsdp_mesh_axis`, the `SmolLM3Context.fsdp_mesh_axis`
  field, and its threading through every `ColParallelLinear` / `RowParallelLinear`
  constructor. (Unused: no test sets it, the pipeline never sets it.)
- **Non-SP path**: the `sp_factor == 1` branches in `SmolLM3TextEncoder.forward` and in
  the wrapper's `_prep_inputs` / `_read_seq_sharded`.
- **Mask path**: `prepare_attention_bias()`, the `elif attention_mask is not None` branch,
  and the `MAX_CHUNK_SIZE`-based input-padding block (`model_smollm3.py:542-551`). The
  `attention_mask` parameter is removed from `SmolLM3TextEncoder.forward` / `.encode`
  (always `None` from the pipeline).

The SP causal-bias path becomes the only path in `forward`. The
`SmolLM3Attention.forward` SP K/V all-gather and the `is_causal=attention_bias is None`
handling are simplified accordingly (attention bias is always the SP causal bias).

In the wrapper (`text_encoder.py`):

- Collapse `_prep_inputs` and `_read_seq_sharded` to the SP>1 branch only.
- **Drop the `use_torch` reference path** (`__init__` `use_torch` arg, `_torch_encoder`,
  and the `use_torch` branch in `encode_prompt`). The full pipeline never sets it. This is
  the one item outside strict "adopt from friedrich" scope; it is included because it is
  dead in the pipeline-only configuration. Easy to reverse if host-reference debugging is
  wanted later.

## Testing strategy

- **Remove** the `(1,1)`-mesh unit tests (mlp / attention / layer / encoder / encode at a
  single device) and the mask-based HF-anchor test in
  `models/tt_dit/tests/encoders/smollm3/test_smollm3.py` — they exercise deleted paths.
- **Keep** the `(2,2)` and `(4,8)` mesh `enc.encode` PCC tests (vs HF) as the correctness
  anchor.
- **Keep** the pipeline latent-PCC test as the end-to-end anchor.
- **Accepted consequence:** the encoder can no longer be unit-tested on a single device;
  correctness is anchored only on real multi-device meshes.

## Risks & migration

- The SP>1 path is behavior-preserving (same weights, same math, same readback), so PCC
  should be identical to today. This is the primary verification gate.
- First run after the change **repopulates the weight cache** because the conversion path
  changes — expected one-time cost.
- The declarative-conversion change is cosmetic (fusion stays imperative); no behavioral
  effect.

## File-by-file impact

- `models/tt_dit/encoders/smollm3/config.py` — slim to a dataclass + `from_hf`.
- `models/tt_dit/encoders/smollm3/model_smollm3.py` — add local `StateConversion`, replace
  top-level `_prepare_torch_state`, remove FSDP / non-SP / mask paths, drop
  `attention_mask` params. Net shrink.
- `models/tt_dit/encoders/smollm3/__init__.py` — export `SmolLm3Checkpoint` (and
  `StateConversion` if module-public).
- `models/tt_dit/pipelines/bria_fibo/text_encoder.py` — use `SmolLm3Checkpoint.build`,
  drop eager load, drop `use_torch`, collapse SP>1-only prep/readback. Net shrink.
- `models/tt_dit/tests/encoders/smollm3/test_smollm3.py` — remove `(1,1)` and mask tests;
  keep `(2,2)`/`(4,8)` PCC tests.

## Verification (definition of done)

1. `(2,2)` and `(4,8)` `enc.encode` PCC tests pass at the same PCC threshold as today.
2. Pipeline latent-PCC test passes unchanged.
3. Encode perf (SP=8×TP=4) unchanged within noise (~12.5s ballpark, readback still fast).
4. No references remain to removed symbols (`is_fsdp`, `fsdp_mesh_axis`,
   `prepare_attention_bias`, `use_torch`, non-SP branches).
