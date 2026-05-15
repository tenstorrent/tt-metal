# LTX Pipeline Weight Cache — Design

**Date:** 2026-05-15
**Branch:** `kevinmi/ltx2-dit-port`
**Author:** vsuresh (with Claude)

## Background

The LTX-2.3 Pro pipeline (`run_ltx_pro.py`) re-loads the 22B safetensors checkpoint from disk and re-pushes weights to all 8 BH chips on every run. Measured overhead is **~85-90 seconds per run** outside the timed denoise/encoding/VAE stages, consisting of:

- safetensors deserialization (~25-30s)
- per-tensor host-side reshape (transpose, bias unsqueeze, AdaLN unsqueeze, …)
- `ttnn.from_torch` dtype conversion + tile padding + mesh sharding
- DMA push to 8 devices

This overhead is paid every run regardless of denoise config, so on a 21-minute Pro run it's ~7% of wall time; on iterative smoke runs it dominates.

The Wan pipeline (`models/tt_dit/pipelines/wan/pipeline_wan.py`) already solves this via the shared `models/tt_dit/utils/cache.py:load_model()` infrastructure, which serializes per-parameter `.tensorbin` files keyed by model + parallel config + mesh shape + dtype. The LTX pipeline simply bypasses it — calling `tt_model.load_torch_state_dict(state_dict)` directly.

## Goal

Wire the LTX pipeline into the existing `cache.load_model()` infrastructure so warm runs skip the disk read, tensor conversion, and device DMA. **Scope is intentionally narrow: no new cache infrastructure, no framework-wide refactor.**

Non-goals:

- Caching text-encoder weights (Gemma) — separate concern, embedding cache already exists.
- Caching across model architecture changes — manual `rm -rf` is the invalidation path.
- A new tt_dit-wide cache abstraction — `cache.load_model` is already framework-level.

## Architecture

Three call sites in `models/tt_dit/pipelines/ltx/pipeline_ltx.py` switch from direct state-dict loading to `cache.load_model(...)`:

1. **Transformer load** — `LTXPipeline.load_transformer(state_dict)` (`pipeline_ltx.py:274`). Variant detection (22B 9-output AdaLN vs 19B-distilled 6-output) stays — it inspects the state dict's `adaln_single.linear.weight` shape before model construction. After construction, replace `self.transformer.load_torch_state_dict(state_dict)` with a `cache.load_model(...)` call passing `get_torch_state_dict=lambda: state_dict`.

2. **VAE load** — `pipeline.load_vae_from_checkpoint()`. Same wrapping pattern, `subfolder="vae"`.

3. **Stage 2 reload in `run_ltx_fast.py`** — the script reloads the transformer between Stage 1 and Stage 2 (`run_ltx_fast.py:382-388`). No code change there: with the cache wired into `load_transformer`, the second call becomes a cache hit automatically.

### Variant detection on warm cache

`load_transformer` today peeks at the state dict to pick AdaLN width and config (`pipeline_ltx.py:276-279`). On a warm cache hit we still need to do this peek — `cache.load_model` operates on an already-constructed `tt_model`, so the model must be built with the right width.

We keep the eager `safetensors.torch.load_file(checkpoint)` + shape probe. `load_file` is mmap-friendly and on a warm OS page cache the probe is sub-second; on a cold cache we'd be reading the safetensors anyway, so no regression. We then construct the model and call `cache.load_model`, which short-circuits before re-reading or re-converting if the cache exists. **The eager load is only "wasted" on warm cache hits and only for the duration of the metadata probe, not the full tensor read.**

(An alternative — caching the variant config separately so we can skip safetensors entirely on warm hits — adds state and complexity for a sub-second saving. Rejected as YAGNI.)

## Cache key

Reuses the existing scheme (`cache.py:124-148`). Composed as:

| Component | Source | Example |
|---|---|---|
| `model_name` | `Path(checkpoint).stem` | `ltx-2.3-22b-dev` |
| `subfolder` | hardcoded per call site | `transformer`, `vae` |
| `parallel_config` | `self.parallel_config` (existing) | encoded via `config_id(...)` |
| `mesh_shape` | `tuple(self.mesh_device.shape)` | `2x4` |
| `dtype` | `"bf16"` | `bf16` |
| `is_fsdp` | from pipeline / `self.is_fsdp` | usually `False` |

The 22B-dev vs 19B-distilled split is handled by `model_name` since the safetensors filename differs (`ltx-2.3-22b-dev.safetensors` vs `ltx-2.3-22b-distilled.safetensors`). No extra key bits needed.

### Storage layout

```
$TT_DIT_CACHE_DIR/
└── ltx-2.3-22b-dev/
    ├── transformer/
    │   └── SP4_0_TP2_1_mesh2x4_bf16/
    │       ├── cache_dict.json
    │       └── <param>.tensorbin × N
    └── vae/
        └── SP4_0_TP2_1_mesh2x4_bf16/
            ├── cache_dict.json
            └── <param>.tensorbin × M
```

This matches Wan's layout. No new directories or conventions.

## Invalidation

- **Cache is opt-in via `TT_DIT_CACHE_DIR`.** If unset, `cache.load_model` falls back to today's eager `load_torch_state_dict` path (`cache.py:92-101`) — no behavior change for users who haven't set the env var.
- **Config changes don't collide** because parallel_config, mesh, dtype, and FSDP are all in the cache key.
- **Model architecture changes are NOT versioned in the key.** If the user pulls new code that adds a layer or changes a parameter shape, the cache becomes stale and load will fail at the per-parameter shape check in `Parameter.load()`. Mitigation: user manually `rm -rf $TT_DIT_CACHE_DIR/ltx-2.3-22b-dev`. Same model as Wan today; not worth adding a code hash for an internal tool.

## Env vars / opt-in

- **`TT_DIT_CACHE_DIR`** — already established. Setting it enables caching; leaving it unset preserves current behavior. No new env vars.
- **No runner CLI flags.** The runners (`run_ltx_pro.py`, `run_ltx_fast.py`) need no changes; users opt in via the env var.

## Testing

1. **Cold/warm parity:** Cold run with `TT_DIT_CACHE_DIR=/tmp/ttdit-cache` populates the cache. Warm run (same env var) hits cache. Both produce **bit-identical output MP4** (deterministic seed + identical weights).
2. **Speedup measurement:** Time `load_transformer` + `load_vae_from_checkpoint` on cold vs. warm runs. Target: warm load ≤ 5s for transformer + VAE combined, vs. ~85s today.
3. **Mesh-shape isolation:** Run on 2x4, then run on 1x8 with the same `TT_DIT_CACHE_DIR`. Verify the 1x8 run misses cache and builds at a different path (`mesh1x8` not `mesh2x4`).
4. **Variant isolation:** Run with `ltx-2.3-22b-dev.safetensors` then with `ltx-2.3-22b-distilled.safetensors` using the same `TT_DIT_CACHE_DIR`. Verify they don't collide and each gets its own cache dir.
5. **Stage 2 reload in Fast pipeline:** `run_ltx_fast.py` reloads the transformer between stages. Warm run should hit cache for both stages.

## Risks / open questions

- **`Module.save()` correctness on LTX-specific params.** Wan exercised this code path; LTX has slightly different parameter naming (per-head gate, cross-attn AdaLN). If any parameter isn't a standard `Parameter` instance or has custom `_prepare_torch_state` that doesn't round-trip, save/load may break. Mitigation: cold/warm parity test (item 1 above) catches this immediately.
- **Cache disk size.** 22B × 2 bytes = ~44 GB per cached transformer config. Multiple configs (2x4 vs 1x8, etc.) multiply this. User should set `TT_DIT_CACHE_DIR` somewhere with enough space (e.g., `/localdev/vsuresh/.cache/ttdit/`, not `~/.cache`).
- **Filename collisions across LTX-2 versions.** `ltx-2.3-22b-dev.safetensors` vs hypothetical `ltx-2-22b-dev.safetensors` would both produce `model_name="ltx-...-22b-dev"`. Acceptable for now; if collisions emerge, add a content hash prefix.

## Files touched

| File | Change |
|---|---|
| `models/tt_dit/pipelines/ltx/pipeline_ltx.py` | Replace direct `load_torch_state_dict` with `cache.load_model` in `load_transformer` and `load_vae_from_checkpoint`. Compute `model_name` from checkpoint path. |
| `models/tt_dit/utils/cache.py` | Unchanged (just reused). |
| Runners (`run_ltx_pro.py`, `run_ltx_fast.py`) | Unchanged. Users opt in via `TT_DIT_CACHE_DIR`. |
