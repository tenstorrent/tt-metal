# FIBO — Separate FIBO changes from shared library into FIBO-owned modules

**Date:** 2026-07-23
**Branch:** `fibo-pipeline`
**Status:** Design — awaiting review

## Goal

Keep the FIBO/Bria pipeline's changes out of shared tt_dit library files. Every FIBO-specific
component should live in a FIBO-owned file; the shared library files FIBO currently modifies should
be restored to their pristine `main` state (or, where a change is a genuine general improvement,
reframed as such rather than living as an unowned FIBO edit).

## Hard constraints

- **Behavior identical.** The FIBO pipeline must produce the same image output (same PCC vs. HF
  reference) before and after. This is a pure move/re-org, not a logic change.
- **Performance identical.** All perf-tuning (conv blockings, matmul configs, 12×10 grid) must remain
  in effect for FIBO. No FIBO op may regress.
- **Copying whole files is acceptable.** Where clean separation would otherwise require surgical
  edits to shared internals, duplicating a whole file into a FIBO-owned copy is explicitly allowed
  (user-approved).
- **Other models unaffected.** Reverting shared files must not break QwenImage, Wan, SD3.5, Flux,
  Motif, LTX, etc.

## Current FIBO footprint in shared files

Measured as the diff of branch `fibo-pipeline` vs. its merge-base with `main`
(`2eb991a4100117464d67d5331bd6968470bf9daa`):

| # | Shared file | FIBO delta | Nature |
|---|---|---|---|
| 1 | `utils/conv3d.py` | +62 | FIBO conv-blocking lookup-table entries (`_BLOCKINGS`, `_DEFAULT_BLOCKINGS`) |
| 2 | `utils/matmul.py` (configs) | — | Already FIBO-owned: registered at runtime via `register_matmul_configs()` from `transformer_bria_fibo.py`. No shared-file edit. |
| 3 | `utils/matmul.py` (grid) | +6 | Global `_BH_GALAXY_MAX_CORE_GRID` flipped `(11,10)` → `(12,10)` |
| 4 | `models/vae/vae_wan2_1.py` | +252 | Wan 2.2 residual-decoder support (`WanDupUp3D`, `WanResidualUpBlock`, `is_residual` branch, `first_chunk`, `decoder_base_dim`, SDPA `k_chunk_size` autoscale) |
| 5 | `parallel/config.py` | +15 | `EncoderParallelConfig.cfg_parallel` optional field + `from_tuples()` classmethod |
| 6 | `utils/sweep_mm_block_sizes.py` | +142 | FIBO sweep topology configs (dev tool, not runtime) |

FIBO components that are **already** cleanly separated (no change needed): the pipeline
(`pipelines/bria_fibo/`), the DiT (`models/transformers/transformer_bria_fibo.py`), the SmolLM3
encoder (`encoders/smollm3/`), and all FIBO tests.

## Key facts discovered

- **`conv3d.py` already exposes `register_conv3d_configs(configs: dict)`** (line 586) — the intended
  mechanism for adding blocking entries without editing the shared module-level dicts.
- **`matmul.py` already exposes `register_matmul_configs()`** — FIBO's DiT already uses it, so the
  matmul *config* data is already FIBO-owned. Only the grid *constant* is a raw shared edit.
- The **12×10 grid** is read through shared `layers/linear.py::get_matmul_core_grid()` by **every**
  model. It cannot be FIBO-scoped without copying `linear.py` (used everywhere) — too invasive.
- `vae_wan2_1.py` base classes (`WanDecoder`, `WanDecoder3d`, `WanResidualBlock`, `WanResample`,
  `WanUpBlock`, `WanAttentionBlock`) are **shared** — imported by `vae_qwenimage.py` and the Wan
  pipelines. On `main` the decoder asserts `not is_residual`, so **no non-FIBO consumer uses the
  residual decode path today** → reverting the residual additions is safe for other models.
- `EncoderParallelConfig` is a **library-wide type** (gemma, t5, clip, qwen25vl, smollm3 encoders +
  all pipelines). The FIBO additions are **additive and backward-compatible**: the new field is
  optional (`default None`) and `from_tuples()` is a new classmethod; no existing caller is affected.

## Design decisions (approved)

- **VAE → full fork.** Copy the entire current `vae_wan2_1.py` (with residual additions) to a new
  FIBO-owned `models/vae/vae_bria_fibo.py`; revert shared `vae_wan2_1.py` to `main`. Zero shared
  coupling, fully self-contained. Accepted tradeoff: the ~2260-line file is duplicated and future
  shared-VAE fixes won't auto-propagate to FIBO.
- **12×10 grid → keep in shared, reframe.** Leave `_BH_GALAXY_MAX_CORE_GRID = (12,10)` in
  `matmul.py`, but rewrite the comment to describe it as a verified general Blackhole-Galaxy
  capability (with 11×10 retained as fallback), not an unowned FIBO edit. Preserves perf for FIBO
  **and** every other Galaxy model. This is the one touchpoint that intentionally stays shared.

## Per-touchpoint plan

### 1. conv3d blocking tables → FIBO-owned + registration
- Create `models/tt_dit/pipelines/bria_fibo/fibo_conv3d_configs.py` holding the FIBO `_BLOCKINGS`
  entries (2×2 and 4×8 image-decode) and the Wan-2.2 `_DEFAULT_BLOCKINGS` entries — exactly the
  lines added in the +62 diff, verbatim.
- Register them via `register_conv3d_configs(...)` at FIBO VAE import time (module-level call in
  `vae_bria_fibo.py`, or in the pipeline's `__init__` before VAE construction).
- **Verify during implementation:** that `register_conv3d_configs` merges into *both* the exact-key
  table (`_BLOCKINGS`) and the default-key table (`_DEFAULT_BLOCKINGS`); if it only handles one,
  extend the FIBO registration accordingly (or add a second registration entry point).
- Revert `utils/conv3d.py` to `main`.

### 2. matmul configs → no change
Already FIBO-owned via `register_matmul_configs()`. Confirm the registration data lives entirely in
`transformer_bria_fibo.py` and nothing FIBO-specific remains inline in `matmul.py`'s config dicts.

### 3. matmul 12×10 grid → keep, reframe comment
Edit only the comment block above `_BH_GALAXY_MAX_CORE_GRID` in `utils/matmul.py` to present 12×10 as
a general verified capability. No functional change. Remove the FIBO-specific framing/cross-reference
so it reads as a library decision.

### 4. VAE → `vae_bria_fibo.py` (full fork)
- Create `models/vae/vae_bria_fibo.py` as an exact copy of the **current** (HEAD) `vae_wan2_1.py`
  — i.e. including all residual additions. Mechanically: `git show HEAD:.../vae_wan2_1.py` → new
  file, guaranteeing byte-identical logic to today's working FIBO VAE.
- Revert shared `models/vae/vae_wan2_1.py` to the merge-base version
  (`git checkout <merge-base> -- models/vae/vae_wan2_1.py`), removing residual support, `first_chunk`
  threading, `decoder_base_dim`, and the SDPA `k_chunk_size` autoscale. (All byte-for-byte safe for
  Wan 2.1 / QwenImage: `dim<=384` keeps original SDPA chunks; `decoder_base_dim` defaulted to
  `base_dim`; the `is_residual` asserts return.)
- Update imports of `WanVAEDecoderAdapter` / `WanDecoder` to point at `vae_bria_fibo`:
  - `pipelines/bria_fibo/pipeline_bria_fibo.py:44`
  - `tests/models/bria_fibo/test_performance_bria_fibo.py:713`
  - `tests/models/bria_fibo/test_vae.py:53,147`
- Leave `pipelines/wan/*` and `models/vae/vae_qwenimage.py` importing the (now pristine) shared file.

### 5. EncoderParallelConfig → FIBO-owned config
- Define a FIBO-owned parallel config (e.g. `FiboEncoderParallelConfig` in
  `pipelines/bria_fibo/`) as a NamedTuple with `tensor_parallel`, `sequence_parallel`, `cfg_parallel`
  and a `from_tuples()` classmethod — the fields the FIBO pipeline and SmolLM3 encoder need.
- Point FIBO's pipeline (`pipeline_bria_fibo.py:45,65,134`) and SmolLM3 encoder type hints at the
  FIBO config. Duck-typing keeps the SmolLM3 encoder (FIBO-owned) working since it only reads
  `.tensor_parallel` / `.sequence_parallel`.
- Revert `parallel/config.py` to `main`.
- **Flagged tradeoff (low separation value):** this change is additive and backward-compatible;
  reverting it duplicates a shared NamedTuple purely for ownership hygiene. Confirm at review whether
  it's worth doing or whether this one touchpoint should stay in shared `config.py`.

### 6. sweep_mm_block_sizes → FIBO-owned dev-tool file
- Move FIBO sweep configs into `utils/sweep_mm_block_sizes_fibo.py` (or a `bria_fibo`-local module),
  wired so the sweep tool can still discover them. Revert `utils/sweep_mm_block_sizes.py` to `main`.
- Lowest stakes (dev tooling, not runtime); no perf/output impact.

## Resulting FIBO-owned file layout

```
models/tt_dit/
  models/vae/vae_bria_fibo.py                     # NEW — full VAE fork (was inline in vae_wan2_1.py)
  models/transformers/transformer_bria_fibo.py    # unchanged (already owns its matmul configs)
  encoders/smollm3/                               # unchanged
  pipelines/bria_fibo/
    pipeline_bria_fibo.py                         # imports updated
    text_encoder.py
    fibo_conv3d_configs.py                        # NEW — FIBO conv blockings + registration
    fibo_parallel_config.py                       # NEW — FiboEncoderParallelConfig (if pursued)
  utils/sweep_mm_block_sizes_fibo.py              # NEW — FIBO sweep configs (dev tool)
```

Shared files after this work: `conv3d.py`, `parallel/config.py`, `sweep_mm_block_sizes.py`,
`models/vae/vae_wan2_1.py` all == `main`. `matmul.py` keeps the 12×10 grid (reframed comment only).

## Verification strategy

1. **Baseline capture (before changes):** run the FIBO VAE + pipeline tests and record image PCC and
   the perf numbers:
   - `tests/models/bria_fibo/test_vae.py` (decode PCC vs. HF reference)
   - `tests/models/bria_fibo/test_pipeline.py`
   - `tests/models/bria_fibo/test_performance_bria_fibo.py`
2. **After changes:** re-run the same tests; PCC and per-op / end-to-end perf must match baseline.
3. **Other-model smoke:** import/build QwenImage and Wan VAE paths to confirm the reverted shared
   `vae_wan2_1.py` and `parallel/config.py` still work (they equal `main`, which is known-good).
4. **Diff check:** confirm `git diff <merge-base> -- <shared files>` is empty for conv3d,
   parallel/config, sweep, vae_wan2_1; and only the comment differs for matmul.py.

## Risks & mitigations

- **Fork drift (VAE):** the duplicated VAE won't receive shared fixes. Mitigation: a header comment
  in `vae_bria_fibo.py` noting it was forked from `vae_wan2_1.py` at this commit, and why.
- **conv3d registration coverage:** if `register_conv3d_configs` doesn't cover the default-key table,
  FIBO convs could silently fall back to the generic default and regress perf. Mitigation: assert the
  registered keys are present after registration; verify perf test in step 2.
- **Registration ordering:** FIBO conv/matmul configs must be registered before the VAE/DiT build.
  Mitigation: register at module import of the FIBO VAE / in pipeline `__init__` prologue.

## Out of scope

- Any logic, perf, or numerical change to FIBO.
- Refactoring shared library code beyond restoring it to `main`.
- Deleting the stale empty `pipelines/fibo/` and `tests/models/fibo/` dirs (only `.pyc` remain) —
  optional cleanup, can be a trivial follow-up.
