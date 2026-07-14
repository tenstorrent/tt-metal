# FIBO 4×8 Blackhole Galaxy bring-up (SP=4, TP=8) — design

**Date:** 2026-07-14
**Branch:** `fibo-pipeline`
**Status:** approved, ready for implementation plan

## Goal

Bring up the Bria FIBO text→image pipeline on a **4×8 Blackhole Galaxy** mesh with
**sequence-parallel SP=4 (mesh axis 0)** and **tensor-parallel TP=8 (mesh axis 1)**.

Success = a PCC-passing end-to-end 1024² image on the 4×8 mesh (transformer mesh unit
test + pipeline golden/PCC test), validated on the physical 4×8 Galaxy machine.

**Explicitly functional-first.** Performance tuning (matmul block configs, SDPA chunk
sizes, encoder FSDP/TP, VAE conv blocking) is a *follow-up*, not this spec. The bar here
is correctness at SP=4/TP=8, not competitive it/s.

## Background

FIBO currently runs only on a **2×2 Blackhole** mesh (SP=2/TP=2). The parallelism config
plumbing is already shape-driven — `BriaFiboPipelineConfig.default()`
(`models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py`) reads `sp_factor=mesh[0]`,
`tp_factor=mesh[1]`, with `sp_axis=0`, `tp_axis=1`, `cfg=(1,0)` — so a `(4,8)` mesh
*already* produces SP=4/TP=8 in `DiTParallelConfig` and a single 4×8 submesh from
`create_submeshes` (cfg factor 1 → cond and uncond run sequentially on the full mesh).

The repo-wide pattern for multi-mesh support (Flux1, SD3.5, Mochi, QwenImage, WAN, LTX,
Motif) is: a `_PRESETS` dict keyed by `tuple(mesh_shape)` → SP/TP/CFG factors, config
`NamedTuple`s of `ParallelFactor(factor, mesh_axis)`, a `CCLManager` whose collectives
**no-op when `mesh_device.shape[mesh_axis] == 1`**, and parallel-aware layers
(`ColParallelLinear`, `Attention` with ring SDPA). FIBO uses all of this infra except the
preset dict.

The one hard blocker is the per-block text injection, which is hard-coded to TP==2.

## Design decisions (approved)

1. **Stay shape-driven** — do *not* adopt a full `_PRESETS` dict. Only introduce a minimal
   preset/override for values that cannot be derived from the mesh shape, namely
   `num_links` and CCL `topology`, which are hardware-dependent. Pin these empirically for
   BH Galaxy during implementation (do not guess — a wrong `num_links` fails CCL loudly).
2. **Generalize `inject_text` via all-gather + concat + reshard**, TP-general. **Drop** the
   TP==2 local-mask path (`_InjectionMask`) entirely — no per-shape special case.
3. **Encoder (SmolLM3): keep replicated** (`encoder_tp` factor 1) for v1. Replication is
   per-device identical footprint to the working 2×2 case; only load time grows. FSDP/TP on
   a reshaped submesh (mirroring QwenImage's Qwen2.5 encoder) is a documented follow-up.
4. **VAE (Wan): keep the shape-driven hw-parallel mapping** — `height=(tp_factor, tp_axis=1)`,
   `width=(sp_factor, sp_axis=0)` → height split 8, width split 4 at the 64×64 latent. The
   Wan VAE has no divisibility asserts (dims are ceil-padded), so this is valid.

## Component changes

### 1. `inject_text` / `_InjectionMask` — the core model change
File: `models/tt_dit/models/transformers/transformer_bria_fibo.py`

Current state (lines 108–204, 345–351, 447, 463): at TP>1, `_InjectionMask` asserts
`inner_dim // tp_factor == inner_dim // 2` (i.e. TP==2 only). `inner_dim=3072`,
`half=1536`; at TP=8 the shard is 384 ≠ 1536 → assertion fails. The mask does a gather-free
local select that only works when the half-boundary lands on a shard boundary *and* each
upper-half device needs exactly the replicated `projected` (true only at TP==2).

New behavior — a single TP-general path, replacing both the tp==2 mask path and keeping the
tp==1 plain concat:

- **tp==1** (unchanged): plain feature-dim concat
  `concat([encoder_hidden_states[..., :half], projected], dim=-1)`.
- **tp>1** (new, general):
  1. `enc_full = ccl_manager.all_gather(encoder_hidden_states, dim=-1, mesh_axis=tp_axis, use_hyperparams=...)`
     → full `[batch, P, inner_dim]` replicated across the TP axis. (`all_gather` no-ops when
     the axis size is 1, so tp==1 could even share this path, but we keep the explicit concat
     fast path.)
  2. `injected_full = concat([enc_full[..., :half], projected], dim=-1)` in full inner-dim
     space; `projected` is already the replicated full `inner_dim//2 = 1536` on every device.
  3. `ttnn.mesh_partition(injected_full, dim=-1, cluster_axis=tp_axis)` → reshard back to
     `[batch, P, inner_dim/tp]`. `mesh_partition` of a replicated tensor is local (no CCL).

Remove: `_InjectionMask` class, `self._injection_mask` construction (lines 345–351), the
`mask=` param on `inject_text`, and the `mask=self._injection_mask` args at the two call
sites (447, 463). `inject_text` will need `ccl_manager`, `tp_axis`, and `inner_dim//2`
(the half) available — pass them in (or restructure so the transformer's `forward` performs
the gather/reshard around a slim helper). Exact signature is a plan detail.

Cost: one all-gather + one reshard per injection × (num_layers + num_single_layers) = 46
blocks. Accepted for v1; optimization (gather-free select or ColParallel caption projection)
is a follow-up.

Note (verify in plan): the surrounding blocks (`TransformerBlock`,
`Flux1SingleTransformerBlock`) expect `prompt` feature-sharded on `tp_axis`
(`[batch, P, inner_dim/tp]`), which the reshard restores. Confirm no block internally
assumes the old exact mask layout.

### 2. Config / plumbing
Files: `models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py` (+ any `default()` callers).

- No new preset dict. Keep `sp_axis=0`, `tp_axis=1`, `cfg=(1,0)`; factors derived from
  `mesh_device.shape`. On `(4,8)` this yields SP=4/TP=8 and one 4×8 submesh.
- Pin `num_links` and CCL `topology` for BH Galaxy. Today FIBO uses `num_links=4`
  (`CCLManager(..., num_links=4, topology=ttnn.Topology.Linear)`); Wan uses `num_links=2` on
  its 4×8 preset. Determine the correct value on the physical mesh; if it must vary by mesh
  shape, add a minimal shape→(num_links, topology) lookup (the only preset we introduce).
- Verify `create_submeshes` (`pipelines/cfg.py`) builds the 4×8 submesh (cfg=1) correctly.

### 3. Text encoder (SmolLM3) — no change for v1
File: `models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py` (lines ~96–98, 160–165).
Keep `EncoderParallelConfig.from_tuple((1, tp_axis))` (replicated). Document the follow-up:
route `SmolLM3TextEncoderWrapper`/`SmolLM3TextEncoder` (which already support `is_fsdp` +
TP>1) through `encoder_tp`+FSDP on a `reshape_device` submesh, mirroring
`pipelines/qwenimage/pipeline_qwenimage.py`.

### 4. VAE decoder (Wan) — no code change for v1
File: `models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py` (line ~108, `_vae_ccl_manager`).
Keep `VaeHWParallelConfig.from_tuples(height=(tp_factor, tp_axis), width=(sp_factor, sp_axis))`
→ height 8 / width 4 at 64×64 latent. No divisibility asserts block this. Keep the separate
`_vae_ccl_manager` on the submesh. Documented perf follow-ups (out of scope): add
`_BLOCKINGS` entries in `utils/conv3d.py` for the exact 4×8 conv shapes (else generic
fallback — slower, not wrong); and the adapter passes `height/width=1024` while the VAE's
own output res is 512 (a `patch_size=2` unpatchify runs after decode) — a blocking-table key
mismatch, perf-only.

## Out of scope (non-blocking tuning gaps, documented as follow-ups)

- `sdpa_chunk_size_map` (`transformer_bria_fibo.py:276-282`) has no `(is_bh, 4, 8)` key →
  falls back to `(128, 512)`. Functional. Add a tuned entry later.
- `_register_fibo_matmul_configs()` is tuned for the 12×10 grid at sp2/tp2; TP=8 shapes miss
  it → generic matmul fallback. Functional. Tune later.
- No head padding needed: `num_attention_heads = 24`, `24 % 8 == 0` (3 local heads/device),
  so `PaddingConfig` stays `None` at TP=8.

## Testing & validation

Run on the physical 4×8 BH Galaxy (available). Tests auto-skip without the hardware.

1. **Transformer mesh unit test** — add `(4,8)` to
   `tests/models/bria_fibo/test_transformer.py::test_fibo_transformer_mesh`. First and
   cleanest correctness signal for SP=4/TP=8 + the new `inject_text`. Reference for the
   parametrization style: `tests/models/motif/test_transformer_motif.py`.
2. **Pipeline PCC / golden e2e** — add `(4,8)` to `tests/models/bria_fibo/test_pipeline.py`
   (`test_fibo_pipeline_latent_pcc` and/or the golden-image test) for the full
   encode→prepare→denoise→decode path at 1024².
3. **Perf harness** — optionally extend
   `tests/models/bria_fibo/test_performance_bria_fibo.py` to accept `(4,8)` (it already
   derives factors from `mesh_device.shape`) once correctness passes.

## Risks

- **`inject_text` reshard correctness (primary)** — confirm `ttnn.mesh_partition(..., dim=-1,
  cluster_axis=tp_axis)` reshards the replicated full-inner-dim tensor to the exact
  feature-shard layout the blocks expect. Mirror the VAE/LTX usage. This is the one op
  sequence that must be exactly right.
- **`num_links`/topology for BH Galaxy** — a wrong value fails CCL (loud, not silent). Pin on
  hardware.
- **Latent PCC drift at TP=8** — more TP shards + the new gather/reshard could shift PCC vs
  the 2×2 golden; may need a per-shape PCC threshold in the test.

## Implementation order (for the plan)

1. Generalize `inject_text`, drop `_InjectionMask` (TDD against the transformer unit test).
2. Add `(4,8)` to the transformer mesh unit test; validate PCC on hardware.
3. Pin `num_links`/topology; confirm pipeline config produces the 4×8 submesh.
4. Add `(4,8)` to the pipeline PCC/golden test; validate e2e image on hardware.
5. Document the deferred perf follow-ups (encoder FSDP/TP, VAE blocking, SDPA/matmul tuning).
