# Migrating v3.2 onto v3.1 infra

Overall goal: extend v3.1 to cover v3.2 and GLM-5.1. The current v3.2 and GLM-5.1 code lives in a separate folder; its functionality needs to be merged into v3.1 by extending the existing architecture, test infra, and pipelines.

## Nomenclature
- v3.1 -> DeepSeek v3.1
- v3.2 -> DeepSeek v3.2

## Migration notes & TODOs

Extracted from in-progress changes on branch `mvasilijevic/dsa_w_ops`. These are the
TODOs and notes left inline while bringing up v3.2 / GLM-5.1 on top of the v3.1 MLA.

- [ ] **`deepseek_v32/tt/mla/mla.py`** — `INDEXER_WEIGHT_NAMES` hardcoded; extend the MLA config to cover indexer weights instead of a module-level tuple.
- [x] **`deepseek_v32/tt/mla/mla.py`** — `_build_index_rope_tables`: reuse the v3.1 RoPE table build. **DONE** (commit `797b02198f8`): `get_cos_sin_matrix(interleave, bake_mscale)`; device-verified DS + GLM. See `migration_log.md` (D3, G1.1 progress).
- [ ] **`deepseek_v32/tt/mla/mla.py`** — `_upload_indexer_weights`: move to weight initialization in v3.1.
- [ ] **`deepseek_v32/tt/mla/mla.py`** — `forward`: keep the v3.1 MLA forward. Swap sparse attention and add the indexer. Identify any new architecture parts that need to be migrated to v3.1.
- [ ] **`deepseek_v32/tt/mla/mla.py`** — `forward`: Keep the v3.1 CCL structure.
- [ ] **`deepseek_v3_d_p/tt/mla/mla.py`** — `ttMLA.__init__` `config: PretrainedConfig` param: figure out how to use this for GLM-5.1 and v3.2.
- [ ] **`deepseek_v3_d_p/tests/test_mla.py`** — Modify this file for v3.2 / GLM-5.1.
- [ ] **`deepseek_v3_d_p/tests/test_mla.py`** — Chunked scenarios: extend for v3.2 / GLM-5.1; figure out how padding works.

## CI pipelines
The CI pipelines we have are:

- https://github.com/tenstorrent/tt-metal/actions/workflows/galaxy-deepseek-prefill-tests.yaml -> tests we run on GLX; includes MLA chunked tests and block determinism
- https://github.com/tenstorrent/tt-metal/actions/workflows/demo-sp-release.yaml -> a small set of tests confirming the model is OK to deploy; also includes perf tests on GLX
- https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-e2e-tests.yaml -> LB tests; perf for LB is here too
- https://github.com/tenstorrent/tt-metal/actions/workflows/tt-metal-l2-nightly.yaml -> some op tests may have been moved here; not sure whether that PR landed
- https://github.com/tenstorrent/tt-metal/actions/workflows/t3000-e2e-tests.yaml -> the BH e2e counterpart but on WH only; not that important to us anymore, more about keeping functionality alive

**Note:** weights need to be cached on CI, with traces run on exabox. There are 2 GLX in CI with different DRAM speeds, so perf can be affected.

---

## Goals
1. Unify **architecture** code step by step
    - G1.1 — reuse RoPE from v3.1.
    - G1.2 — use the same CCL pattern as v3.1.
    - G1.3 — extend the v3.1 MLA by adding the indexer and any new functionality. The majority of the code should be shared.

2. Unify **weight** loading and sharding
    - G2.1 — extend the v3.1 code: indexer weights should be added to the existing weight loader/cache.

3. Extend MLA tests to cover v3.2 and GLM-5.1
    - G3.1 — identify current v3.1 tests (accuracy, determinism, perf) that should be extended to cover v3.2 and GLM-5.1. This should be a subset of the existing v3.2 tests. Migrate the identified v3.2 tests to v3.1 infra.
    - G3.2 — figure out what to do with v3.2 tests that cannot be covered by v3.1.
    - G3.3 — measure the total time budget for running existing v3.1 CI tests, and how v3.2 and GLM-5.1 affect it.
    - G3.4 — create a comprehensive overview of existing tests and their coverage, and list ideas to (1) speed up tests and (2) increase coverage.

## Execution Order
- **[DONE]** G1.1, G1.2 — find duplicates in v3.2 (like RoPE) that can be replaced fully or with small modifications.
    - **G1.1 (RoPE) — DONE:** indexer table build now shares v3.1's `get_cos_sin_matrix` (commit `797b02198f8`); `get_rot_transformation_mat` + `RotarySetup` were already shared. Device-verified (DS + GLM).
    - **G1.2 (CCL) — diagnosed + decided (D1):** v3.2's `_tp_rs_ag` / `_tp_ag_reduce` are byte-for-byte v3.1's inline blocks. Team prefers inline over helpers, so no shared-helper refactor; converting v3.2's helpers to inline is optional style churn with no functional benefit — deferred.
- **[IN PROGRESS]** G1.3 — extend the v3.1 MLA to support v3.2: the indexer + DSA sparse-attention forward are now folded into v3.1's `ttMLA` (single class, no v3.2 subclass). Phased — see `migration_log.md` (D4).
    - **P1 DONE:** `ops.py` lives in `deepseek_v3_d_p/tt/ops.py` (v3.1 owns the DSA ops).
    - **P2/P3 DONE:** indexer + DSA methods merged into v3.1 `ttMLA`, gated by `_has_indexer` (set when DSA indexer weights are in the state_dict). `forward` dispatches: indexer present & `end_pos > index_topk` → sparse `_dsa_forward`; else the renamed dense `_dense_forward` (byte-for-byte the old v3 forward). Dense v3.1 has no indexer → path unchanged.
    - **P4 DONE:** v3.2 subclass retired. `deepseek_v32/tt/mla/__init__.py` re-exports v3.1's `ttMLA`; the `deepseek_v32/tt/{mla/mla.py,ops.py}` shims were **deleted** (v3.2 is new code on this branch — no back-compat to preserve, only tests must pass; ops test imports repointed to `deepseek_v3_d_p.tt`).
    - **P5 DONE:** indexer config (`index_n_heads/head_dim/topk/rope_interleave`) is read off the HF `config` via `getattr` (DS defaults); the `index_args` param + the `deepseek_v32.reference_cpu.ModelArgs` import are gone. **v3.1 now has zero references to the v32 package.** GLM sets `index_rope_interleave=True` etc. on its HF config. Device-verified (DS + GLM L0, 15 passed).

  **G1.3 complete.** v3.1 owns one unified `ttMLA`; the v32 package only re-exports it.
- **[DONE]** G2.1 — extend weight loading and caching. `_upload_indexer_weights` now uses `ttnn.as_tensor(cache_file_name=…)` with the same `layer_{idx}.mla.indexer_*` scheme + weight_cache_path as the MLA weights, so indexer weights ride the on-disk cache. Verified: indexer device tests (DS + GLM) unchanged + cache round-trip (5 `indexer_*` files written & reused). Follow-up: the cache-only *pre-build* path (`build_ttnn_cache`, device=None) doesn't yet emit indexer files — they cache on first real load — because indexer presence isn't detectable from config/placeholders alone.
- **[ANALYZED]** G3.1–G3.4 — full test-coverage overview, migration plan, time budget, and speed-up/
  coverage ideas written to [`test_coverage_overview.md`](test_coverage_overview.md). Key findings:
  v3.1 has a `TestVariant` registry (`model_variants.py`) that v3.2/GLM plug into; post-G1.3 the
  v3.2 MLA tests already exercise v3.1's `ttMLA`. Migratable: op shape/numerics + indexer-chunked
  (ops live in v3.1 now) and `test_mla` (becomes a variant). Not migratable: `test_vs_gpu_ref`
  (needs vLLM trace bundles) + `test_mla_perf` (hardcoded mesh) — keep bespoke. **The code-migration
  itself (registering variants, rewiring shared `@parametrize` lists) is a reviewed follow-up — it
  touches shared v3.1 test infra + the DSV3/Kimi gates, so it shouldn't land unattended.**
