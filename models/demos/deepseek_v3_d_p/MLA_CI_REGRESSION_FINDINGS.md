# MLA CI regression — findings & HW verification plan

Reported 2026-07-24 (Blackhole e2e + Blaze prefill). Trigger commit:
`b4a103334b0` — "DeepSeek: Add mixed-format KV cache for sparse MLA" (#50207, pjosipovic, merged 07-23 18:40).

There are **two independent problems**. This branch (`ipotkonjak/mla-ci-regression-findings`, off `main`)
already contains the fix for Part A. Part B is what you need HW to confirm.

---

## Part A — three crash regressions (OURS) — FIXED on this branch

#50207 changed `init_mla_kv_cache` to return an `MlaKvCache` **wrapper** (`.storage` / `.geometry` /
`.format`) selected by a `MlaKvCacheFormat` enum, replacing the old raw-tensor + `dtype`/`layout` API.
Three test callers were left on the old API — two of them collide with earlier same-day commits of ours:

| CI failure | Site | Cause | Fix |
|---|---|---|---|
| `AttributeError: 'Tensor' object has no attribute 'geometry'` (Transformer Determinism) | `test_prefill_transformer.py:514` | #50207 passed `tt_kvpe_cache.storage` (raw Tensor); `mla.forward` reads `.geometry` | pass the wrapper `tt_kvpe_cache` (matches normal path @573) |
| `NameError: name 'kvpe_dtype_layout'` (Chunked GLM accuracy) | `test_prefill_transformer_chunked.py:680` | our `5c881bf` added a preload using `kvpe_dtype_layout`; #50207 deleted its definition | derive `cache_format.storage_dtype` / `.storage_layout` |
| `TypeError: init_mla_kv_cache() got 'dtype'` (Chunked GLM 55k) | `test_prefill_transformer_chunked.py:1282` | our `b801256` had `**kvpe_dtype_layout`; #50207 swapped in `init_mla_kv_cache` (rejects it) and hardcoded `BFP8_TILE` | drop kwarg; `BF16_RM if resolve_has_indexer else BFP8_TILE` |

### Verify Part A is green on HW (8x4 Blackhole):
```bash
pytest "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer.py::test_ds_prefill_transformer[blackhole-deepseek_v3-mesh-8x4-iter2-with_determinism-e256_device_fp32-5_layers-25600-8-balanced-smoke-longbook_qa_eng-random-kv_cache-temp_sweep-right_pad]"
pytest "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_glm_prefill_transformer_chunked[blackhole-glm52-mesh-8x4-L78-warm_cache]"
pytest "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_glm_prefill_transformer_chunked_no_pcc[blackhole-glm52-mesh-8x4-L78-preload0-chunks_eleven-ten_iters]"
```
These are Python-only fixes — no rebuild needed. Expect all three to PASS on this branch (they crash on `main`).

---

## Part B — `scaled_fp8` PCC collapse (NOT ours; a main-side LLK fp8 regression)

The new `scaled_fp8` cases fail with output PCC **~0.009–0.025** (vs the 0.98 gate) — i.e. essentially
uncorrelated. But the **same cases PASSED at ~0.995 on the author's own Blaze prefill run** (branch
`pjosipovic/fp8-sparse-kv-cache`, SHA `5809b2b25e`, 07-23 10:24, jobs 89183467174 / 89183467130).
bf16 cases pass on both. So the feature is sound; **main regressed the fp8 path.**

### Regression window (all 07-23)
| time | event |
|---|---|
| 10:24 | author branch CI: `scaled_fp8` **GREEN** (~0.995) |
| 14:44 | **#49119** `reconfig_data_format — always derive int8 state` (#47381) merges to main |
| 16:12 | **#49473** `unpack tilize fp8 perf feature` merges to main |
| 18:40 | #50207 squash-merged **on top of** both |

Both suspects landed **after** the green run and are **Blackhole LLK** (matches the BH-only failure):

- **#49119 (`6af7eec8584`) — top suspect.** Rewrites `tt_metal/hw/inc/api/compute/reconfig_data_format.h`
  (+333) **and directly edits `.../compute/compute_per_token_cast_to_fp8.cpp`** — the exact kernel
  `MlaKvCache._pack_scaled_fp8` invokes via `per_token_cast_to_fp8`. "Always derive int8 state" alters
  the shared 8-bit (fp8/int8) format state machine → would corrupt fp8_e4m3 while leaving 16-bit bf16
  intact. Its revert **conflicts in `.../sdpa/device/kernels/compute/sparse_sdpa_compute.cpp`** (the
  sparse-SDPA compute kernel that consumes the scaled_fp8 cache), further tying it to this path.
- **#49473 (`3b99e8cc934`) — second suspect.** Rewrites Blackhole `llk_unpack_tilize.h` (+217) and the
  datacopy LLK. The scaled_fp8 SDPA consumer unpacks/tilizes the fp8 cache. Reverts **cleanly**.

### Reproduce the failure on this branch (8x4 Blackhole):
```bash
pytest "models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py::test_sparse_mla_accuracy[blackhole-fabric2d-deepseek_v32-8x4-seq5120-kv_scaled_fp8]"
pytest "models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py::test_sparse_mla_chunked[blackhole-kv_scaled_fp8-c1k-fabric2d-deepseek_v32-8x4-seq5120]"
pytest "models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py::test_sparse_mla_accuracy[blackhole-fabric2d-glm_5_1-8x4-seq5120-kv_scaled_fp8]"
pytest "models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py::test_sparse_mla_chunked[blackhole-kv_scaled_fp8-c1k-fabric2d-glm_5_1-8x4-seq5120]"
```
Expect output PCC ~0.01 (FAIL). The bf16 sibling (`-kv_bf16` / drop the `-kv_scaled_fp8` suffix) should PASS — use it as the control.

### Confirm the culprit (these are C++/kernel changes → **rebuild required** after each revert)
**Test #49473 first (clean revert):**
```bash
git revert --no-commit --no-edit 3b99e8cc934   # #49473 unpack tilize fp8
# <rebuild metal/kernels> then re-run the 4 scaled_fp8 cases above
# PCC recovers to ~0.99  => #49473 is the culprit
git revert --abort   # or: git reset --hard HEAD to undo the staged revert
```
**Then #49119 (revert conflicts in sparse_sdpa_compute.cpp — resolve manually):**
```bash
git revert --no-commit --no-edit 6af7eec8584   # #49119 reconfig_data_format
# resolve the conflict in
#   ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sparse_sdpa_compute.cpp
# (keep #50207's scaled_fp8 additions, drop #49119's reconfig change), git add, then rebuild + re-run.
# PCC recovers => #49119 is the culprit.
```
If neither alone fixes it, they may interact — revert both. Loop in **Skrsmanovic** (#49473) and the
**#47381 author** (#49119), plus **pjosipovic** (#50207).

**Do NOT relax the PCC thresholds or xfail-as-broken** — that masks a real cross-team LLK regression.

### Perf "outside range" (BH 2x4)
Not a real slowdown: #50207 renamed the perf case IDs / CSV output dirs and added a new
`sparse-kv_scaled_fp8` case with no baseline. The external regression check is flagging a
missing/renamed baseline. Re-baseline after Part B is resolved.
