# GLM-5.1 on Galaxy — bring-up briefing (read me first on the Galaxy machine)

> **Purpose:** everything an agent needs to continue the GLM-5.1 bring-up on a **Galaxy (Wormhole)**
> box, given the work was done & verified on a **LoudBox (Blackhole, 8 chips)**. Read this, then
> `GLM_5_1_MOE_BLOCK.md` (architecture + the dispatch-overflow story) and `GLM_DSA_LEARNING.md`
> (the indexer/topk/sparse_mla kernels) for depth.

---

## 1. What exists (built + verified on LoudBox)

All in `models/demos/deepseek_v32/tests/test_vs_gpu_ref.py` (GLM tests are additive on top of the
DeepSeek-V3.2 trace tests):

| test | what it does | LoudBox status |
|---|---|---|
| `test_glm_block_device_vs_reference` | one whole decoder block (MLA+DSA+MoE) vs `decoder_output_layer_L` | ✅ L0/L1 dense 0.9996, L30/L60 MoE 0.998 |
| `test_glm_block_device_vs_reference_galaxy` | same block, Galaxy meshes | ⏳ wired, **never run on real Galaxy** |
| `test_glm_model_vs_reference` | **streaming whole-model chain** (build block L → fwd → free → L+1), PCC running-hidden vs trace each layer | ✅ all 78 layers PASS, no overflow |
| `test_glm_model_vs_reference_galaxy` | same chain, Galaxy `(8,2)` | ⏳ wired, **never run on real Galaxy** |
| `test_glm_*_host/device_vs_reference`, `test_trace_*`, `test_topk_*`, `test_logits_*` | MLA/indexer/topk/kv op-level + trace sanity | ✅ |

The whole-model test **streams** one `TtPrefillBlock` at a time (build → forward → `del`+`gc` → next),
because the resident `TtPrefillTransformer` builds all layers in `__init__` and 78 GLM MoE layers ≈
45 GB/chip won't fit on 8 chips. It chains `decoder_input_layer_0` (the post-embed hidden) through the
layers and PCCs the **running hidden** vs `decoder_output_layer_L` — it measures error ACCUMULATION,
not teacher-forcing. It covers the **decoder stack only** (no embed / final `model.norm` / `lm_head`).

GLM-5.1 decoder = **pure pre-norm** (verified from the `zai-org/GLM-5.1` checkpoint: each layer has only
`input_layernorm` + `post_attention_layernorm`; **no** GLM-4 `post_self_attn_layernorm`/`post_mlp_layernorm`
sandwich). Between layers = just the residual stream. So the `TtPrefillBlock` (DeepSeek pre-norm) is faithful.

---

## 2. Galaxy mesh constraints (IMPORTANT — verified against Galaxy docs)

- A **single Galaxy is 8×4** (32 chips); valid sub-meshes go up to 8 rows × 4 cols.
- **GLM needs `tp ≤ 2`** (64 q-heads; sparse_sdpa needs per-chip H = 64/tp ≥ 32). So the native `(8,4)`
  [tp=4] mesh is **NOT usable** for GLM.
- → On **one Galaxy use `(8,2)`** [16 chips, sp=8, tp=2]. This is the only GLM-viable Galaxy mesh on a
  single chassis.
- **`(16,2)` cannot open on a single Galaxy** (16 rows > the 8-row bound) — it needs a **multi-Galaxy
  scale-out**. The block test lists `(16,2)` too (it auto-skips off one Galaxy via `requires_mesh_topology`);
  the **model test is `(8,2)`-only** by request (`_GALAXY_MODEL_CASES`).
- Galaxy needs **`FABRIC_2D` + `RELAXED_INIT`** and **`num_links=2`** (vs LoudBox's `FABRIC_1D`, 1 link).
  This is encoded in `_moe_device_params(FABRIC_2D, relaxed_init=True)` + the `_GALAXY_*_CASES` params.

---

## 3. How to run on Galaxy

```bash
source /localdev/nmilicevic/tt-metal/python_env/bin/activate    # or your env

# Whole model, (8,2), full 78 layers, with a cache dir (builds it on first run):
GLM_MODEL_LAYERS=78 GLM_USE_CACHE=/path/to/glm_cache \
  python -m pytest models/demos/deepseek_v32/tests/test_vs_gpu_ref.py \
  -k "glm_model_vs_reference_galaxy" -s

# A single decoder block on (8,2) (fast smoke; pick layers via GLM_BLOCK_LAYERS):
GLM_BLOCK_LAYERS=0,30 python -m pytest models/demos/deepseek_v32/tests/test_vs_gpu_ref.py \
  -k "glm_block_device_vs_reference_galaxy" -s
```

Env knobs (same as LoudBox):
- **`GLM_MODEL_LAYERS`** — how many layers to chain (default 2; `78` = full main decoder; stops early at
  the first unpulled trace layer). [layer 78 in the checkpoint is the MTP head, NOT a main layer — don't chain it.]
- **`GLM_USE_CACHE=<dir>`** — cache ROOT path; a subfolder `sp{sp}xtp{tp}_{dtype}` is created under it.
  Unset → real-load every layer (slow but authoritative PCC). **The cache is mesh-keyed**, so the LoudBox
  `sp4xtp2_bf4` cache does NOT apply — Galaxy builds its own `sp8xtp2_bf4` cache on first run.
- **`GLM_MODEL_CAPACITY_FACTOR`** — default **8** (the fix, see §4). Leave it.
- **`GLM_MODEL_EXPERT_DTYPE`** — `bf4` (default) / `bf8` / `bf16` (bf8/bf16 may `TT_THROW`; the MoE kernel is bf4-only).

---

## 4. Key findings to carry over (don't re-investigate)

1. **The dispatch-buffer-overflow fix = `dispatch_buffer_capacity_factor` 2 → 8 (now the default).**
   Symptom on LoudBox: the chained model collapsed to `inf` at layer 7. Root cause (kernel-traced + empirically
   confirmed): at an imbalanced layer the tile-padded per-expert **region offset** exceeds the flat dispatch
   buffer (`max_dispatch_buffer_token_size = dispatch_group_size × seq_per_chip × factor`), tokens are silently
   dropped, combine leaves slots unwritten, and the reduce reads uninitialized DRAM → `inf`. Factor 8 sizes the
   buffer for the worst case (every token's 8 experts on one chip) → no drops. **On Galaxy `(8,2)`:**
   dispatch_group=8, seq_per_chip=5120/8=640, so `max_dispatched_tokens_per_expert = 8×640 = 5120`, buffer =
   `5120×8 = 40960` — same as LoudBox; factor 8 still correct. (TODO worth doing: promote the `tt_moe.py:~665`
   overflow check from `return_intermediates`-only to an always-on assert.)
2. **Whole-model PCC curve (LoudBox sp4xtp2, factor=8, real-load):** 0.9996 (L0) → smooth erosion → ~0.88
   trough (L62) → recovers to 0.955 (L77); ~45 leading layers ≥ 0.90; test PASSES. The deep erosion is
   **bf4/bf16 chained-precision drift, NOT the overflow** (factor=8 ≈ the old init_zeros mask at depth) — would
   need bf8/fp8 experts (kernel-blocked) to improve. Expect a similar shape on Galaxy.
3. **The cache is faithful on *write* but ~0.0025 mean / ~0.01 max lossier on *load*** (cached bf4 vs fresh
   convert). Use real-load (no `GLM_USE_CACHE`) for the authoritative PCC; the cache is for fast iteration.

---

## 5. Galaxy risks / open questions (verify early on the real box)

- **Can ttnn actually open `(8,2)` on a physical Galaxy?** The `_galaxy` tests are wired with
  `requires_mesh_topology(mesh_shape=(8,2), topology="mesh-8x2")` but have **never run on real Galaxy hardware**.
  First step on Galaxy: run the *block* galaxy test (fast) before the full model.
- **FABRIC_2D + RELAXED_INIT** path for the GLM MoE on Galaxy is unexercised — watch the dispatch/combine CCLs.
- The whole-model run downloads the **gated `zai-org/GLM-5.1`** checkpoint (~0.8 TB fp8) JIT and needs the
  **bit_sculpt trace** (`<repo>/bit_sculpt/results/glm-51`, git-LFS-pulled per layer). See §6.

---

## 6. Prerequisites (a fresh machine needs both)

- **Trace:** clone the `bit_sculpt` repo (correct branch) so it sits at `<repo>/bit_sculpt/results/glm-51`
  (sibling/in-tree; an env var overrides the dir), then `git lfs pull` the `decoder_io/decoder_output_layer_*`
  (+ `decoder_input_layer_0`) streams. Missing/un-pulled → graceful skip / early-`break`, not a crash.
- **Weights:** auto-downloaded via `hf_hub_download` from **`zai-org/GLM-5.1`** (gated) into the HF cache —
  needs an HF token with access + network + ~0.8 TB disk (fp8 e4m3 + per-128-block scale). No manual step,
  but access + disk are hard requirements.
- **Cache (optional):** first `GLM_USE_CACHE=<dir>` run builds `<dir>/sp8xtp2_bf4/` (~hundreds of GB bf4);
  reruns load it (~30× faster, slightly lossy).

---

## 7. The PR diff (4 files, all additive / env-gated)

`tests/test_vs_gpu_ref.py` (block+model tests, `_run_glm_block`/`_run_glm_model`, caching, galaxy variants),
`reference_cpu/weights.py` (`load_moe_block_weights`/`load_dense_block_weights`), `tt/tt_prefill_block.py`
(`index_args` + `emb_dim` GLM support), `context/GLM_5_1_TRACE.md`. The shared `tt_moe.py` / `mla.py` are
**untouched** (the investigation diagnostics were all reverted). The only behavioral change is the
capacity-factor default (8).

Related docs: `GLM_5_1_MOE_BLOCK.md` (block arch + dispatch-overflow case study), `GLM_DSA_LEARNING.md`
(indexer/topk/sparse_mla kernels + dtypes), `GLM_5_1_TRACE.md` (trace streams + run commands).
