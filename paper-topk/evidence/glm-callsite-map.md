# GLM / LM-head Top-K Call-Site + Test Map
Scouted 2026-08-17 on nkapre/sorting (read-only). Blaze intel: repo IS checked out locally at `/home/nachiket/tt-blaze` (main @ d39ab22) — mined directly; Glean used only for issue #1971 text.

## Summary table

| # | Site | Engine | Shape (rows × W, k) at real config | Disqualifier args | Served by our op/routing today? | Ledger status |
|---|------|--------|-----------------------------------|-------------------|--------------------------------|---------------|
| 1 | GLM/DSA indexer prefill — `models/demos/deepseek_v3_d_p/tt/mla/indexer.py:737` | `ttnn.experimental.topk_large_indices` | Galaxy SP8xTP4, chunk 5120: **160 × T, k=2048** (T = prealloc ctx width, 65536 @64k; `valid_length=end_pos` grows 5120→T per chunk) | none (no indices_tensor/sub_core_grids/stable) | **YES — is the op by name** | COVERED: `dsa_indexer_k2048` (verify note below) |
| 2 | Same site, DeepSeek-V4 variant (k=512) | topk_large_indices | 160 × 65536, k=512 | none | YES | COVERED: `dsa_indexer_v4_k512` |
| 3 | Same site, loudbox meshes (8 dev) | topk_large_indices | (2,4): **320 × T**; (4,2): **640 × T**, k=2048 | none | YES | **NEW candidate row** (row-count axis) |
| 4 | Same site, single-chip (1,1) — what test_sparse_mla runs on p150a | topk_large_indices | **5120 × T (T≈8192 default), k=2048** | none | YES | **NEW candidate row** (p150a-runnable geometry) |
| 5 | MiniMax-M3 MSA — `models/demos/minimax_m3/tt/attention/msa.py:147` | topk_large_indices | 1M ctx / block 128: rows × **8192 blocks, k=16** | none | YES (floor-k corner) | COVERED: `msa_blocks_k16` |
| 6 | LM-head sampling chain — `models/common/sampling/tt_sampling.py:836-920`, `models/common/modules/sampling/sampling_1d.py:543-614` | `ttnn.topk` k=32 | 32 × {65536, 32768, 64128} | relaxed where route fires (I5, 575ff18a1be); sub-grid-pinned calls never relaxed | YES where predicate fires | COVERED: `sampling_qwen36_tp4`, `sampling_tp8_pow2`, `sampling_1chip_split` |
| 7 | Log-probs narrow stage — `models/common/sampling/tt_log_probs.py:586` | `ttnn.topk` k=32 | batch × **256**, k=32 | **sub_core_grids passed** (pinned) | NO — and provably can't route (W=256 below every threshold) | **NEW no-change control row** candidate |
| 8 | B1 decode LM-head — `models/demos/deepseek_v3_b1/unified_kernels/sampling.hpp` + `micro_ops/sampling/op.py:74` | **fused top32_rm LLK** (never calls ttnn.topk) | vocab 129280, 101 matmul cores, k=32 | n/a — LLK-fused pipeline | NO by design (blaze-style in-tree twin) | **NEW context row** (engine "b1-fused") |
| 9 | Blaze decode distributed indexer — `/home/nachiket/tt-blaze/blaze/ops/distributed_indexer/` | FusedProgram (SDPA→local topk→cross-device merge), topk_xl LLK | per device: 8 banks × 16384 pos (128k/dev) or 32k/dev, **k=2048**; 8-device loudbox | explicit `topk_cores` list, L1-sharded streamed input | NO — FusedProgram-locked; grid-interface gap (§E) | blaze cell exists (24.4µs, comp3); 128k/dev variant unmeasured |
| 10 | Blaze LM-head top-32 — `/home/nachiket/tt-blaze/blaze/ops/{local_top_k? via run_top32_llk,cross_core_topk_merge,cross_device_top32_merge}` | top32_rm LLK, k≤32 (32 = sorted-run width) | per-device vocab shard → 32-cell runs → 2-stage mesh merge | n/a | NO (LLK pipeline) | context only |

## A. GLM prefill top-k in tt-metal

**GLM prefill routes through the SHARED DSA indexer** — there is no separate GLM demo dir. GLM-5.1/5.2 are adapters over `deepseek_v3_d_p`:

- Adapter: `models/demos/deepseek_v3_d_p/tt/runners/adapters/glm_5_1.py` (subclasses `MLAPrefillAdapter`; hand-built config via `reference/glm_5_1_config.py`). GLM-5.2 sibling adds cross-layer indexer reuse.
- GLM config (`reference/glm_5_1_config.py:46-48`): `INDEX_TOPK = 2048`, `INDEX_HEAD_DIM = 128`, `INDEX_N_HEADS = 32`. HF repo `zai-org/GLM-5.1-FP8`.
- **The call site** — `models/demos/deepseek_v3_d_p/tt/mla/indexer.py:737`:
  ```python
  idx = ttnn.experimental.topk_large_indices(logits, k=self.index_topk_capacity, valid_length=topk_valid_length)
  ```
  - `index_topk_capacity = min(index_topk, seq_len)`, asserted in [16, 2048] and %16 (indexer.py:314-317). GLM at real ctx → k=2048.
  - Rows per call = chunk/(sp·tp) after TP×SP query split (indexer.py:669-688): Galaxy (8,4) with chunk 5120 → **160 rows**. Loudbox (2,4) → 320; (4,2) → 640; single chip → 5120.
  - `logits` width = full preallocated ctx width T with a STALE tail; `valid_length=end_pos = start_pos + chunk_global` bounds the search (indexer.py:694-737). **W grows per chunk**: 5120, 10240, … → T (65536 @64k ctx).
  - Chunk size: `models/demos/deepseek_v3_d_p/tt/tt_prefill_runtime.py:30` — `chunk_size: int = 5 * 1024` = **5120 confirmed**. Production GLM-5.2 trace is 55k → 56320 rows = 11 × 5120 chunks (`models/demos/common/prefill/tests/test_producer_runner_e2e.py:29-31`).
  - Called once per sparse-MLA layer per prefill chunk (GLM-5.1: all 78 layers full; GLM-5.2: `ReuseIndexer` — shared layers reuse a full layer's indices, fewer topk calls).
  - No routing disqualifiers — the model calls our op **by name**; indices come back ROW_MAJOR uint32; 0xFFFFFFFF sentinel for -inf/pad columns consumed by sparse_mla.
- Decode for GLM is NOT in this repo (colleague intel confirmed: blaze owns it; `deepseek_v3_d_p` is the prefill stack, `deepseek_v3_b1` is the in-tree blaze-style decode twin).

**Op signature** (`ttnn/cpp/ttnn/operations/experimental/topk_large_indices/topk_large_indices_nanobind.cpp:92-97`): `(input_tensor, k, valid_length=None, return_values=False, num_slices=None)`. `valid_length` is runtime (no recompile — a loop growing it reuses one program). **No grid/sub_core_grids parameter exists.**

## B. Tests exercising the prefill sites

| Test | Hardware | p150a-runnable? | Notes |
|------|----------|-----------------|-------|
| `models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py::test_sparse_mla_{accuracy,rotated,indexer_reuse,determinism,chunked,kv_only}` | BH-only (`skipif not is_blackhole`), box-adaptive: 1 dev→(1,1); 4→(2,2),(1,4); 8→(2,4),(4,2); 32→(8,4),(8,2) | **YES** — fallback mesh (1,1); ids like `glm_5_1-1x1-seq5120[-fp8/bf16]` | Variants `deepseek_v32`, `glm_5_1`, `glm_5_2`; seqs 256 (inert top-k) + 5120 (real pruning, anchor). Runs indexer forward → topk_large_indices live. Needs GLM weights (HF `zai-org/GLM-5.1-FP8` or ref-cache env vars) — check availability before running |
| `.../sparse_mla/test_sparse_mla_perf.py` | **Galaxy 8×4 only** | no | Per-op device time incl. a `TopkLargeIndices` bucket (line 492) via realtime profiler |
| `.../sparse_mla/test_sparse_mla_vs_trace.py` | multi-dev + `/mnt` golden vLLM trace | no (no mount) | Monkeypatches `ttnn.experimental.topk_large_indices` to capture the head-summed logits feeding top-k |
| `.../tests/test_mla.py::test_mla_chunked_prefill` | meshes (2,2)/(2,4)/(8,4) | no | Variants dsv3/kimi/k3 only — **no GLM here**; GLM MLA coverage lives in the sparse suite |
| `.../tests/test_prefill_transformer_chunked.py`, `test_prefill_block_chunked.py` | **8×4 galaxy** + `/mnt` caches | no | GLM-5.1/5.2 layer-PCC pins in comments (indexer-K nope 0.952 @L52 etc.) |
| `models/demos/common/prefill/tests/test_producer_runner_e2e.py` | galaxy e2e | no | `glm52_full_depth_kv_table` scenario, GLM52 55k trace, PREFILL_MODEL=glm_5_2 |
| `tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py` + contract suite + `_canonical_topk_sweep.py --model-scenarios` | single BH chip | **YES** (our existing harness) | The measured-cell machinery; scenario rows below |
| Blaze: `tt-blaze/tests/blaze/micro_ops/dsa/test_indexer_local_topk.py::test_indexer_local_topk[{128k_4cpb,32k_2cpb}-pos*]` | **p150 single device, requires grid (11,10)** | YES (blaze env) | k=2048, cur_pos sweep incl. boundary ties; asserts bit-exact vs torch |
| Blaze: `.../dsa/test_indexer_sdpa_local_topk.py::test_glm52_indexer_sdpa_streaming_local_topk[64k]` | p150 | YES — already our comp3 `--with-blaze` cell (24.4µs median) | fused SDPA+localTopK |
| Blaze: `.../dsa/test_cross_device_allgather_tree_merge.py` | multi-device (loudbox) | no | the decode top-2048 cross-device merge |

## C. LM-head top-32 (token selection)

Two disjoint engine families:

1. **ttnn.topk chain (patched, I5)** — `models/common/sampling/tt_sampling.py:836-920` + `models/common/modules/sampling/sampling_1d.py:543-614`, consumed by ALL `models/common/models/*` and tt_transformers decode (per-device `ttnn.topk` k=32 → all-gather of [*,32] tuples → `ttnn.sampling`; see `models/common/models/executor.py:465-471`). Relaxation mirror `topk_would_route_to_large_indices` in `models/common/sampling/_utils.py`. Already ledger rows `sampling_*`. Quasar clones under `models/experimental/llama32_1b_quasar/` are unpatched copies (non-BH arch, out of scope).
2. **Fused top32_rm LLK (b1 / blaze lineage)** — never calls ttnn.topk:
   - In-tree: `models/demos/deepseek_v3_b1/unified_kernels/sampling.hpp` + `micro_ops/sampling/op.py` (golden `torch.topk(k=32)` at :74; vocab 129280, 101 matmul cores, CCL broadcast). Tests: `models/demos/deepseek_v3_b1/tests/unit_tests/test_lm_head_sampling.py` (has single-device `skip_ccl=True` mode — likely p150a-runnable) and `test_sampling.py`. LLK shadow tree `models/demos/deepseek_v3_b1/kernel_includes/.../top32_rm*`. This is the "any test would hit the top32" claim — true for the b1/blaze decode stack, not the ttnn chain.
   - Blaze: `run_top32_llk.hpp` driver → `local_top_k`, `cross_core_topk_merge`, `cross_device_top32_merge` ops (k≤32; 32 is the LLK's fixed sorted-run width — k can be 1/8/32). Tests `tt-blaze/tests/blaze/micro_ops/sampling/{test_local_top_k,test_cross_core_topk_merge,test_cross_device_top32_merge,test_sampling}.py`.

Other ttnn.topk sites checked and classified NOT-LM-head: `tt_log_probs.py:586` (k=32, **W=256**, `sub_core_grids` pinned — second-stage narrowing, can't route, control-row candidate); MoE gates (mixtral_moe.py:117 & grok k=32/W64-pad, gpt_oss/tt/topk.py:26, gemma4 router.py:116, tt_moe_gate.py:639, tt_moe_gate_prefill.py:875 — gate controls already in ledger); informer/tt/ops.py:263 (ProbSparse, tile-padded k, experimental); tt_symbiote default_dispatcher:963/982 (generic torch-dispatch fallback).

## D. Blaze side (local checkout, main @ d39ab22)

- **Decode distributed indexer** (`blaze/ops/distributed_indexer/op.py`): one FusedProgram chaining `DsaIndexerSdpa` → `IndexerLocalTopK` → `CrossDeviceAllgatherTreeMerge`. Top-2048 per device, then 8-device tree merge — the loudbox top-2048 decode from the colleague intel.
- **Placement constraint — the exact mechanism** (`blaze/ops/distributed_indexer/config.py:31-34, 98-109`): 8 DRAM banks/device (hard-coded); per bank the SDPA group is 1×4 cores (`dsa_compute_grid_groups`); the topk cores are the 1×4 row **directly below** each SDPA group, `topk_y = (sdpa_y + 1) % grid_h` — disjoint by construction, 32 SDPA + 32 topk cores. GLM5 32k/dev variant: 2 topk cores/bank + 1 chunk/core (16 topk cores). SDPA **streams** scores to the topk cores' L1 in 2048-position chunks (`STREAM_CHUNK`, 2 tiles) — this is the latency cut; topk input is HEIGHT_SHARDED L1 on the explicit `topk_cores` list. Emitted index encodes bits 0-13 within-bank (16384/bank), 14-16 bank, 17-19 device.
- **IndexerLocalTopK** (`blaze/ops/indexer_local_topk/op.py`): binary tree over the 32 topk cores (core_id = block*4+slot), fused→unfused transition at the bank boundary; per-bank validity from `cur_pos` metadata; early-out when valid ≤ K. Its sort/merge is **topk_xl** (test comment: fp32 DEST enabled at runtime inside a bf16-dest-acc FusedProgram; interim CBs uint32 so packing can't denormal-flush the index bits).
- **topk_xl vendoring status**: blaze carries a full shadow copy under `blaze/kernels/kernel_includes/.../experimental/ckernel_sfpu_topk_xl.h` etc. It has **diverged from tt-metal's canonical copy** (1304 diff lines vs `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_topk_xl.h`; ours carries the INT32-mode fused load/store denormal protection and, branch-only, the SFPLOADMACRO scheduling 725748766a6).
- **Issue #1971 status** (Glean, github.com/tenstorrent/tt-blaze/issues/1971, rdjogoTT, updated 2026-07-30): it is an **inventory + migration ANALYSIS, not a landed migration**. topk_xl is classified §2B blaze-only (users: glm_5_1/glm_5_2 distributed_topk, glm_sparse_read_sdpa, cross_device_topk_merge; shared sampling), "suitable but needs a golden + multi-face harness"; canonical-dir promotion recommended only for durable LLKs — topk_xl named. Since then, upstream tt-metal main landed "[LLK] Add topk_xl tests" (#51777, df31fd4a847, 2026-08-05) — canonical topk_xl + tests now exist in tt-metal, but blaze still consumes its own fork → the drift hazard #1971 warns about is live. top32_rm is §2A (forked copy of the tt-metal b1 shadow tree, already tested upstream via test_top32_rm_dev.cpp).

## E. Synthesis

**Verified**: `dsa_indexer_k2048` (160 × 65536, k=2048) IS the GLM prefill shape at production config — chunk 5120 (`tt_prefill_runtime.py:30`), Galaxy SP8×TP4 → 5120/32 = 160 rows, T=65536 at 64k ctx, GLM `INDEX_TOPK=2048`. Ledger row callsite/shape stand.

**New fact on the covered row**: the call site now passes `valid_length=end_pos` — the effective width grows 5120 → T across the 11 chunks of a 55k prefill; the fixed-W ledger cell measures only the last-chunk worst case. Our op applies valid_length at runtime (no recompile), so a **valid_length sweep column** (same program, W_eff ∈ {5120, 10240, …, 57344}) is a new harness axis the current cells don't exercise — it also interacts with the row-parallel chunk-skip math (early chunks have tiny W_eff). Also: k snaps to min(2048, seq) — short prompts run smaller k.

**NEW scenario-ledger row candidates** (harness cell specs, no measurements):
1. `glm_prefill_loudbox` — rows=320 (mesh 2,4) and rows=640 (mesh 4,2), W=65536, k=2048, bf16, engines [op, routed]; callsite indexer.py:737. Row-count axis between 160 (galaxy) and 5120 (1chip).
2. `glm_prefill_1chip` — rows=5120, W=8192 (PREFILL_MAX_SEQ_LEN default), k=2048, bf16, engines [op, routed]; **this is the exact geometry `test_sparse_mla.py::test_sparse_mla_accuracy[glm_5_1-1x1-seq5120]` runs on our p150a**.
3. `glm_prefill_validlen` — rows=160, W=65536 fixed alloc, valid_length ∈ {5120, 20480, 40960, 57344}, k=2048; measures the real per-chunk cost profile of a 55k prefill.
4. `logprobs_k32_w256` — rows=batch, W=256, k=32, STOCK-ONLY no-change control (sub_core_grids-pinned; W below every routing threshold) — same spirit as the existing gate controls.
5. `b1_lm_head_top32` — context row, engine "b1-fused" (top32_rm LLK, vocab 129280); not servable by ttnn.topk routing by design; useful as the in-tree analog of blaze's LM-head for the sampler-contract conversation (Bazyli/Saad).
6. `blaze_decode_128k` — comparison-only: blaze local-topk at 131072 positions/dev (4 cores/bank) vs our op at rows=1, W=131072, k=2048 (near our 2^19 width cap, single-row column-parallel) — quantifies the decode-shape gap the FusedProgram currently owns.

**Blaze compatibility gap (grid interface)** — flag for the op's roadmap: our op exposes only `num_slices`; the factory self-places slices, input must be interleaved TILE bf16, and **there is no core-placement parameter at all** (the sampling chain's routing predicate already treats sub-grid pinning as a hard disqualifier for the same reason). Blaze's constraint is stronger than "accept sub_core_grids": (1) topk cores must be an explicit list (the 1×4 rows below each SDPA group, disjoint from SDPA's 32 cores); (2) input arrives HEIGHT_SHARDED in those cores' L1, streamed in 2048-position chunks with per-bank validity (not a materialized DRAM tensor); (3) indices must carry bank/device bit-stamps; (4) co-residency inside a FusedProgram. A `sub_core_grids`/CoreRangeSet param on `topk_large_indices` closes only (1) — worth doing as the first adoption step (and needed anyway if any ttnn caller wants to keep the op off SDPA cores), but (2)-(4) are why blaze #1971's practical near-term item is LLK-level unification (one canonical topk_xl consumed by both repos — upstream #51777 already gives it a tested home; our unpushed SFPLOADMACRO work is the coordination piece).
