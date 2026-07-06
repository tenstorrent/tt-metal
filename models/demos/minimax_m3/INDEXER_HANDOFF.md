# MiniMax-M3 MSA Indexer — Handoff (C++ kernel work)

**Objective:** make `ttnn.experimental.indexer_score_msa` (the MSA block-selection op) match the
fp32 torch reference well enough that the device picks the **same top-16 blocks** as the golden — in
one-shot first, then on the chunked NdShard cache-read path. This is a **prerequisite for correct
multi-chunk (>5120-token) prefill**: the indexer chooses which KV blocks MSA attends, so if its
scores drift, block selection diverges and every downstream MSA layer's attention is wrong.

This doc is self-contained. It does NOT require the originating chat's context.

---

## 1. Where the indexer sits (M3 MSA pipeline)

M3 attention is hybrid: layers 0–2 dense GQA; layers 3–59 **MSA block-sparse**. Each MSA layer:
1. `indexer_score_msa(index_q, index_k_full, chunk_start_idx=cached_len)` → per-(group,query) **block
   scores** `[G, Sq, nblk]` (block_size=128, num_index_heads=4, index_dim=128).
2. Host/op picks **top-16 blocks** (+ forced-local current block + sink block 0), causal-masked.
3. `sparse_sdpa_msa` attends only the selected blocks. (Separate op, separate owner — see §7.)

If the indexer scores are wrong, the WRONG blocks are selected → `sparse_sdpa_msa` attends the wrong
KV → attn_out wrong → residual wrong → compounds over 57 MSA layers.

## 2. The op and all relevant files

- **C++ op dir:** `ttnn/cpp/ttnn/operations/experimental/indexer_score/`
  - `device/indexer_score_device_operation.{hpp,cpp}` — device op
  - `device/kernels/reader_indexer_score.cpp`
  - `device/kernels/writer_indexer_score.cpp`
  - `device/kernels/compute_indexer_score.cpp`  ← **the dot + block-max-pool + fp32-dest fallback lives here**
  - `indexer_score_nanobind.cpp` — exposes BOTH `indexer_score_dsa` (DeepSeek raw-dot variant) and
    `indexer_score_msa` (the M3 per-GQA-group variant). **We use `indexer_score_msa`.**
- **Owner / coordinate with:** **Slavko Krstic** (sole author of this op, 6 commits). Likely upstream too.
- **Python call site:** `models/demos/minimax_m3/tt/attention/msa.py`
  - `msa_indexer_sparse(...)` (~L116) calls `ttnn.experimental.indexer_score_msa(...)` (~L155)
  - `gather_natural(t)` (~L300) + `_blockcyclic_to_natural(...)` (~L238) — the chunked cache-read
    reorder path (the #4 mis-order suspect below)
- **Torch reference (golden math):** `models/demos/minimax_m3/reference/model.py`
  - `msa_block_selection(index_q, index_k, scale, block_size, topk_blocks, sink_block)` (~L223):
    scaled index dot → causal −inf for future tokens → block max-pool (score_type "max") →
    force-local current block (+inf) → sink block (+inf) → top-k. Returns bool `[G, S, nblk]`.

## 3. The three indexer problems (scoped)

Scope matters: **one-shot** exercises indexer COMPUTE only (no cache-read); **chunked** additionally
reads `index_k` from the NdShard cache and adds the read-side bugs on top. Multi-chunk hits ALL three
(a broken chunk-1 write poisons chunk-2's read regardless of read-side fixes).

| # | Problem | Where | Scope | Status |
|---|---------|-------|-------|--------|
| 6 | **fp32-DEST accum** for the 128-dim index dot. bf8/bf16 accumulation perturbs scores → hard top-16 selection flips picks. `compute_indexer_score.cpp` has an fp32-dest fallback path ("guarded: no-op in bf16; only the fp32-dest fallback reconfigs") — needs to be validated/enabled for selection stability. Python interim: pass HiFi4 `compute_kernel_config`. | `compute_indexer_score.cpp` | **one-shot + chunked** (compute precision) | OPEN — prime suspect for one-shot `index_k`→0.74 |
| 4 | **index_k gather mis-order.** `gather_natural`/`_blockcyclic_to_natural` (`to_memory_config(DRAM)` un-shard + allgather + row-major reshape) does NOT delinearize the 32-token round-robin NdShard "slab" to logical token order → the indexer scores against a mis-ordered `index_k_full`. Measured `index_k` PCC **0.517** (diverges from row 0). | `msa.py` gather path (interim) / **proper fix = kernel reads NdShard cache directly, slab-aware** | **chunked only** | LOCALIZED — the chunked-first-token differentiator |
| 5 | **Not slab-aware** NdShard cache-read for `index_k`. Same root as #4 — the proper fix is to have the op read the block-cyclic slab in-kernel (like the dense path's `ring_joint_sdpa` does via TensorAccessor), eliminating the `to_memory_config`+AllGather+reshape workaround (which also throws "Logical DRAM core 8-0 outside valid range" when host-reordered). | `reader_indexer_score.cpp` (add slab-aware cache-read) | **chunked only** (>1 chunk / >5120 tok) | OPEN |

**Write-side vs read-side framing (important):** #6 corrupts the `index_k`/scores COMPUTED and written
to cache during chunk-1 (garbage-in). #4/#5 corrupt the READ of that cache in chunk-2. Multi-chunk needs
BOTH clean. **Fix #6 first, gate on one-shot KV-PCC (index_k → ~0.99); then #4/#5, gate on chunked
KV-PCC matching one-shot.** You can't validate the read-side while the write-side still poisons the cache.

## 4. What we HAVE (repro tests, golden, instrumentation)

### (a) Intrinsic indexer accuracy test — the primary repro
`models/demos/minimax_m3/tests/unit/test_indexer_score_msa_accuracy.py`
- Compares `ttnn.experimental.indexer_score_msa` block scores vs the fp32 reference at **real M3 dims**,
  parametrized over `(Sq, T)`. Reports **PCC + elementwise atol/rtol + max-abs-error + worst
  (query,block) cell**, and checks sentinel/±inf causal structure (no opposite-sign infinities).
- Device-guarded — skipped unless `RUN_INDEXER_ACCURACY=1`.
- Run: `RUN_INDEXER_ACCURACY=1 pytest models/demos/minimax_m3/tests/unit/test_indexer_score_msa_accuracy.py -s`
- This is the fast inner-loop for #6 (precision). It does NOT exercise the cache-read (#4/#5) — that
  needs the chunked KV-PCC harness below.

### (b) Golden KV trace (torch reference, fp32 compute, dense+MSA)
`/data/philei/models/minimax-m3-prefill-cache/golden/longbook_8192`
- 8192 tokens, 60 layers, `dense_only=False`, `attention_matches_real_model=True`, dtype bf16 / fp32
  compute. Contains per-layer K / V / **index_k** in `kv_cache/layer_*.safetensors` + `metadata.json`
  (token_ids). Read-only (owned by philei) — copy if you need to write.
- Regenerate (if ever needed): `models/demos/minimax_m3/scripts/generate_golden_kv_cache.py`
  (needs ~426 GB RAM via mmap, ~15–30 min CPU).

### (c) Full-model KV-PCC harness (per-layer K / V / index_k vs golden)
`models/demos/minimax_m3/tests/galaxy_prefill_kv_pcc.py`
- Env: `PREFILL_TRACE_DIR` (the golden dir), `PREFILL_CHUNKED` (0=one-shot, 1=chunked cache-read),
  `PREFILL_CHUNK_SIZE` (default 5120), `PREFILL_NUM_LAYERS=N` (partial model, fast — also sets
  `M3_LOAD_NLAYERS`), `EXPERT_DTYPE=bf8`.
- **One-shot baseline (already measured, current branch):** per-layer decay, dense L0–2 clean (0.999),
  MSA decays with depth → **min K=0.686, V=0.351, index_k=0.743** across 60 layers. index_k: L4≈0.998,
  L20≈0.956, L40≈0.802, L59≈0.801. This is the gate #6 must move.
- **Chunked (yesterday, pre/partial-fix):** cratered — e.g. 3-layer chunked K=0.86 V=0.51; chunked
  first-token was WRONG ('</' instead of '<mm:think>') due to #4.

### (d) On-device instrumentation (env-gated, in msa.py, uncommitted)
- `DBG_MSA_BLOCKIDS=1` → `_dbg_msa_save` dumps `block_scores` / `block_ids` / `index_q` / `index_k` per
  layer. Used to localize #4 (index_q PCC 0.9998 clean vs index_k 0.517 broken). Compare scripts were in
  scratchpad (`diff_blockids` / `diff_scores` / `diff_inputs.py`).
- `M3_INDEX_CACHE_BF16=1` (in `tt/attention/kv_cache.py`) → store `index_k` cache in bf16. Tested: did
  NOT fix chunked selection (proved #4 is order, not cache dtype). Keep as a knob.

## 5. What we NEED (deliverables, prioritized)

1. **#6 fp32-DEST accumulation in `compute_indexer_score.cpp`** — validate/enable the fp32-dest path for
   the 128-dim dot (and block-max-pool) so bf8 doesn't flip top-16 picks. Gate: `test_indexer_score_msa_
   accuracy.py` PCC up + one-shot KV-PCC `index_k` → ~0.99. (There's a known TT_FATAL to drop / LLK to
   validate on the fp32-dest path — see the kernel's guarded reconfig.)
2. **#4/#5 slab-aware NdShard cache-read for `index_k`** in `reader_indexer_score.cpp` — read the
   block-cyclic 32-token round-robin slab directly in-kernel (mirror the dense path's `ring_joint_sdpa`
   TensorAccessor cache-read: `cache` + `kv_cache_batch_idx` + `kv_actual_isl`). Eliminates
   `msa.py::gather_natural` (to_memory_config+AllGather+reshape) and the "Logical DRAM core 8-0 outside
   valid range" FATAL. Gate: chunked KV-PCC `index_k` matches one-shot.

## 6. How to run (env)

```bash
cd /data/vmelnykov/tt-metal && source python_env/bin/activate
export TT_METAL_HOME=/data/vmelnykov/tt-metal PYTHONPATH=/data/vmelnykov/tt-metal
export HF_MODEL=/data/vmelnykov/MiniMax-M3-ref
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
ulimit -u 1048576
# Galaxy reset if device is wedged (~35s): tt-smi -glx_reset

# (a) intrinsic indexer accuracy (fast, no golden needed):
RUN_INDEXER_ACCURACY=1 pytest models/demos/minimax_m3/tests/unit/test_indexer_score_msa_accuracy.py -s

# (c) one-shot KV-PCC vs golden (index_k column is the indexer signal):
EXPERT_DTYPE=bf8 PREFILL_CHUNKED=0 \
  PREFILL_TRACE_DIR=/data/philei/models/minimax-m3-prefill-cache/golden/longbook_8192 \
  python3 models/demos/minimax_m3/tests/galaxy_prefill_kv_pcc.py
# chunked cache-read path: PREFILL_CHUNKED=1 (same golden). Partial model: PREFILL_NUM_LAYERS=6
```

Build after C++ changes: `./build_metal.sh -b Release --build-tt-train` (incremental cmake may NOT
redeploy the loaded .so — do a clean `./build_metal.sh` if a kernel change doesn't take effect).

## 7. Coordination / boundaries

- **`indexer_score_msa` (this work): owner = Slavko Krstic.**
- **`sparse_sdpa_msa` (the attention-output op, SEPARATE): owner = pjosipovic.** It has its own bug set
  (intra-block causal mask leak; slab-aware cache-read; `matmul_init`→`mm_init` fp8-determinism at
  `sparse_sdpa_msa_compute.cpp:76`). Track separately — do not conflate with the indexer.
- All M3 Python (`models/demos/minimax_m3/`) is owned by this team (vmelnykov + Pavlo Hilei). The C++
  ops live under `ttnn/cpp/...` and touch ttnn CODEOWNERS — expect a broad reviewer set on any PR there.
