# Task: fix `sparse_sdpa_msa` accuracy (numerical precision + diagonal causal mask)

You are working on the tt-metal C++ kernel op **`ttnn.transformer.sparse_sdpa_msa`** — the block-sparse
MSA attention used by MiniMax-M3 decoder layers 3–59 on Blackhole. This is **kernel work, orthogonal to
the MiniMax-M3 model code** — you should not need to touch `models/demos/minimax_m3/` except to run the
end-to-end validation harness at the very end.

Repo: tt-metal. Op is **Blackhole-only**. Owner of the op: **pjosipovic**. A related in-flight mask PR is
**#48700** by **zbaczewski (Ziemowit)** — see item (2). Coordinate with both.

## The op and its files
- Op dir: `ttnn/cpp/ttnn/operations/transformer/sdpa/`
  - `sparse_sdpa_msa.{cpp,hpp}` — op entry (takes `scale`, `block_size`, `cache_batch_idx`,
    `chunk_start_idx`, `cluster_axis`, `compute_kernel_config`)
  - `device/sparse_sdpa_msa_device_operation.cpp` — validation + attrs
  - `device/kernels/compute/sparse_sdpa_msa_compute.cpp` — QK·softmax·PV (flash-style online softmax)
  - `device/kernels/dataflow/sparse_sdpa_msa_reader.cpp` — block gather + diagonal-mask metadata
  - `device/kernels/dataflow/sparse_sdpa_msa_writer.cpp`
- Op unit test (iterate here): `tests/ttnn/unit_tests/operations/sdpa/test_sparse_sdpa_msa.py`
  + `sparse_sdpa_msa_test_utils.py`
- Semantics: q `[1,H,S,d]` bf16 (or fp8_e4m3) ROW_MAJOR; k/v `[1,n_kv,T,d]` bf16/bf8 TILE; block-ids
  `[1,n_kv,S,TOPK]` uint32 (sentinel `0xFFFFFFFF`, valid blocks a contiguous prefix); `d=128`,
  `block_size=128`. Causality is block-granular via the selected block-ids.

## (1) PRIMARY — numerical precision (bf16 intermediates)

**Finding (measured on the real M3 model, SP=8×TP=4, one-shot 5120-token prefill, per-layer KV-cache PCC
vs an fp32 torch golden):** the attention **intermediate precision** inside `sparse_sdpa_msa` is the
**dominant** accuracy-loss driver, compounding over the 57 stacked MSA layers.

Ablation — replacing ONLY the device `sparse_sdpa_msa` with an identical **host fp32** attention (same
bf16 q/k/v inputs upcast to fp32, same block selection, same causality):

| configuration | min KV-PCC V | min K | min index_k |
|---|---|---|---|
| all-device (bf16 sdpa) | **0.444** | 0.719 | 0.777 |
| host fp32 sdpa (target) | **0.867** | 0.959 | 0.973 |

The recovery is **+0.42 on V**, growing monotonically with depth (≈0 at layer 3 → +0.4 by layer 54) —
the signature of a per-layer precision error accumulating in the residual stream.

**Already ruled out (do NOT re-try as the fix):**
- `compute_kernel_config` with **`fp32_dest_acc_en=True` + `MathFidelity::HiFi4`** → tested on-device,
  **~0 effect** (min V 0.444 → 0.444). It is *legal* for bf16 q (only fp8 q *requires* it, see the
  `TT_FATAL` in the device op), the op accepted it, it ran — but the DEST-register precision and
  math-fidelity are **not** where the loss is.
- Input precision is NOT the cause: host and device both start from the same bf16 q/k/v.

**Therefore the loss is in the compute kernel's fp32-vs-bf16 handling of the flash-attention
INTERMEDIATES** — the QK-scores and softmax-probability circular buffers and/or the online-softmax
running max/sum accumulation, which stay in bf16 regardless of the DEST/fidelity flags. **The fix is a
compute-kernel change** to carry those intermediates (and the softmax reduction) in fp32.

**Goal:** on-device `sparse_sdpa_msa` should match the host-fp32 accuracy (per-layer KV-PCC V ≥ ~0.87
matching the table above), ideally gated so it can be enabled via `compute_kernel_config`
(`fp32_dest_acc_en`) or made the default for bf16 q if the perf cost is acceptable. Measure the perf
impact (this op is on the prefill hot path).

## (2) Diagonal-block token-level causal mask — PR #48700 (Ziemowit)

The op's causality is block-granular only; the indexer force-selects the current (diagonal) block, so a
query attends the **future tokens inside its own block** unless a token-level mask is applied on the
diagonal block. **PR #48700** adds exactly this: new `chunk_start_idx` (global position of query row 0)
+ `cluster_axis` (per-device SP offset) args → token-level causal mask on the diagonal block. Measured
worth **+0.086 on min V** on the same harness. Constraint already encoded: the mask **requires bf16 q**
(fp8 q with `chunk_start_idx` is rejected). Land / finish / coordinate this — it is complementary to (1)
and lives in the same compute + reader kernels, so do them together.

## (3) Also on this op's backlog (lower priority, mention only)
- `matmul_init` → `mm_init` fp8-determinism fix (PR #48401 applied it to the DSA `sparse_sdpa`, NOT to
  `sparse_sdpa_msa` — `sparse_sdpa_msa_compute.cpp` still uses `matmul_init`).
- slab-aware NdShard cache-read (for the chunked/long-context read path) — separate concern from
  accuracy; skip unless asked.

## How to verify
- **Iterate** on the op unit test: `tests/ttnn/unit_tests/operations/sdpa/test_sparse_sdpa_msa.py`
  (add/extend an fp32-reference-vs-op precision case if one isn't there; #48700 added causal cases).
- **End-to-end validation** (owner's team can run; needs the M3 branch `vmelnykov/minimax_m2_prefill`):
  `models/demos/minimax_m3/tests/galaxy_prefill_kv_pcc.py` with
  `PREFILL_TRACE_DIR=/data/philei/models/minimax-m3-prefill-cache/golden/longbook_5120 PREFILL_CHUNKED=0
  EXPERT_DTYPE=bf8` on a Blackhole galaxy. Reports per-layer K/V/index_k PCC. Success = min V climbs
  from ~0.44 toward ~0.87 with the precision fix (before any MoE work).
- Build after C++ changes: `./build_metal.sh -b Release`. A clean build may be needed if an incremental
  one doesn't redeploy the loaded `.so`.

## Deliverables
1. Compute-kernel precision fix so on-device `sparse_sdpa_msa` reaches host-fp32 KV-PCC (min V ≥ ~0.87),
   with a perf number for the prefill path.
2. Land/finish the diagonal-block causal mask (#48700), token-level causal on the diagonal block, bf16-q.
3. (If cheap) the `matmul_init`→`mm_init` determinism fix.
