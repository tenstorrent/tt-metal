# P1: skip the indexer top-k TP regather for thin-head-shard models

**Change** (this commit): `TtIndexer(skip_tp_regather=...)`, injected from ttMLA as
`self._needs_head_to_seq_reshard`. When it fires, `TtIndexer.forward` returns the
TP-seq-sharded `[1,1,S/(sp·tp),k]` top-k indices directly instead of
all-gather → `[1,1,S/sp,k]` → (consumer) mesh_partition back. Deletes per full DSA layer
per prefill chunk: one `high_bw_all_gather` over TP, one `mesh_partition`, one RM→TILE
tilize, one TILE→RM untilize, and the 5.2 MB `indexer_topk_indices` DRAM scratch
(allocation now skipped). Default `False` → all existing paths byte-identical.

**Who it fires for** (`tp>1 and (H/tp < 32 or H/tp % 32)`):
- GLM-5.1 / GLM-5.2: 64 heads, tp=4 → 16 → **fires** (21 full layers × 13 chunks on 5.2)
- DeepSeek-V4: 64 heads → **fires**
- DeepSeek-V3.2: 128 heads → 32 → does not fire (its sparse_sdpa genuinely needs the gather)
- tp=1 (single-chip/module tests): does not fire; gather was a no-op anyway

**Why it is safe** (static, verified):
- `_sparse_mla` is the sole consumer (mla.py `_attention` → `_sparse_mla`); its guard
  `transpose_head_to_seq and idx.shape[2] != q_rm.shape[2]` skips the TP partition when the
  rows already match the resharded q (`S/(sp·tp)`), and the `sp>1` full-glob guard doesn't
  trigger. GLM-5.2 shared-layer reuse injects the same shape → same guards.
- Ownership: the returned tensor is the topk op's own output (not the shared gather scratch),
  so the transformer's reuse-chain `ttnn.deallocate` calls stay safe.
- Ordering: relies on `mesh_partition ∘ high_bw_all_gather = identity` per chip on the TP
  axis — the same assumption today's code already depends on (a2a'd q rows vs partitioned idx).

**Silicon validation needed (8×4 mesh; cannot run on this single-chip box):**
1. `blaze-models-prefill-tests` GLM-5.2 DSA/sparse-MLA suites at mesh 8×4 (PCC + shapes).
2. GLM-5.2 chunked no-pcc perf (`test_glm_prefill_transformer_chunked_no_pcc`, GH #51331
   baseline ~2.6 s/chunk): expect a small per-chunk improvement (~1 ms/layer-order CCL cost
   per GH #47803 × 21 layers → tens of ms/chunk ceiling); regression = ordering assumption broken.
3. Loudbox 2×4 module tests (sp=2, tp=4 → fires there too).

**P2 (RM gather, delete the TILE round-trip) — PARKED, aliasing trap found:** the TILE→RM
`to_layout` doubles as the copy-out of the shared named scratch
(`indexer_topk_indices`). A naive RM gather would return the scratch *wrapper*; the
transformer's reuse chain deallocates held indices (tt_prefill_transformer.py:474-476,
492-493), which would free the model-owned buffer aliased by every full layer → corruption
on later chunks. An RM variant must either gather into a fresh per-call output or clone out
of scratch — at which point the saving over tilize+untilize is marginal. Only worth
revisiting for DS-V3.2 (where the gather remains), with the copy-out kept.
