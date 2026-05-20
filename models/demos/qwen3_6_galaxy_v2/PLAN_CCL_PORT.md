# Plan — port llama70b fused/optimized CCL pattern to qwen3.6 v2

## STATUS (2026-05-20): Phase 1 empirically null — plan invalidated

The premise of this plan — that the 4.4× per-call RS gap (193 µs FA vs 44 µs DN) was caused by qwen3.6 using stock `ttnn.all_reduce` while llama70b uses `tt_ccl.line_all_reduce(use_optimal_ccl_for_llama=True)` — was tested empirically:

| config | tok/s/u | ms/step |
|---|---:|---:|
| Baseline (stock `ttnn.all_reduce` everywhere) | 18.99 | 52.66 |
| `QWEN36_DELTA_LAR=1` (DN out_proj LAR enabled) | 19.00 | 52.63 |
| `QWEN36_FULLATTN_WO_LAR=1` (FA WO LAR added in this session) | 19.00 | 52.63 |

**Zero measurable wall-clock win** from swapping to `tt_ccl.line_all_reduce(use_optimal_ccl_for_llama=True)` at either the DN or FA reduction sites. PCC stayed ≥ 0.999 — the math is correct, just not faster.

This invalidates the projected ~12 ms/step recovery in this plan. The 32.6 ms gap to Qwen3-32B's 50 tok/s/u target must come from other buckets (DeltaNet kernel, no prefetcher, lm_head, bf8 attn weights). See `PERF_GAP_VS_QWEN3_32B.md` § "CORRECTION: Bucket 1 empirically falsified" for the corrected attribution.

The FA WO LAR code path remains in the codebase (env-gated, default-off) for future experimentation. Phases 2-4 (`llama_rs_create_heads`, `all_gather_concat`) are not pursued because the underlying premise (CCL implementation is the bottleneck) was wrong.

---

## Goal (original, for record only)

Close the **~12 ms / step** decode wall-clock gap caused by qwen3.6 using stock
`ttnn.all_reduce` where llama3-70B / Qwen3-32B use the fused, persistent-buffer,
`use_optimal_ccl_for_llama=True` variants in `tt_ccl`.

**Projected outcome**: 18.99 → ~24 tok/s/u (52.66 → ~41 ms/step) at V2-DN-TP
HEAD on the BH GLX 8×4 mesh, no PCC regression.

Evidence backing the projection (from
`PERF_GAP_VS_QWEN3_32B.md` § V2-DN-TP per-call-site decode breakdown):

- qwen3.6 FA `ttnn.all_reduce` (cluster_axis=1, post-WQKVG) measured at
  **193 µs/call avg** in the 1L FullAttn tracy
  (`generated/profiler/reports/2026_05_19_21_43_08/`)
- DeltaNet's analogous calls (same cluster_axis, same ring size, but smaller
  output dim) run at **44 µs/call** — the lower bound for what optimized
  ops achieve on this hardware/topology
- Closing the per-call delta on the 480 FA RS calls per 1L capture saves
  68.6 ms aggregate decode device time = 0.72 ms/chip/step/FA-layer
- 16 FA layers × 0.72 ms = **~11.5 ms / step** at 64L assuming similar
  pipelining factor to current (1.79×)

## Topology context (DO NOT CHANGE)

The current branch is at HEAD `e99df7e66f1` (V2-DN-TP, full 2D-TP across all
32 chips). The `dims=` layout matches Qwen3-32B exactly:

- FA WQKVG: `ShardTensor2dMesh(dims=(3, 2))` — heads on rows (8-way), K on cols (4-way)
- FA WO: `ShardTensor2dMesh(dims=(2, 3))` — input on rows, output on cols
- DN qkvz / ba: 2D-TP equivalents
- MLP w1/w3: `dim=(-1, -2) = (3, 2)`
- lm_head: `dims=(3, 2)`

The cluster_axis choices (post-WQKV cluster_axis=1, post-WO cluster_axis=0)
are identical to Qwen3-32B and **must not be changed** — they are correct
for the layout. This plan changes only the CCL OP IMPLEMENTATION, not the
topology.

## Reference patterns

### llama3-70B / Qwen3-32B FA decode (the target pattern)

`models/demos/llama3_70b_galaxy/tt/llama_attention.py:390-582`:

```python
# Step 1: WQKV matmul → fused RS+create_heads
(q_heads, k_heads, v_heads) = self.tt_ccl.llama_rs_create_heads(
    xqkv_fused_sharded,
    cluster_axis=1,
    num_links=self.model_config["GALAXY_NUM_LINKS"],
    dim=3,
    qkv_memory_config=self.model_config["CREATE_HEAD_OUTPUT_MEMCFG"],
    use_optimal_ccl_for_llama=True,
)

# Step 2: ...QK norm, RoPE, KV cache, SDPA...

# Step 3: post-SDPA → fused AG+concat
attn_output_cat = self.tt_ccl.all_gather_concat(
    attn_output_1G4D_sharded,
    dim=1,
    cluster_axis=1,
    num_links=self.model_config["GALAXY_NUM_LINKS"],
    memory_config=self.model_config["SHARDED_ATTN_WO_INPUT_RING_MEMCFG"],
    num_heads=self.n_local_heads,
)

# Step 4: WO matmul → optimized line_all_reduce
dense_out_reduced = self.tt_ccl.line_all_reduce(
    dense_out_ttnn,
    cluster_axis=0,
    num_links=self.model_config["GALAXY_NUM_LINKS"],
    memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
    use_optimal_ccl_for_llama=True,
)
```

### qwen3.6 v2 FA decode (current state, the source pattern to replace)

`models/demos/qwen3_6_galaxy_v2/tt/llama_attention.py:1655-2001`,
`_forward_decode_qwen36`:

```python
# After WQKVG matmul (line ~1717):
xqkvg = ttnn.all_reduce(
    xqkvg_partial,
    cluster_axis=1,
    num_links=1,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
# Then SEPARATE ttnn.slice ×4 for q/g/k/v, _qwen36_qknorm_flat_to_heads,
# _qwen36_flat_to_heads, partial_rope_apply ×2, paged_update_cache ×2

# After WO matmul (line ~1982):
dense_out_full = ttnn.all_reduce(
    dense_partial,
    cluster_axis=0,
    num_links=1,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

The matmul math is the same. The CCL ops are stock library calls without
the fused / optimized / persistent-buffer variants.

## Critical files

- `models/demos/qwen3_6_galaxy_v2/tt/llama_attention.py` —
  `_forward_decode_qwen36` (line 1655-2001) and the WQKVG/WO weight loaders
  (line 435-560)
- `models/demos/qwen3_6_galaxy_v2/tt/qwen36_delta_attention.py` —
  DeltaNet `forward_decode` and the qkvz / ba / out_proj all_reduces
  (lines 649-689, 2154-2214)
- `models/demos/qwen3_6_galaxy_v2/tt/llama_ccl.py` — persistent buffer
  setup (the qwen36 buffer keys; already has the line_all_reduce path
  from V2-13 / V2-CCL but only partially wired to the FA decode site)
- `models/demos/qwen3_6_galaxy_v2/tt/qwen36_model_config.py` — needs new
  memcfg entries: `CREATE_HEAD_OUTPUT_MEMCFG`, `SHARDED_ATTN_WO_INPUT_RING_MEMCFG`,
  `DECODE_RESIDUAL_MEMCFG` analogues, plus `WQKVG_OUTPUT_RING_MEMCFG`
- `models/demos/qwen3_6_galaxy_v2/demo/tracy_perf_1L_fullattn.py` and
  `tracy_perf_1L_delta.py` — re-run for per-call validation
- `models/demos/qwen3_6_galaxy_v2/tests/test_decode_perf_intrace.py` —
  64L wall-clock validation gate
- `models/demos/qwen3_6_galaxy_v2/tests/test_4layer_hybrid_pcc.py` — PCC gate

## Phased migration

### Phase 1 — V2-CCL-P1: line_all_reduce for FA WO

**Smallest, safest first step.** Replace `ttnn.all_reduce(cluster_axis=0)`
at `_forward_decode_qwen36` line ~1982 with the persistent-buffer
`tt_ccl.line_all_reduce(..., use_optimal_ccl_for_llama=True)`.

This is the WO row-axis 8-way reduction. The persistent buffer already
exists for the BF16 size in `llama_ccl.py:518-538` (added in V2-CCL-followup
when the `QWEN36_FULLATTN_WO_SHARDED` flag was explored). V2-CCL-followup
left this code in place but default-off because the SHARDED variant was
perf-neutral. **The line_all_reduce part itself is the change worth keeping.**

**Files**: `llama_attention.py:1980-1988`, `llama_ccl.py:518-538` (verify
buffer keys match the V2-DN-TP per-chip output shapes).

**Test gates**:
1. `test_layer3_fullattn_block_pcc.py` (1L full-attn PCC ≥ 0.99)
2. `test_4layer_hybrid_pcc.py` (4L hybrid PCC ≥ 0.99)
3. `test_64layer_full_pcc.py` (64L PCC; the headline correctness gate)
4. `tracy_perf_1L_fullattn.py` — verify post-WO RS µs drops from ~193 → ~50

**Projected gain**: ~3-4 ms / step (FA-WO RS is one of the two 193 µs/call
sites). Modest but de-risks the larger Phase 2 work.

### Phase 2 — V2-CCL-P2: llama_rs_create_heads for FA WQKVG → Q/K/V split

This is the **biggest single win**. Replaces the qwen3.6 sequence:
1. `ttnn.all_reduce(cluster_axis=1)` after WQKVG matmul (the slow 193 µs RS)
2. 4× `ttnn.slice` to extract q / gate / k / v from the concatenated output
3. `_qwen36_qknorm_flat_to_heads` for q, k (reshape + QK norm)
4. `_qwen36_flat_to_heads` for v

with a single fused `tt_ccl.llama_rs_create_heads(cluster_axis=1, ...)`
plus minor adaptations for the gate output (which llama70b doesn't have).

**Complication: qwen3.6 has 4 outputs (Q, gate, K, V), not 3 (Q, K, V).**
The `llama_rs_create_heads` op was designed for 3-output WQKV. Two options:

- **Option A**: Extend `llama_rs_create_heads` to accept a 4-output split
  spec, with the gate handled as a separate scalar tensor pulled off the
  same fused output buffer. Requires C++ kernel work.
- **Option B**: Run `llama_rs_create_heads` on a wqkvg restructured to put
  the gate at the end and slice it post-fused. Keeps the C++ unchanged but
  loses the post-RS slice fusion for gate. Estimated win ~2/3 of the full
  port.

Recommend **Option B first** as a low-risk validation: get the RS savings
on the Q/K/V part, then optionally chase the gate fusion in a follow-up.

**Files**: `llama_attention.py:1717-1745` (matmul + post-RS slices),
`llama_ccl.py` (add `llama_rs_create_heads` call site + qwen36 memcfg
plumbing), `qwen36_model_config.py` (add `CREATE_HEAD_OUTPUT_MEMCFG`
analogue sized for n_q_per_chip=3, hd=256, B=1, T=1).

**Test gates**: same 4 as Phase 1, plus:
5. `test_layer3_fullattn_block_pcc.py` should now also verify the gate
   path produces matching outputs vs the pre-fused path (record a parity
   golden tensor before the refactor).

**Projected gain**: ~6-8 ms / step (the bulk of the FA RS savings).

### Phase 3 — V2-CCL-P3: all_gather_concat for post-SDPA path

Replaces qwen3.6's separate `ttnn.permute(q_rot, (2, 0, 1, 3))` +
`ttnn.permute(attn_out_1bnd, (1, 2, 0, 3))` + `_qwen36_heads_to_flat`
sequence with a fused `tt_ccl.all_gather_concat`.

The post-SDPA path in qwen3.6 has more transformations (gate multiplied
after attention, before WO) so this fusion is partial — but it removes
several `permute` / reshape ops from the trace.

**Files**: `llama_attention.py:1810-1944` (post-SDPA path through gate
multiplication and into WO input prep).

**Test gates**: same 4, plus 1L tracy to verify the post-SDPA AGA µs
drops.

**Projected gain**: ~1-2 ms / step (smaller than Phase 2 but cleans up
~10 ops/layer from the critical path).

### Phase 4 — V2-CCL-P4: DeltaNet stock-`all_reduce` → optimized variants

Apply the same pattern to qwen36_delta_attention.py:

- `qkvz = ttnn.all_reduce(qkvz_partial, cluster_axis=1, ...)` (line 651)
  → `tt_ccl.line_all_reduce(use_optimal_ccl_for_llama=True)` or a
  DN-specific fused-RS variant
- `ba = ttnn.all_reduce(ba_partial, cluster_axis=1, ...)` (line 689)
  → same treatment
- Out_proj `all_reduce(cluster_axis=0)` (line ~2179) — already partially
  optimized via V2-CCL `QWEN36_DELTA_LAR`; verify it's using the optimal
  path at V2-DN-TP HEAD (commit `e99df7e66f1`)

DN already runs at 44 µs/call for some of its RS, so the Phase 4 win is
smaller than Phase 2 (DN code already has more of the optimization wired
in). Estimate **~1-2 ms / step** of remaining DN-CCL win.

**Files**: `qwen36_delta_attention.py:651, 689, 2179`, `llama_ccl.py`
(any missing buffer keys).

## Verification rules (all phases)

- **PCC > 0.99** mandatory at every phase before committing. The
  4L-hybrid PCC test is the fastest gate (~2 min); the 64L PCC test is
  the headline correctness gate (~5 min).
- **Tracy per-op verification at each phase**: confirm the targeted RS
  call drops to ≤60 µs/call. If it doesn't drop, the persistent-buffer
  hookup is wrong or the use_optimal_ccl_for_llama=True path didn't
  take effect.
- **Wall-clock perf test** (`test_decode_perf_intrace.py` 32-step decode
  loop) at end of each phase. The headline metric is tok/s/u. Baseline
  18.99; phase-by-phase projection:
  - Phase 1: 18.99 → ~20.5
  - Phase 2: 20.5 → ~23
  - Phase 3: 23 → ~24
  - Phase 4: 24 → ~25

  Anything more than 0.5 tok/s/u below projection at any phase is a
  signal that the optimization didn't land; investigate before
  proceeding.
- **No `tt-smi -r`-induced state** between phase tests — the L1 / CB
  allocation patterns change across this refactor; reset cleanly
  between iterations.

## Risks

- **Option B (Phase 2 without gate fusion)** may not produce the full
  projected gain. If after Phase 2 the FA RS is still > 80 µs/call,
  the bottleneck has shifted to the gate handling or some other
  unfused op; reassess before Phase 3.
- **PCC drift on bf16 attention weights**: V2-7b documented that
  composed PCC was sensitive to fp8 quantization. The line_all_reduce
  variant uses a different reduction order than stock `ttnn.all_reduce`;
  it could produce slightly different bf16 outputs. The 4L hybrid PCC
  gate (currently ≥ 0.9997 hidden / 0.9997 logits) catches this.
- **Persistent-buffer L1 footprint**: line_all_reduce reserves L1
  buffers at construction time. Adding new persistent buffers raises
  per-chip L1 usage. The `worker_l1_size` may need adjustment if total
  reserved L1 exceeds budget; monitor `mesh_device` L1 alloc messages
  on first construction.

## Rollback strategy

Each phase is a single commit on the V2-DN-TP HEAD branch. Roll back any
phase by reverting its commit. The phases are independent (each can be
shipped without the others) so a Phase-N regression doesn't block
Phase-N-1.

If the persistent-buffer infrastructure proves incompatible with
qwen3.6's shapes:

- Fall back to `tt_ccl.line_all_reduce(use_optimal_ccl_for_llama=False)`
  for any failing site — still better than stock `ttnn.all_reduce` by
  ~30-50%.

## Out of scope

- Re-enabling the prefetcher (V2-grid task #47) — separate plan,
  independent ~3-5 ms / step win on top of the CCL work
- bf8 attention weights (V2-7b bf16-forcing) — separate per-layer
  probe needed; potential ~1 ms / step
- LM head sampling fusion — separate; ~1-2 ms / step
- V2-17 / V2-18 DeltaNet kernel — device-blocked, separate track

## Order-of-operations recommendation

Run phases in order 1 → 2 → 3 → 4. Phase 1 de-risks the persistent-buffer
infrastructure on the smallest-impact site. If Phase 1 lands clean, the
larger Phases 2-3-4 follow with high confidence.

After all four phases, projected qwen3.6 perf: **~25 tok/s/user
(40 ms/step)** vs current 18.99 tok/s/u (52.66 ms/step) — closing about
40% of the gap to Qwen3-32B's 50 tok/s/u target. The remaining ~10 ms/step
gap would then be addressed by the prefetcher re-enable (task #47) and
the V2-17/V2-18 DeltaNet kernel work.
