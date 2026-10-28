# Sparse ring attention: refactor plan (2026-10-28)

A separate op for block-sparse ring attention, with Sol-Attn, VSA and the frame-block sparse mode of
`sosborne/sparse-ring-sdpa` (and later variants) as compile-time configurations of one kernel, sharing components
with the dense ring op instead of copying them. Anything specific to one variant is a parameter or a compile-time
argument; the shared path never branches on a variant at run time.

## 0. Status (2026-10-28, branch `sosborne/general-sparse-ring-attn`, uncommitted)

- R1 started: `ttnn.transformer.sparse_ring_sdpa` (host API, device op, factory, kernels) is a copy of the
  Sol-containing `ring_joint_sdpa`, called with the resident mode fixed. `ring_joint_sdpa`, `compute_streaming.hpp`,
  the all-gather and `kernel.cpp` are back to main. Model and tests call the new op; old mode tests removed;
  `test_sparse_ring_sdpa_minimax_h3.py` compares the op with the dense ring op and a torch reference.
- Next: strip the dense / joint / KV-cache paths and the gather / chunk / compact / pack Sol modes from the new
  op, then the traits (R2).

## 1. Goals and constraints

- **Mergeable to main.** The dense `ring_joint_sdpa` is already large and serves the common case; reviewers will not
  take more sparse code in it. The sparse paths move to their own op, and `ring_joint_sdpa` loses its Sol code
  (resident, chunk, compact, pack, gather modes), so the dense op gets smaller too.
- **No duplication.** Shared pieces are factored into headers and host helpers used by both ops. Variant
  differences are compile-time policies inside one kernel set, not forked kernels.
- **Keep every measured optimization** (Sol resident path as of `sosborne/sol-attn-cleanup` and the D4 work on
  `sosborne/sol-attn-d4-pairs`): resident Q with in-place slot state, block-level masking, M10 exp / P@V interleave,
  D1 group multicast, the full piece union, the one-page plan without phase-alignment dummies, 256-key pieces at
  5 s, and D4 pair steps if `d4p` confirms them. The Sol variant starts as a port of that path, not a rewrite.
- **Multi-pass Q** (10 s / 15 s: the resident rows do not fit L1) is a first-class part of the design (section 5).

## 2. What the variants need

| | Sol-Attn | VSA | Frame-block (`sosborne/sparse-ring-sdpa`) | shared |
|---|---|---|---|---|
| Block | 64 tokens, contiguous in sequence order | 64 tokens, one (4,4,4) cube after a static reorder; ragged boundary cubes | one video frame (`tokens_per_frame`, tile-aligned) | blocks of whole tiles |
| Selection input | per (head, Q block): kept-block bits per shard (block-form mask) | per (head, Q block): padded list of block ids with a sentinel tail (`indices [1,H,S/64,W]`), `block_counts` per block, optional `dense_row_mask`, `stream_order` | one static frame x frame allow table, bit-packed (<= 32 frames, <= 32 words), same for all heads, host-side runtime args | per-row block sets |
| Selection computed by | the pre-pass (pool, moments, threshold; ~0.58 ms at 5 s) | the model's coarse stage (pool, top-k) | the caller (fixed pattern) | outside the op |
| Skipped blocks | pooled proxy route inside the same softmax: one pseudo-token per block, score q*k_mean, value v_sum, mass 64 (n for a tail) in the denominator only; on device as pooled K/V entries plus a generated column bias | nothing in the op; the gated coarse branch is added by the model | dropped (reader and compute skip disallowed frame pairs) | - |
| Normalizer | whole sequence (exact + proxy) | selected blocks only | allowed frames only | online softmax, lazy rescale |
| Ragged blocks | no (tails are padding, never kept) | yes: columns past `block_counts[j]` masked | mask hook |
| Forced blocks | block 0 / near-diagonal / sinks (in the mask) | dense rows (`dense_row_mask`) | - |
| Rows per core | ~9-10 at 5 s; passes at 10 s / 15 s | `rmax` 18 (ring); 2-3 passes per head | resident Q, passes |
| Delivery | D1: one producer per head group multicasts each piece (push), credits | one leader per head streams blocks into slots and multicasts a log; workers pull | one streaming core per head group |
| Visit batching | M9 one step per piece; D4 two pieces per step | windows of `stream_depth/2` arrivals, phase-major over all visits of a window | batch visits per phase |
| Rescale | lazy (M7), two PACK_DONE round trips per step | lazy anchor + threshold, RISC compares, mailbox | lazy rescale |
| Row sums | P x e0 matmul into the slot (bf16) | partial sums per visit, 64-bit fixed point on the writer RISC | - |

"VDN-attention" has no code in the repo; section 4 keeps the policy points general enough for a third variant
(a different selection input, a different skipped-block route), to be confirmed once its definition is known.

## 3. Architecture

### 3.1 Op and API

`ttnn.experimental.sparse_ring_sdpa(q, k, v, selection, *, variant_config, program_config, ring/gather args)`:

- `selection`: a tagged input. `SolSelection{block_mask, pooled_k, pooled_v, [row bounds]}`,
  `VsaSelection{indices, block_counts, dense_row_mask?, stream_order?}` or
  `FrameSelection{tokens_per_frame, num_frames_padded, frame_mask words}`.
- `variant_config`: `SparseVariant::{Sol, Vsa}` plus per-variant options (VSA ragged masking on/off, Sol pooled
  route on/off for ablation-free A/B, passes override).
- Ring and all-gather arguments as `ring_joint_sdpa` takes them today (fused gather, topology, links, semaphores).
- One device operation, one program factory, one reader / writer / compute kernel set. The factory selects the
  variant and passes it as named compile-time args, so each variant compiles only its own code.

### 3.2 Kernels: one skeleton, variant policies

| component | role | shared with dense | variant points |
|---|---|---|---|
| ring K/V receiver | arrival waits on the fused gather | `fused_op_receiver.hpp`, `ring_fusion` host side | none |
| plan builder (reader) | per pass and ring iteration: which pieces each core needs, per-row block sets, the one-page plan | - | **SelectionFormat**: block bits (Sol) or index lists + counts (VSA) |
| delivery (reader / writer) | one streaming core per head group feeds the group | - | **Delivery**: push multicast with credits (D1) or slot ring + log + pull (VSA); see 3.3 |
| step engine (compute) | batched visits: Q@K^T into a compact scratch, row max, lazy rescale, exp, P@V, row sums, in-place slot state | compute primitives in `compute_common.hpp` / `compute_streaming.hpp` (no-MOP matmul, bcast subtract, pack-thread exp, `salad_correct_fused`, `normalize_row_streaming`) | **MaskHook**: none (Sol) or ragged column mask (VSA); **RowSum**: P x e0 (Sol) or fixed-point service (VSA) |
| proxy route (reader + compute) | pooled K/V entries with the generated column bias | `sol_bias_gen.hpp` | **SkippedRoute**: pooled proxy (Sol) or none (VSA) |
| Q residency and passes | rows per pass, slot CBs, output write per pass | - | rows per pass from L1 (both) |
| writer | outputs by row index per pass | `dataflow_common.hpp` | VSA row-sum service |

The policies are small traits structs selected by the factory (`if constexpr` on named compile-time args), e.g.

```cpp
template <SparseVariant V> struct SparseTraits;
template <> struct SparseTraits<SparseVariant::Sol> {
    static constexpr bool kProxyRoute = true, kRaggedBlocks = false;
    using Selection = BlockBitsSelection;   // plan words from the block-form mask
    using Delivery = GroupMulticast;        // D1
};
template <> struct SparseTraits<SparseVariant::Vsa> {
    static constexpr bool kProxyRoute = false, kRaggedBlocks = true;
    using Selection = IndexListSelection;   // indices + block_counts
    using Delivery = GroupMulticast;        // or SlotLogPull, see 3.3
};
template <> struct SparseTraits<SparseVariant::FrameBlock> {
    static constexpr bool kProxyRoute = false, kRaggedBlocks = false;
    using Selection = FrameTableSelection;  // static bit table in runtime args, block = frame
    using Delivery = GroupMulticast;
};
```

### 3.3 One delivery protocol, not two

Both variants stream each block once per pass per head group. D1 (push, multicast) and VSA's leader log + pull
solve the same problem; VSA's pull lets the leader run ahead of slow workers, D1's push avoids the workers' reads.
Plan: start the shared op with D1 (it carries the Sol measurements), keep the interface (`Delivery` policy) so
VSA's pull can be ported as the second implementation, measure both on both variants, and keep one. Two delivery
paths in the merged op is the duplication to avoid; this is the decision point.

### 3.4 What moves where

- **New files** (`ttnn/cpp/ttnn/operations/experimental/sparse_ring_sdpa/`): device operation, program factory
  (L1 plan, core groups, delivery rectangles, passes), nanobind, kernels (`sparse_ring_reader.cpp`,
  `sparse_ring_writer.cpp`, `sparse_ring_compute.cpp`), variant headers (`selection_*.hpp`, `delivery_*.hpp`,
  `sparse_step.hpp`).
- **Moved out of `compute_streaming.hpp`**: the resident section (step, run groups, flushes, `sdpa_ring_resident`,
  ~1,000 lines) into `sparse_step.hpp`. The primitives it calls stay in the shared headers.
- **Removed from `ring_joint_sdpa`**: every `sol_*` input, named arg, CB and kernel branch (reader resident and
  block-gather branches, `sol_bias_gen` use, the Sol chain variants, the factory's Sol L1 plan). Dense behaviour
  and binaries unchanged; checked by the existing dense ring tests.
- **Model side**: `attention_minimax_h3.py` calls the new op for Sol (and VSA on its branch); the pre-pass stays.

## 4. Variant policy points

1. **SelectionFormat** - builds per pass and ring iteration: the pieces to stream (union over the group), each
   row's kept blocks per piece, forced rows. Sol: block bits from the mask rows (as the one-page plan does today).
   VSA: index lists merged per row into the same per-row bitmaps (VSA already builds bitmaps per pass).
2. **SkippedRoute** - Sol's pooled entries after the local pieces on the first active iteration, with the column
   bias; none for VSA. A third variant could plug a different correction here.
3. **MaskHook** - per (row, block) column limit: VSA's `block_counts` tails; Sol none.
4. **RowSum** - Sol's P x e0 into the slot; VSA's fixed-point writer service (needed at VSA's list lengths, 858
   blocks per row; Sol rows keep ~120 blocks at tau 1). Shared interface, two implementations, until one is shown
   to cover both.
5. **Delivery** - section 3.3.

## 5. Multi-pass Q

A pass is a set of resident rows per core that fits L1; each pass streams the group's K/V once. The plan builder,
the slot CBs and the writer all iterate passes; the ring all-gather runs once (pass 1 consumes shards as they
arrive, later passes read the gathered K/V from DRAM, no arrival waits). VSA already works this way (2-3 passes per
head), and the 10 s / 15 s Sol fix on the other branch does the same in the current op.

Integrating that fix later is straightforward if the new op has passes from the start: its logic maps onto the
pass loop (row assignment per pass, re-streaming, output per pass) instead of being patched into the current op's
single-pass resident path. I need the branch name to read it before the design is fixed.

## 6. Tests

- **Equivalence**: the new op's Sol variant against the current `ring_joint_sdpa` resident mode on the existing
  geometries (512 / 1024 / 1184 rows, 5 s parity at 256 and 384 keys, keep-all = dense equivalence), before the
  Sol code is removed from `ring_joint_sdpa`.
- **Per variant**: Sol vs the torch reference (`sol_ref`), VSA vs its existing tests (`test_vsa_*`), ragged tails,
  dense rows, empty rows, multi-pass shapes (10 s / 15 s row counts on 4x8).
- **Dense regression**: the ring joint dense tests unchanged after the Sol removal.
- **Unit tests in `tests/ttnn`** for the op (small shapes, 2x4 / 4x8) so CI covers it without the model.

## 7. Milestones

| step | what | effort (days) | check |
|---|---|---|---|
| R0 | design review with the VSA owner (delivery, row sums, selection format); read the multi-pass branch | 1-2 | agreed interfaces |
| R1 | new op skeleton: device op, factory, nanobind, kernels as a straight port of the Sol resident path (cleanup branch), passes loop with one pass | 4-6 | equivalence with `ring_joint_sdpa` resident, same traced time |
| R2 | variant traits; Sol through the traits; step engine in `sparse_step.hpp` | 3-4 | equivalence, binary size, same time |
| R3 | multi-pass (port of the 10 s / 15 s fix) | 2-4 | 10 s / 15 s correctness and time against the other branch |
| R4 | VSA variant: index-list selection, ragged mask, row-sum service; delivery A/B | 5-10 (with the VSA owner) | VSA tests, VSA traced time vs `vsa_ring_sdpa` |
| R4b | frame-block variant (port of `sosborne/sparse-ring-sdpa`: frame table selection, no proxy route) | 2-3 | its `test_ring_joint_sparse_frames.py` cases against the new op |
| R5 | remove Sol from `ring_joint_sdpa`; model switch; unit tests; PR | 3-4 | dense regression, model e2e |
| R6 | port D4 pairs (if `d4p` confirms) and continue optimizing in the new op | - | - |

Total ~3-5 weeks before the PR, most of it R1, R2 and R4.

## 8. Open questions

1. VDN-attention: its definition (selection input, skipped-block handling), to check the policy points.
2. VSA: is its owner on board with one shared op, and which delivery and row-sum implementations to keep.
3. Multi-pass Q: the branch with the 10 s / 15 s fix.
4. Name and location: `ttnn.experimental.sparse_ring_sdpa` under `experimental/` (lower review bar,
   can move later) or `transformer/sdpa/`.
5. Whether the dense joint (text) tensors are needed for the sparse op (MiniMax-H3 Sol runs without them).
