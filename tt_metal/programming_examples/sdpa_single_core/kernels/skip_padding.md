# Feasibility Study: Skipping Padded K/V Tile Computation in SDPA

## Problem Statement

When the K sequence length doesn't divide evenly into Sk_chunk_t-sized chunks, the last chunk is zero-padded. Currently, all computation proceeds on the full chunk — matmuls on zero K tiles, masking with -inf, exp(-inf)=0, multiplying 0×V — all producing results that contribute nothing to the output. This is wasted compute.

**Goal:** Skip unnecessary computation on padded tiles in the last K chunk.

## Key Parameters and Terminology

```
Sk_chunk_t     = tiles per K chunk (e.g., 16)
padded_k_tiles = number of zero-padded tiles in the last K chunk (e.g., 6)
effective_Sk   = Sk_chunk_t - padded_k_tiles (e.g., 10)
sbh            = subblock height (1 or 2)
qkt_subblock_w = 8 / sbh (8 when sbh=1, 4 when sbh=2)
kt_num_subblocks = Sk_chunk_t / qkt_subblock_w (e.g., 2 for Sk=16/sbw=8)
```

## Current Flow on the Last K Chunk (with padding)

### Phase 1: Q@KT + Softmax
For each q_subblock (0..Sq_chunk_t/sbh - 1):
1. **Q@KT matmul** — `kt_num_subblocks` calls to `blocked_matmul_and_pack`. Each computes `[sbh × head_dim_t] @ [head_dim_t × qkt_subblock_w]`. Padded K columns produce 0.
2. **apply_padded_mask** — L1-accumulates -inf onto the `padded_k_tiles` rightmost positions in cb_qkt_im (overwriting the 0s from step 1).
3. **sub_exp** (for prev row) — subtracts max, applies exp in-place. On padded positions: exp(-inf) = 0. Reduces to row sum (padded tiles add 0).
4. **reduce_block_max_row<Sk_chunk_t>** — max over all columns. Padded columns are -inf, don't affect the max.

### Phase 2: QKT@V + SALAD
1. **Drain loop (q_sub=0, sbh=1):** `kt_num_subblocks` iterations, each doing sub_exp drain + split V matmul. The padded sub-iterations multiply 0 × V = 0.
2. **Full V matmul (q_sub=1..N-1):** `blocked_matmul_and_pack` with INNER_DIM=Sk_chunk_t. The `padded_k_tiles` inner iterations multiply 0 × V = 0.
3. **SALAD corrections** — operate on max/sum/out, unaffected by Sk dimension.
4. **Normalization** — operates on sum/out, unaffected by Sk dimension.

### Why All Padded Computation is Mathematically Redundant
- Q × 0_K = 0, then masked to -inf → exp(-inf) = 0
- max(valid_values, -inf) = max(valid_values)
- sum(valid_exp_values, 0) = sum(valid_exp_values)
- 0 × V = 0, contributes nothing to output

## Concrete Example: pad=6, Sk=16 (test case `3q_5k-random-sk16-pad6`)

**Config:** Sq_chunk_t=7, 3 Q-chunks, 5 K-chunks, Sk_chunk_t=16, padded_k_tiles=6, sbh=1
- effective_Sk = 10
- qkt_subblock_w = 8, kt_num_subblocks = 2
- Subblock layout in last K chunk:
  - kt_sub=0 (cols 0-7): all 8 valid
  - kt_sub=1 (cols 8-15): 2 valid (cols 8-9) + 6 padded (cols 10-15)

**Baseline PCC verified:** 0.999815 (test `3q_5k-random-sk16-pad6` in `generate_and_test_sdpa.py`)

## Approach A: Skip Fully-Padded kt_subblocks

### Concept
Only skip subblocks where ALL tiles are padded. Granularity = qkt_subblock_w tiles.

### Applicability
- `skippable_subblocks = padded_k_tiles / qkt_subblock_w`
- Only applies when `padded_k_tiles >= qkt_subblock_w`

### Critical Limitation: Fails for pad=6/Sk=16
- padded_k_tiles=6 < qkt_subblock_w=8 → `skippable_subblocks = 0`
- kt_sub=1 has 2 valid tiles, so it CANNOT be skipped
- **Approach A provides ZERO savings for this case**

### General Coverage
| Sk | sbh | qkt_subblock_w | Padding that enables skipping |
|----|-----|----------------|-------------------------------|
| 16 | 1   | 8              | pad ≥ 8 only                  |
| 8  | 1   | 8              | Never (pad < Sk required)     |
| 16 | 2   | 4              | pad ≥ 4                       |
| 8  | 2   | 4              | pad ≥ 4                       |
| 4  | 2   | 4              | Never (pad < Sk required)     |

For sbh=1 with Sk=16, Approach A only helps when half or more of the chunk is padded.

### max_reduce Correctness Problem
`reduce_block_max_row<Sk_chunk_t>` is a compile-time template — can't narrow at runtime. If we skip the matmul for padded subblocks and write 0 there instead of -inf, `max(valid_values, 0)` is wrong when all valid values are negative. Writing -inf requires sub_exp to convert -inf→0 before V matmul, which defeats the savings.

**Verdict: Approach A is not worth pursuing.** Too coarse-grained, doesn't cover the common case (non-subblock-aligned padding), and has the max_reduce correctness issue.

## Approach B: Dual-Path Inner Loop with `effective_Sk`

### Concept
On the last K chunk, dispatch to a second `sdpa_inner_loop_step` instantiation parameterized by `effective_Sk = Sk_chunk_t - padded_k_tiles`. All LLK calls (matmul, sub_exp, reduce, etc.) use `effective_Sk` dimensions. No padding mask needed at all.

### Dispatch Pattern
```cpp
// In kernel_main() K-chunk loop:
if (is_last && padded_k_tiles > 0) {
    sdpa_inner_loop_step<..., effective_Sk, ...>(...);  // reduced path
} else {
    sdpa_inner_loop_step<..., Sk_chunk_t, ...>(...);    // standard path
}
```

### What Changes on the Reduced Path

**Phase 1:**
- `cb_reserve_back(cb_qkt_im, Sq_chunk_t * effective_Sk)` instead of `Sq_chunk_t * Sk_chunk_t`
- Matmul subblocking recalculated:
  - `eff_full_subblocks = effective_Sk / qkt_subblock_w`
  - `eff_remainder = effective_Sk % qkt_subblock_w`
  - Full subblocks use standard width, last subblock uses `eff_remainder` width
- `reduce_block_max_row<effective_Sk>` instead of `<Sk_chunk_t>`
- sub_exp processes only `effective_Sk` columns
- `apply_padded_mask` is **eliminated entirely**
- `row_tiles = sbh * effective_Sk` for `cb_push_back_hold_wr_ptr`

**Phase 2:**
- Drain loop: `eff_kt_num_subblocks` iterations (may differ from `kt_num_subblocks`)
- Full V matmul: `INNER_DIM = effective_Sk` (not Sk_chunk_t)
- V matmul indexing: `qktv_in0_block_w = effective_Sk`

**Reader interaction (no reader changes needed):**
- Reader still loads full `Sk_chunk_t` K/V tiles
- Compute pops full amount: `cb_pop_front(cb_kt_in, head_dim_t * Sk_chunk_t)`
- Compute pops full amount: `cb_pop_front(cb_v_in, Sv_chunk_t * head_dim_t)` (where Sv_chunk_t = Sk_chunk_t)
- The extra loaded tiles are simply ignored

### Handling Non-Aligned effective_Sk (the Partial Subblock)

When `effective_Sk % qkt_subblock_w != 0` (e.g., effective_Sk=10, sbw=8 → remainder=2):

The last subblock has width `eff_remainder` instead of `qkt_subblock_w`. This requires:

1. **mm_block_init_short** with ct_dim=`eff_remainder` before the narrow matmul
   (re-init already happens at Q@KT ↔ QKT@V transitions, so the pattern exists)
2. **blocked_matmul_and_pack** instantiation with SUBBLOCK_W=`eff_remainder`
3. **sub_exp_block_bcast_cols** instantiation with SBW=`eff_remainder`
4. DST pressure is reduced (fewer tiles), so no hardware concern

Implementation sketch for Phase 1:
```cpp
constexpr uint32_t eff_Sk = Sk_chunk_t - padded_k_tiles;
constexpr uint32_t eff_full_subs = eff_Sk / qkt_subblock_w;
constexpr uint32_t eff_rem = eff_Sk % qkt_subblock_w;

for (uint32_t kt_sub = 0; kt_sub < eff_full_subs; ++kt_sub) {
    // sub_exp prev row (width = qkt_subblock_w) ...
    // blocked_matmul_and_pack<..., qkt_subblock_w, ...>(...);
}
if constexpr (eff_rem > 0) {
    // sub_exp prev row (width = eff_rem) ...
    // mm_block_init_short(..., eff_rem, ...);
    // blocked_matmul_and_pack<..., eff_rem, ...>(...);
}
```

### Savings Quantification for pad=6/Sk=16

Every operation that scales with Sk sees a **37.5% reduction** (6/16) on the last K chunk:

| Operation (per q_subblock) | Current | Approach B | Savings |
|---------------------------|---------|------------|---------|
| QKT matmul tile-MACs | 64 (2×4×8) | 40 (4×8 + 4×2) | 37.5% |
| apply_padded_mask | 6 tile copies | 0 | 100% |
| sub_exp tiles | 16 | 10 | 37.5% |
| max_reduce columns | 16 | 10 | 37.5% |
| V matmul inner iters | 16 | 10 | 37.5% |

As fraction of total (5 K chunks): `6 / (5 × 16) = 7.5%` overall compute savings.

### General Savings Formula
```
per_chunk_savings = padded_k_tiles / Sk_chunk_t
overall_savings   = padded_k_tiles / (num_k_chunks × Sk_chunk_t)
```

Impact scales inversely with num_k_chunks. Extreme cases:
- pad=15/Sk=16, 2 K chunks → 47% overall savings
- pad=1/Sk=16, 100 K chunks → 0.06% savings (negligible)

### Code Size Impact
- Two instantiations of `sdpa_inner_loop_step` (~400 lines of template code)
- The reduced path only executes once per Q chunk (last K iteration), so I-cache impact is minimal
- Additional template instantiations for narrow subblock (blocked_matmul_and_pack, sub_exp_block_bcast_cols)

### Implementation Considerations

1. **sdpa_inner_loop_step currently takes Sk_chunk_t as a template parameter** — it already parameterizes everything on this value. A second instantiation with effective_Sk naturally generates correct code for all the sub-calculations (subblock counts, row widths, etc.).

2. **The pop counts for cb_kt_in and cb_v_in must still use Sk_chunk_t** (to match the reader). This means the reduced path needs to pop more tiles than it uses. This can be handled by keeping the pop calls outside `sdpa_inner_loop_step`, or by adding a separate template parameter for "pop size" vs "compute size".

3. **cb_qkt_im sizing**: The host allocates cb_qkt_im for `Sq_chunk_t × Sk_chunk_t` tiles. The reduced path only uses `Sq_chunk_t × effective_Sk` tiles — fine, it just uses less of the CB.

4. **Phase 2 V matmul with non-aligned effective_Sk**: In the drain loop (sbh=1), the split matmul uses `matmul_inner = effective_Sk / eff_kt_num_subblocks`. This may not divide evenly. For effective_Sk=10 with 2 sub-iterations: inner=8 + inner=2 works if we handle the two iterations with different INNER_DIM template params. Alternatively, don't split — do one full matmul with INNER_DIM=effective_Sk (no split needed, just overlap the last sub_exp with the full matmul).

5. **sbh>1 drain path**: Already does full matmul (no split). Just needs INNER_DIM=effective_Sk.

## Files Involved

| File | Changes Needed |
|------|---------------|
| `kernels/compute/sdpa.cpp` | Dual-path dispatch in kernel_main; second sdpa_inner_loop_step instantiation with effective_Sk; narrow subblock handling for blocked_matmul_and_pack, sub_exp_block_bcast_cols, reduce_c_row_group |
| `kernels/dataflow/reader.cpp` | None (still loads full chunks) |
| `kernels/dataflow/writer.cpp` | None |
| `sdpa_single_core.cpp` (host) | None (CB sizes unchanged, padded_k_tiles already passed as compile-time arg) |
| `generate_and_test_sdpa.py` | Test case `3q_5k-random-sk16-pad6` already added |

## Test Cases for Validation

| Test ID | Config | Why It Matters |
|---------|--------|---------------|
| `3q_5k-random-sk16-pad6` | pad=6, Sk=16, sbh=1 | Non-aligned padding (eff_rem=2), the motivating case |
| `3q_5k-random-sk16-pad8` | pad=8, Sk=16, sbh=1 | Aligned padding (eff_rem=0), cleanest case |
| `1q_5k-random-sk16-pad4` | pad=4, Sk=16, sbh=1 | Small padding, eff_Sk=12 (rem=4) |
| `1q_5k-random-sk8-pad2` | pad=2, Sk=8, sbh=1 | Sk=8 variant, eff_Sk=6 (rem=6, single subblock) |
| `3q_5k-random-sk4-sbh2-pad1` | pad=1, Sk=4, sbh=2 | sbh=2 variant, eff_Sk=3 (rem=3) |
| `3q_19k-random-sk16-pad8-sq9` | pad=8, Sk=16, sq=9 | Larger workload, aligned padding |

All existing padded test cases must pass with identical PCC (no regression).

## Decision: Approach B Recommended

Approach A is dead for the common case (non-subblock-aligned padding). Approach B provides near-optimal savings for any padding amount with manageable complexity. The implementation is a clean template-level optimization — no algorithmic changes, no reader modifications, no host changes.

Optional future enhancement: Approach C (reader-side skip) eliminates DRAM reads for padded K/V tiles. For pad=8/Sk=16 with head_dim_t=4: saves 64 tile reads (131 KB DRAM bandwidth). Worth pursuing only if DRAM bandwidth is the bottleneck (profile first).
