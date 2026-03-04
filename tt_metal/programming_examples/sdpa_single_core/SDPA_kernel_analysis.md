# SDPA Single-Core Kernel Analysis

Comprehensive analysis of the single-core Scaled Dot-Product Attention (SDPA) implementation at `tt_metal/programming_examples/sdpa_single_core/`. A performance benchmarking playground and numerically correct bidirectional SDPA.

---

## Table of Contents
1. [Algorithm Overview: Online Softmax / Flash Attention](#1-algorithm-overview)
2. [Kernel Structure and Roles](#2-kernel-structure)
3. [Circular Buffer Map](#3-circular-buffer-map)
4. [Data Flow: End-to-End](#4-data-flow)
5. [Compute Kernel Detailed Walkthrough](#5-compute-kernel-walkthrough)
6. [Skip-Padding Optimization](#6-skip-padding)
7. [CB Usage Static Analysis (sbh=1)](#7-cb-usage-static-analysis)
8. [Indexing Correctness Audit](#8-indexing-correctness-audit)
9. [Comparison with Reference SDPA (compute_common.hpp)](#9-comparison-with-reference)
10. [Missing Features for Numerical Correctness](#10-missing-features)
11. [Suggested Next Steps](#11-next-steps)

---

## 1. Algorithm Overview: Online Softmax / Flash Attention <a id="1-algorithm-overview"></a>

The kernel implements the **FlashAttention online softmax** algorithm. Given Q, K, V matrices chunked along the sequence dimension:

```
For each Q chunk (q):
    prev_max = -inf, prev_sum = 0, prev_out = 0
    For each K/V chunk (k):
        S = Q_chunk @ K_chunk^T                          # [Sq, Sk] attention scores
        cur_max = max(prev_max, row_max(S))              # running max per row
        P = exp((S - cur_max) * scale)                   # softmax numerator
        cur_sum = sum(P, dim=-1)                         # partial softmax denominator
        cur_out = P @ V_chunk                            # weighted values

        # SALAD correction (Scale-And-Level-And-Divide):
        correction = exp((prev_max - cur_max) * scale)
        cur_sum += correction * prev_sum                 # rescale old denominator
        cur_out += correction * prev_out                 # rescale old output

        prev_max, prev_sum, prev_out = cur_max, cur_sum, cur_out

    # Final normalization
    output = prev_out / prev_sum                         # divide by softmax denominator
```

Key insight: The scale factor `1/sqrt(d)` is fused into the `exp()` calls rather than being a separate multiplication on the attention scores. This gives "scaling for free" on the performance-critical exp computations.

Reference: [FlashAttention paper (online softmax)](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)

---

## 2. Kernel Structure and Roles <a id="2-kernel-structure"></a>

### 2.1 Reader Kernel (`kernels/dataflow/reader.cpp`)

Runs on RISCV_0 (BRISC) via NOC0. Responsibilities:
- Reads Q tiles from DRAM once per Q-chunk iteration
- Reads K tiles with **on-the-fly transposition** (tile grid transpose, not element transpose): K is stored `[Sk, d]` in DRAM but loaded as `[d, Sk]` into `cb_kt_in` so the compute kernel can do `Q @ K^T` via standard matmul with `transpose=true` on individual tiles
- Reads V tiles directly (no transposition)
- Double-buffered CBs (`2x` sizing for Q, K, V) enable pipelining

### 2.2 Writer Kernel (`kernels/dataflow/writer.cpp`)

Runs on RISCV_1 (NCRISC) via NOC1. Responsibilities:
- Generates the identity scaler tile (all 1.0s) for `reduce_block_max_row` → `cb_identity_scale_in` (c5)
- Generates a column identity tile (1.0 in column 0, zeros elsewhere) → `cb_col_identity` (c8) for matmul_reduce normalization
- If padded_k_tiles > 0, generates a single -inf tile → `cb_mask_in` (c7), stays fronted for reuse
- Streams normalized output tiles from `cb_normalized_out` (c9) to DRAM

**Writer batching:** The writer waits for a full `head_dim_t`-tile batch per `cb_wait_front`/`cb_pop_front` cycle, then issues all `noc_async_write_tile` calls back-to-back before a single `noc_async_write_barrier`. This pipelines NoC transactions: 4 pipelined 2048 B writes complete in ~543 cycles vs 4x427 = 1708 cycles serialized (3.15x speedup). The `cb_normalized_out` CB is double-buffered (`2 * head_dim_t` pages) so compute can fill the next batch while the writer drains the current one.

### 2.3 Compute Kernel (`kernels/compute/sdpa.cpp`)

Runs on the Math+Pack RISC-V cores (TRISC_MATH, TRISC_PACK). This is the main kernel containing all the SDPA logic.

---

## 3. Circular Buffer Map <a id="3-circular-buffer-map"></a>

### Concrete sizes for default config: Sq_chunk_t=7, Sk_chunk_t=16 (or 8), head_dim_t=4, sbh=1

| CB Index | Name | Capacity (tiles) | Role | Producer | Consumer |
|----------|------|-------------------|------|----------|----------|
| c0 | `cb_q_in` | 2 × 7×4 = 56 | Q chunk (double-buffered) | Reader | Compute |
| c1 | `cb_kt_in` | 2 × 4×16 = 128 | K^T chunk (double-buffered) | Reader | Compute |
| c2 | `cb_qkt_im` | 7×16 = 112 | Q@K^T intermediate (raw + softmax'd, in-place) | Compute | Compute |
| c3 | `cb_v_in` | 2 × 16×4 = 128 | V chunk (double-buffered) | Reader | Compute |
| c5 | `cb_identity_scale_in` | 1 | Identity scaler (all 1.0s) | Writer | Compute |
| c7 | `cb_mask_in` | 1 (optional) | -inf tile for padded K masking | Writer | Compute |
| c8 | `cb_col_identity` | 1 | Column identity for matmul_reduce | Writer | Compute |
| c9 | `cb_normalized_out` | 2 × 4 = 8 | Normalized output streaming (double-buffered) | Compute | Writer |
| c10 | `cb_recip_scratch` | 1 | 1/sum scratch | Compute | Compute |
| c25 | `cb_out_A` | 7×4 = 28 | Ping-pong output accumulator A | Compute | Compute |
| c26 | `cb_out_B` | 7×4 = 28 | Ping-pong output accumulator B | Compute | Compute |
| c27 | `cb_max_A` | 7 | Ping-pong max A | Compute | Compute |
| c28 | `cb_max_B` | 7 | Ping-pong max B | Compute | Compute |
| c29 | `cb_sum_A` | 7 | Ping-pong sum A | Compute | Compute |
| c30 | `cb_sum_B` | 7 | Ping-pong sum B | Compute | Compute |
| c31 | `cb_exp_max_diff` | 7 | exp(prev_max - cur_max) | Compute | Compute |

**Total L1 usage (tiles):** 56 + 128 + 112 + 128 + 1 + 1 + 8 + 1 + 28 + 28 + 7 + 7 + 7 + 7 + 7 = **525 tiles** × 2048 bytes = **~1.05 MB** (well within 1.5 MB L1)

**Note:** CB indices c4 and c6 are unused. The row buffers that previously occupied these slots were eliminated by the `cb_push_back_hold_wr_ptr` optimization (see Section 5.2), saving 32 tiles (64 KB) of L1.

---

## 4. Data Flow: End-to-End <a id="4-data-flow"></a>

### Outer Loop Structure
```
for each Q chunk (q = 0..num_q_chunks-1):
    for each K chunk (k = 0..num_k_chunks-1):
        is_first = (k == 0)
        is_last  = (k == num_k_chunks - 1)

        if (is_last && padded_k_tiles > 0):
            call_step_reduced(...)   # reduced path: effective_Sk = Sk - padded_k_tiles
        else:
            call_step(...)           # standard path: full Sk_chunk_t

        # Post-iteration: pop prev buffers (skip on first iter), swap ping-pong aliases
        # On last K: per-row normalize → cb_normalized_out → writer drains to DRAM
    # Q already popped inside sdpa_inner_loop_step after Phase 1 of the last K chunk
```

No `InitPrevBuffers` is needed — the first K-chunk iteration uses `is_first_iter` gating to skip all prev-buffer reads (eltwise_max, SALAD corrections, exp_max_diff). This follows the same pattern as the reference SDPA's `processed_k_chunks > 0` guard.

**Dual-path dispatch:** When the last K chunk has padding, `kernel_main` dispatches a second instantiation of `sdpa_inner_loop_step` with `Sk_chunk_t = effective_Sk`, `Sv_chunk_t = effective_Sk`, `padded_k_tiles = 0`, and `KT_stride = original_Sk_chunk_t`. This skips all compute on padded tiles (see Section 6).

**Early Q pop:** Q tiles are only used during Phase 1 (Q@KT matmul). On the last K chunk, Q is popped immediately after Phase 1 completes (before Phase 2 begins), allowing the reader to start prefetching the next Q chunk while Phase 2 (QKT@V, SALAD, normalization) is still running. On non-last K chunks, Q is kept in the CB for the next iteration's Phase 1.

### Data Flow Diagram (one K-chunk iteration, sbh=1)

```
    Reader                    Compute                           Writer
    ──────                    ───────                           ──────
                              ┌─────────────────────────────┐
    Q ──→ cb_q_in ──→        │  PHASE 1: Q@KT + Softmax    │
                              │                             │
    K ──→ cb_kt_in ──→       │  for each q_subblock:       │
                              │    matmul(Q, KT) → cb_qkt_im│
                              │    max_reduce → cur_max     │
                              │    sub_exp in-place on      │
                              │      cb_qkt_im (prev row)   │
                              │      → row_sum (L1 accum)   │
                              │    cb_push_back_hold_wr_ptr │
                              │                             │
                              │  if last_k: pop Q early     │
                              ├─────────────────────────────┤
                              │  PHASE 2: QKT@V + SALAD     │
                              │                             │
    V ──→ cb_v_in ──→        │  for each q_subblock:       │
                              │    matmul(qkt_im, V) → out  │
                              │    if !first_k:             │
                              │      exp_max_diff           │
                              │      SALAD: sum/out correct │
                              │    if last_k: normalize     │
                              │      → cb_normalized_out ──→│──→ DRAM
                              └─────────────────────────────┘
    (if last_k: start
     fetching next Q)
```

### Ping-Pong Buffer Strategy

The kernel uses three pairs of ping-pong buffers:
- **max**: `cb_max_A` / `cb_max_B` — running row maximums
- **sum**: `cb_sum_A` / `cb_sum_B` — running row sums (partial softmax denominator)
- **out**: `cb_out_A` / `cb_out_B` — running weighted-value accumulator

After each K-chunk, the "current" buffers become "previous" via pointer swaps. The "previous" buffers are popped (freed) after being consumed by the SALAD correction of the next iteration. On the first K-chunk (`is_first_iter`), no prev buffers exist to pop — the cleanup is skipped.

---

## 5. Compute Kernel Detailed Walkthrough <a id="5-compute-kernel-walkthrough"></a>

### 5.1 sdpa_inner_loop_step Template Parameters

```cpp
template <
    bool PROFILING_ENABLED,
    uint32_t Sq_chunk_t,       // Q rows in tiles
    uint32_t Sk_chunk_t,       // K columns in tiles (= effective_Sk on reduced path)
    uint32_t Sv_chunk_t,       // V rows in tiles (= Sk_chunk_t)
    uint32_t head_dim_t,       // head dimension in tiles
    uint32_t cb_q_in, cb_kt_in, cb_v_in, cb_qkt_im,
    uint32_t cb_identity_scale_in, cb_exp_max_diff,
    uint32_t scale_fp32,       // 1/sqrt(d) as bit-cast uint32
    uint32_t subblock_h,       // 1 or 2
    uint32_t cb_col_identity, cb_recip_scratch, cb_normalized_out,
    uint32_t padded_k_tiles,   // 0 on reduced path (masking compiled out)
    uint32_t cb_mask_in,
    uint32_t KT_stride = Sk_chunk_t>  // physical KT/V CB row width (= original Sk on reduced path)
```

**KT_stride** separates the physical KT/V CB layout (reader-side) from the compute dimension. On the standard path, `KT_stride == Sk_chunk_t` (default). On the reduced path, `Sk_chunk_t = effective_Sk` while `KT_stride = original_Sk_chunk_t`. This allows all CB operations (reserve, push_hold, pop) to use KT_stride-aligned amounts, keeping circular buffer pointers aligned across Q iterations, while all compute operations (matmul, sub_exp, reduce) operate on only the `effective_Sk` valid tiles.

### 5.2 First-Iteration Gating (`is_first_iter`)

Instead of pre-filling dummy "previous" buffers (zeros for sum/out, -inf for max), the kernel uses an `is_first_iter` flag to skip all operations that read from prev buffers on the first K-chunk:

1. **Phase 1 — `reduce_c_row_group`**: passes `do_eltwise_max = !is_first_iter`. When false, the max-reduce writes directly into DST (zeroed by `tile_regs_acquire`) without comparing against prev_max.
2. **Phase 2 — exp_max_diff + SALAD**: entirely skipped on first iteration. The V matmul result is the cur_out directly, and the sub_exp row sums are the cur_sum directly — no rescaling correction needed.
3. **Post-iteration cleanup**: prev buffer pops are skipped (they were never filled).

This matches the reference SDPA's `processed_k_chunks > 0` pattern and eliminates the per-Q-chunk `InitPrevBuffers` overhead and the `cb_neginf` tile.

### 5.3 Phase 1: Q@KT with Softmax

**One-time initialization:** At the start of `kernel_main`, `cb_wait_front(cb_identity_scale_in, 1)` is issued once to ensure the scaler tile is ready. This avoids repeated waits inside `reduce_c_row_group`.

**Derived constants:**
- `qkt_subblock_w` = 8/sbh (8 when sbh=1, 4 when sbh=2)
- `q_num_subblocks` = Sq_chunk_t/sbh
- `kt_num_full_subblocks` = Sk_chunk_t/qkt_subblock_w
- `kt_remainder` = Sk_chunk_t % qkt_subblock_w
- `has_partial_subblock` = (kt_remainder > 0)
- `row_tiles` = sbh × KT_stride

On the standard path, `Sk_chunk_t` is always subblock-aligned so `has_partial_subblock = false` and `kt_remainder = 0`. On the reduced path (effective_Sk), the last subblock may have fewer than `qkt_subblock_w` columns. The partial-subblock code is guarded by `if constexpr (has_partial_subblock)` and compiles out on the standard path.

| sbh | Sk | qkt_subblock_w | kt_num_full_subblocks | DST tiles |
|-----|-----|----------------|----------------------|-----------|
| 1   | 16  | 8              | 2                    | 1×8=8     |
| 1   | 8   | 8              | 1                    | 1×8=8     |
| 2   | 16  | 4              | 4                    | 2×4=8     |
| 2   | 8   | 4              | 2                    | 2×4=8     |
| 2   | 4   | 4              | 1                    | 2×4=8     |

**Flow for each q_subblock (0..N-1):**

All Q@KT matmul output is written directly to `cb_qkt_im` at absolute tile offsets via `pack_tile<true>`. The entire `cb_qkt_im` (Sq_chunk_t × KT_stride tiles) is reserved upfront. After each row is written, `cb_push_back_hold_wr_ptr` makes the row visible to UNPACK (for reading) while rewinding the PACK `wr_ptr` back — so all subsequent `pack_tile<true>` offsets remain relative to the same stable base.

1. **Wait for Q tiles** (cumulative: already loaded by reader)
2. **Init matmul HW** for this q_subblock: `mm_block_init_short` (Wormhole) or `mm_no_mop_init_short` (Blackhole).
3. **For each kt_subblock (0..kt_num_full_subblocks-1):**
   - If q_subblock > 0: in-place sub_exp of the *previous* row for this kt_subblock
     - `sub_exp_block_bcast_cols()`: reads from cb_qkt_im at the previous row's absolute position (already pushed/fronted via held push), subtracts cur_max, applies exp with scale, writes back to the same position via `pack_tile<true>`, and reduces (L1 accum) to cur_sum. L1 acc is reset at the start of each sub-row (i) when at the first kt_subblock, ensuring the first write to each reduce position overwrites rather than accumulating with stale data.
     - After sub_exp: re-init matmul HW (`mm_block_init_short` / `mm_no_mop_reinit_short`) since sub_exp reconfigures the unpacker.
   - `blocked_matmul_and_pack()`: Q × KT for this subblock, packs to cb_qkt_im at absolute offset `(q_subblock * sbh + r) * KT_stride + kt_subblock * qkt_subblock_w + c`. Matmul init is done by the caller (not inside `blocked_matmul_and_pack`) for better control over re-init patterns.
4. **If `has_partial_subblock`:** Same as step 3 but with `kt_remainder`-wide subblock. MOP config and matmul HW are re-initialized for the narrow width before the partial matmul.
5. **If padded_k_tiles > 0 and last K chunk:** `apply_padded_mask()` — L1-accumulates -inf onto padded tile positions in cb_qkt_im at `(q_subblock * sbh + row) * KT_stride + col` (still in reserved state). Processes in batches of up to 8 tiles (DST capacity). **Note:** On the reduced path, `padded_k_tiles = 0` so this block compiles out entirely.
6. **`cb_push_back_hold_wr_ptr(cb_qkt_im, row_tiles)`** — makes the row visible to UNPACK for reading while keeping PACK's wr_ptr at the original base.
7. **Max reduce:** reads from cb_qkt_im at `in0_row_group_index = q_subblock`, writes to cur_max
   - `reduce_c_row_group()`: uses `reduce_block_max_row<Sk_chunk_t>` (reduces only valid columns) with `ROW_STRIDE = KT_stride` (for correct row indexing in the CB). Plus eltwise_max with prev_max (skipped on first K-chunk via `do_eltwise_max = !is_first_iter`).

**After all q_subblocks:** Pop KT (`head_dim_t * KT_stride` tiles). On the last K chunk, also pop Q early (see Early Q Pop above).

**Key optimization:** The in-place sub_exp of the *previous* row overlaps with the matmul of the *current* row's subblock. Since sub_exp uses SFPU (exp) while matmul uses FPU, they can overlap.

**Key optimization (L1 savings):** By writing directly to cb_qkt_im and using `cb_push_back_hold_wr_ptr` to keep wr_ptr stable, the two ping-pong row buffers (c4/c6, 32 tiles = 64 KB) are eliminated. The PACK side always sees wr_ptr at the beginning of cb_qkt_im, so absolute offsets address the correct positions regardless of how many rows have been pushed to the UNPACK side.

**Blackhole-specific optimizations (ARCH_BLACKHOLE):** On Blackhole, several LLK-level optimizations are enabled:
- **`matmul_block_no_mop` / `mm_no_mop_init_short` / `mm_no_mop_reinit_short`**: Bypass the MOP (Machine Operation Processor) for matmul, reducing overhead.
- **`blocked_pack=true`**: Both `blocked_matmul_and_pack` and `sub_exp_block_bcast_cols` use blocked packing — a single `pack_tile<true>` call per row writes `SUBBLOCK_W` contiguous tiles, instead of individual per-tile packing.
- **`sub_bcast_cols_init_short_custom` / `sub_tiles_bcast_cols_custom`**: Custom blocked subtraction that processes `tiles_per_column` tiles in a single call.
- **`llk_pack_mop_config`**: Configures the packing MOP for different tile widths at transition points (e.g., switching between qkt_subblock_w-wide matmul packing and 1-wide reduce packing).

### 5.4 Phase 2: QKT@V + SALAD

**Derived constants:**
- `qktv_subblock_w` = 4 (V matmul produces tiles in groups of 4)
- `qktv_v_num_subblocks` = head_dim_t / 4
- `qktv_in0_block_w` = Sv_chunk_t (= effective_Sk on reduced path)
- `qktv_in0_row_tiles` = qktv_subblock_h × KT_stride

**q_subblock 0 (drain + first V matmul):**

The drain performs in-place sub_exp on the last row of cb_qkt_im (row N-1, which has raw matmul output from Phase 1), interleaved with QKT@V matmul (FPU) for overlap. All rows are already pushed (fronted) via `cb_push_back_hold_wr_ptr` from Phase 1, so sub_exp reads the already-visible tiles and writes back to the same positions. No push/pop of row buffers is needed. The strategy depends on sbh and whether partial subblocks exist:

**sbh=1 with no partial subblock:** Split matmul along inner dimension. Loops `kt_num_full_subblocks` times, each iteration draining one sub_exp block (in-place on cb_qkt_im) then running a matmul with `inner_dim = Sv_chunk_t / kt_num_full_subblocks`. L1 accumulate for iterations after the first.
- kt_num_full_subblocks=2 (Sk=16): 2 iterations, each with inner dim 8
- kt_num_full_subblocks=1 (Sk=8): 1 iteration with full inner dim 8

**sbh>1 or has_partial_subblock:** Can't split along inner dimension — either because `matmul_block` uses `INNER_DIM` as the in0 row stride (must equal KT_stride for multi-row subblocks), or because the split would produce uneven inner dims. Instead, drains all sub_exp in-place first (full subblocks + partial subblock), then runs a single full-inner-dim matmul. SFPU/FPU overlap still occurs between the last sub_exp's EXP phase and the matmul's FPU phase.

**q_subblocks 1..N-1 (SALAD interleaved with V matmul):**

Before the loop, `exp_packthread_tile_init<EXP_APPROX_MODE, false>()` is called once to configure the exp hardware for sub_exp_first_col_blocks (used in exp_max_diff computation).

For each q_subblock:
1. **exp_max_diff** for *previous* row (**skipped on first K-chunk**): `sub_exp_first_col_blocks(prev_max, cur_max)` → cb_exp_max_diff
2. **Full V matmul** for *current* row: QKT_im[row_q, :] @ V → cur_out[row_q]. Uses `KT_stride` as both mm_block_init kt_dim and `KT_DIM_MATMUL` in `blocked_matmul_and_pack` (for correct in0 row stride in the unpack).
3. **SALAD corrections** for *previous* row (**skipped on first K-chunk**):
   - `mul_bcast_cols_l1_acc(prev_sum, exp_max_diff, cur_sum)`: rescale prev_sum into cur_sum
   - `mul_block_bcast_cols_acc(prev_out, exp_max_diff, cur_out)`: rescale prev_out into cur_out
4. If `is_last_iter`: `normalize_row()` pushes cur_sum/cur_out and calls `normalize_row_streaming()`:
   - For each tile row: matmul_reduce(sum × col_identity) → 1/sum → multiply output row → cb_normalized_out

On the first K-chunk, steps 1 and 3 are skipped entirely. If it is also the last K-chunk (num_k_chunks == 1), step 4 still runs — normalization is decoupled from SALAD via a separate `normalize_row` lambda.

**Pipeline drain:** SALAD + normalization for the last row (no more matmuls to overlap with). Also gated on `!is_first_iter` for SALAD; normalization runs unconditionally on `is_last_iter`.

**After Phase 2:** Pop V (`KT_stride * head_dim_t` tiles) and cb_qkt_im (`Sq_chunk_t * KT_stride` tiles). Both use KT_stride to keep CB pointer advancement aligned.

### 5.5 normalize_row_streaming (last K iteration only)

For each of the Sq_chunk_t tile rows:
1. `matmul_reduce`: sum_tile × col_identity → scratch (collapses partial row sums to a single value per row; this is needed because the row sum accumulated via L1 accum across K subblocks is still a full tile with partial sums)
2. `recip_tile_first_column`: scratch = 1/sum (computed directly in DST, fused with matmul)
3. `mul_tiles_bcast_cols`: output_tiles × bcast_cols(1/sum) → normalized_out (streamed to writer)

**Important:** recip and normalize only operate on column 0:8 of each face (VectorMode::C), consistent with the reduce_block_max_row output format where meaningful values are only in column 0.

### 5.6 Custom exp Implementation

The kernel uses `exp_packthread_tile` (PACK-thread variant of exp) and `calculate_exponential_polynomial` — both forked from the standard `exp_tile` / `ckernel_sfpu_exp.h`. The polynomial path (used when EXP_APPROX_MODE=false) computes:
1. Range reduction: x → k, r where x = k·ln(2) + r
2. Polynomial evaluation: exp(r) ≈ c0 + c1·r + c2·r² (degree 2 for fp16b, degree 4 for fp32)
3. Reconstruction: exp(x) = exp(r) · 2^k

Packer ReLU is enabled during exp to clamp any negative results from approximation errors (InputClamping::None mode).

**Init hoisting:** `exp_packthread_tile_init` for the Schraudolph fast-approx path (used in sub_exp_block_bcast_cols) is called at the start of `sdpa_inner_loop_step`. For the polynomial path (used in sub_exp_first_col_blocks), `exp_packthread_tile_init<EXP_APPROX_MODE, false>()` is called once before the Phase 2 q_subblock loop.

---

## 6. Skip-Padding Optimization <a id="6-skip-padding"></a>

When the K sequence length doesn't divide evenly into Sk_chunk_t-sized chunks, the last chunk is zero-padded. The skip-padding optimization eliminates all wasted computation on padded tiles.

### 6.1 Dual-Path Dispatch

`kernel_main` maintains two call lambdas:

```
call_step         → sdpa_inner_loop_step<..., Sk_chunk_t, Sv_chunk_t, ..., padded_k_tiles, ..., Sk_chunk_t>
call_step_reduced → sdpa_inner_loop_step<..., effective_Sk, effective_Sk, ..., 0, ..., Sk_chunk_t>
```

On the last K chunk when `padded_k_tiles > 0`, `call_step_reduced` is dispatched. The key template parameter differences:
- `Sk_chunk_t` / `Sv_chunk_t` = `effective_Sk` (compute dimension narrowed)
- `padded_k_tiles = 0` (masking compiled out)
- `KT_stride = original_Sk_chunk_t` (CB layout unchanged)

### 6.2 What Gets Skipped on the Reduced Path

Every compute operation uses `effective_Sk`:

| Operation | Standard Path | Reduced Path | Savings (pad=6/Sk=16) |
|-----------|--------------|--------------|----------------------|
| Q@KT matmul columns | Sk_chunk_t (16) | effective_Sk (10) | 37.5% fewer tile-MACs |
| sub_exp columns | Sk_chunk_t (16) | effective_Sk (10) | 37.5% fewer exp tiles |
| reduce_block_max_row | Sk_chunk_t (16) | effective_Sk (10) | 37.5% fewer columns |
| apply_padded_mask | padded_k_tiles (6) | 0 (compiled out) | 100% eliminated |
| V matmul inner dim | Sv_chunk_t (16) | effective_Sk (10) | 37.5% fewer inner iters |

### 6.3 KT_stride: Why CB Geometry Stays Full-Width

The cb_qkt_im circular buffer uses `KT_stride` (= original Sk_chunk_t) for all pointer management: reserve, push_hold, and pop. This is critical because `cb_push_back_hold_wr_ptr` does not advance the write pointer. If reserve/pop used the reduced `effective_Sk` amount, the pointer advancement would not be a multiple of the CB capacity, causing read/write pointer misalignment on subsequent Q iterations.

Concretely: with a 112-tile CB (7 × 16), each standard iteration advances pointers by 112 (wraps cleanly). A reduced iteration advancing by 70 (7 × 10) would leave pointers at offset 70, misaligning all subsequent pack_tile<true> offsets vs consumer read positions.

**KT_stride is used for:** cb_qkt_im reserve/push_hold/pop, OUT_NUM_COLS in Q@KT matmul, sub_exp cols_in_row, V matmul mm_block_init kt_dim and KT_DIM_MATMUL, qktv_in0_row_tiles, qktv_in0_index_offset stride, KT/V CB wait/pop.

**Sk_chunk_t (effective_Sk) is used for:** kt_num_full_subblocks / kt_remainder (matmul loop bounds), reduce_block_max_row<Sk_chunk_t> (reduce width), V matmul INNER_DIM = Sv_chunk_t.

### 6.4 reduce_c_row_group ROW_STRIDE Parameter

```cpp
template <PoolType pool_type, ReduceDim reduce_dim, uint32_t scale_cb,
          uint32_t cols, uint32_t SBH, uint32_t ROW_STRIDE = cols>
```

The `ROW_STRIDE` parameter (defaults to `cols`) separates the physical row stride in cb_qkt_im from the number of columns to reduce. On the standard path, `ROW_STRIDE == cols`. On the reduced path, `ROW_STRIDE = KT_stride` (full row width) while `cols = Sk_chunk_t` (= effective_Sk). The reduce reads only `effective_Sk` valid columns per row, skipping garbage in padded positions, while using `ROW_STRIDE` for row indexing and cumulative tile count.

### 6.5 Partial Subblock Handling

When `effective_Sk % qkt_subblock_w != 0` (e.g., effective_Sk=10, qkt_subblock_w=8 → remainder=2), the last subblock has width `kt_remainder` instead of `qkt_subblock_w`. This affects:
- **Phase 1:** After the full subblock loop, an `if constexpr (has_partial_subblock)` block handles the narrow subblock with re-initialized MOP config and matmul HW.
- **Phase 2 drain:** The split-drain (sbh=1) requires even inner dim splits. When `has_partial_subblock`, the non-split drain path is used instead (all sub_exp first, then one full V matmul).

---

## 7. CB Usage Static Analysis (sbh=1) <a id="7-cb-usage-static-analysis"></a>

### 7.1 Phase 1 CB Protocol Trace

**cb_qkt_im (direct write + held push):**
```
reserve_back(cb_qkt_im, 112)              # Sq_chunk_t * KT_stride, reserved at start
  -- for each q_subblock (0..6):
  [matmul writes to cb_qkt_im at absolute offsets via pack_tile<true>]
  [sub_exp overwrites prev row in-place at absolute offsets via pack_tile<true>]
  cb_push_back_hold_wr_ptr(cb_qkt_im, 16) # push row (UNPACK sees it), rewind wr_ptr
  [max_reduce reads from cb_qkt_im at q_subblock position]
  -- Phase 2: sub_exp drain writes last row in-place --
  -- Phase 2: V matmul reads via cumulative cb_wait_front --
pop_front(cb_qkt_im, 112)                # Sq_chunk_t * KT_stride, freed at end
```
Protocol: correct. Reservation is KT_stride × Sq_chunk_t tiles upfront; rows are exposed incrementally via `cb_push_back_hold_wr_ptr` (KT_stride tiles per q_subblock, 7 pushes = 112 tiles total). PACK's wr_ptr stays at the start throughout, so all `pack_tile<true>` offsets address absolute positions. UNPACK sees rows appear one at a time for cumulative `cb_wait_front`. Phase 2 drain writes the last row in-place (already pushed/fronted). Phase 2 pops all 112.

**cur_max (c27 or c28):**
```
reserve_back(cur_max, 1)                  # per q_subblock
  reduce_c_row_group packs via pack_tile<false>
push_back(cur_max, 1)                     # per q_subblock
  -- after all q_subblocks: cumulative 7 tiles available --
  [sub_exp_block_bcast_cols reads via cb_wait_front((q_subblock+1)*1)]
  [sub_exp_first_col_blocks reads via cb_wait_front((salad_row+1)*1)]
  -- cleanup --
pop_front(cur_max, 7)
```
Protocol: correct.

**cur_sum (c29 or c30):**
```
reserve_back(cur_sum, 7)                  # at start of inner_loop_step
  [sub_exp_block_bcast_cols does L1 accum into cur_sum at absolute offsets]
  -- if !is_first_iter: SALAD corrections also L1 accum into cur_sum --
  -- if last_iter: push_back + normalize consumes --
  -- if not last_iter: push_back(cur_sum, 7) at end --
pop_front(cur_sum, 7)                     # in next iteration's cleanup (skipped on first iter)
```
Protocol: correct. The reservation of 7 tiles upfront creates the write region for L1 accumulate.

### 7.2 Phase 2 CB Protocol Trace

**cur_out (c25 or c26):**
```
reserve_back(cur_out, 28)                 # Sq_chunk_t * head_dim_t = 7*4
  [V matmul writes via pack_tile<true> at absolute offsets]
  [SALAD correction L1-accumulates via pack_tile<true>]
  -- if last_iter: push_back per row (sbh*head_dim_t = 4), normalize, pop --
  -- if not last_iter: push_back(cur_out, 28) at end --
pop_front(cur_out, 28)                    # in next iteration's cleanup
```
Protocol: correct.

**cb_exp_max_diff (c31):**
```
-- Only produced when !is_first_iter:
-- per q_subblock (1..6) + pipeline drain:
reserve_back(cb_exp_max_diff, 1)
  sub_exp_first_col_blocks packs 1 tile
push_back(cb_exp_max_diff, 1)
  -- SALAD reads from exp_max_diff, cumulative wait --
-- at end of outer K loop (skipped on first iter):
pop_front(cb_exp_max_diff, 7)
```
Protocol: correct.

### 7.3 Normalization CB Protocol

**cb_recip_scratch (c10):**
```
reserve_back(1) → matmul+recip → pack_tile → push_back(1)
  → mul_bcast_cols reads → pop_front(1)
```
Protocol: correct. 1-tile scratch, used and freed per row.

**cb_normalized_out (c9):**
```
reserve_back(head_dim_t=4) → pack head_dim_t tiles → push_back(4)
  → writer waits for head_dim_t batch: wait_front(4) → write 4 tiles back-to-back → pop_front(4)
```
Protocol: correct. Double-buffered (2 × head_dim_t = 8 pages).

---

## 8. Indexing Correctness Audit <a id="8-indexing-correctness-audit"></a>

### 8.1 Q@KT Matmul Indexing

```cpp
// blocked_matmul_and_pack — always uses pack_tile<true> at absolute offsets
in0_index_start = q_index_offset = q_subblock * sbh * head_dim_t
in1_index_start = kt_index_offset = kt_subblock * qkt_subblock_w
// Output: pack_tile<true> at (q_subblock * sbh + r) * KT_stride + kt_subblock * qkt_subblock_w + c
```

For sbh=1, q_subblock=2: `in0_index_start = 2 * 1 * 4 = 8` → reads Q tiles [8..11]
For kt_subblock=1: `in1_index_start = 1 * 8 = 8` → reads KT tiles starting at 8

Inside blocked_matmul_and_pack:
```cpp
for inner = 0..head_dim_t-1:
    matmul_block(cb_q, cb_kt, in0_index, in1_index, ...)
    in0_index++       // Q: 8, 9, 10, 11
    in1_index += KT_stride  // KT: 8, 24, 40, 56 (stride by KT_stride=16)
```

KT layout in CB: `[d=4 rows × KT_stride cols]` transposed from `[Sk × d]`. So tile at `(d_idx, sk_idx)` is at `d_idx * KT_stride + sk_idx`. Reading KT[8] = (0, 8), KT[24] = (1, 8), KT[40] = (2, 8), KT[56] = (3, 8). This is correct for accumulating over the inner dimension.

Pack output for q_subblock=2, kt_subblock=1, r=0: `out_row_offset = (0 + 2*1) * KT_stride = 32`, plus `out_col_offset = 1*8 = 8` → writes at tile 40. ✓

**Verdict: CORRECT**

### 8.2 sub_exp_block_bcast_cols Indexing (in-place on cb_qkt_im)

```cpp
in0_tile_index = (max_row_base + i) * cols_in_row + global_col_base + j
max_tile = max_row_base + i
```

`cols_in_row = KT_stride` matches the physical cb_qkt_im row layout.

For q_subblock=2, kt_subblock=1 (sbh=1):
- `max_row_base = 2 * 1 = 2`
- `global_col_base = 1 * 8 = 8`
- `in0_tile_index = (2 + 0) * 16 + 8 + j` for j=0..7 → tiles [40..47] ✓ (row 2, second half)
- `max_tile = 2 * 1 + 0 = 2` → reads max tile for row 2 ✓

**Verdict: CORRECT**

### 8.3 QKT@V Matmul Indexing

```cpp
qktv_in0_index_offset advances by qktv_subblock_h * KT_stride = 1 * 16 = 16 per q_subblock
```

For q_subblock=0: reads cb_qkt_im tiles [0..Sv_chunk_t-1] (first effective_Sk tiles of row 0)
For q_subblock=1: reads [16..16+Sv_chunk_t-1], etc.

V matmul uses `INNER_DIM = Sv_chunk_t` (= effective_Sk on reduced path), `KT_DIM_MATMUL = KT_stride` (for correct in0 row stride when sbh>1).

**Verdict: CORRECT**

### 8.4 SALAD Correction Indexing

```cpp
mul_bcast_cols_l1_acc<sbh>(prev_sum, cb_exp_max_diff, cur_sum, salad_row, w_salad)
mul_block_bcast_cols_acc<sbh, head_dim_t>(prev_out, cb_exp_max_diff, cur_out, salad_row, w_salad)
```

**Verdict: CORRECT**

### 8.5 DST Capacity

With sbh=1:
- Q@KT: sbh × qkt_subblock_w = 1 × 8 = 8 tiles → **at limit** (8 tiles max with fp16b double-buffer)
- sub_exp_block_bcast_cols: sbh × qkt_subblock_w = 1 × 8 = 8 tiles → **at limit**
- QKT@V: qktv_subblock_h × qktv_subblock_w = 1 × 4 = 4 tiles → OK
- normalize: head_dim_t = 4 tiles → OK

**Verdict: All within 8-tile DST limit for fp16b double-buffer mode.**

---

## 9. Comparison with Reference SDPA (compute_common.hpp) <a id="9-comparison-with-reference"></a>

### 9.1 Structural Differences

| Feature | Reference SDPA | This Implementation |
|---------|---------------|---------------------|
| **Q@KT matmul** | Single `matmul_blocks()` call, full Q@KT at once | Row-by-row directly into cb_qkt_im via `pack_tile<true>` |
| **Softmax** | `sub_exp_block_bcast_cols_inplace()` — in-place on cb_qkt_im | `sub_exp_block_bcast_cols()` — in-place on cb_qkt_im (reads/writes same absolute positions) |
| **Max reduce** | `reduce_c()` on full cb_qkt_im | `reduce_c_row_group()` per row, interleaved with matmul; supports separate ROW_STRIDE for skip-padding |
| **SALAD correction** | Separate steps: `sub_exp_block`, `mul_tiles_bcast_cols_inplace`, `add_block_inplace`, `mul_block_bcast_cols` | Fused: `sub_exp_first_col_blocks` + `mul_bcast_cols_l1_acc` + `mul_block_bcast_cols_acc` |
| **QKT@V matmul** | Single `matmul_blocks()` call | Subblock-by-subblock with overlap |
| **Final normalization** | `matmul_reduce` + `recip_block_inplace` + `mul_block_bcast_cols` (all in-place) | `normalize_row_streaming()` — per-row streaming with fused matmul+recip |
| **Row buffering** | Full cb_qkt_im written at once | `cb_push_back_hold_wr_ptr` keeps wr_ptr stable (no row buffers needed) |
| **Q pop timing** | After all K chunks complete | After Phase 1 of last K chunk (early pop) |
| **Normalized output buffering** | Single-buffered | Double-buffered (2 × head_dim_t), batched NoC writes |
| **Blackhole LLK optimizations** | None | blocked_pack, no_mop matmul, custom blocked sub/bcast, MOP config |
| **Skip-padding** | Not supported (always processes full chunks) | Dual-path dispatch with reduced instantiation; narrower reduce via ROW_STRIDE; masking eliminated |
| **Causal masking** | Full support via `add_block_inplace(qk, mask)` | **NOT IMPLEMENTED** |
| **Attention sink** | Full support | **NOT IMPLEMENTED** |
| **Ring attention** | Full support (RING type) | **NOT IMPLEMENTED** |
| **Provided mask** | Full support | **NOT IMPLEMENTED** |
| **K-chunk transposition** | Done by compute kernel (matmul with `transpose=true`) | Done by reader kernel (tile grid transpose during DMA); matmul still uses `transpose=true` for within-tile transpose |

### 9.2 Algorithmic Differences

**Reference approach (per K-chunk):**
1. Q@KT → cb_qk_im (full matrix at once)
2. Optionally add mask
3. reduce_max → cur_max (with eltwise_max against prev_max)
4. sub_exp_inplace on cb_qk_im (modifies in place, also reduces to cur_sum)
5. QKT@V → cur_out
6. If not first K-chunk: SALAD correction (separate sub_exp, mul_inplace, add_inplace, mul_bcast)
7. Swap ping-pong
8. After all K-chunks: matmul_reduce(sum), recip_inplace(sum), mul_bcast(out, 1/sum)

**This implementation (per K-chunk):**
1. Phase 1: Row-by-row Q@KT directly into cb_qkt_im with interleaved softmax processing
   - Pipelined: matmul row N while softmax'ing row N-1
   - Supports partial subblocks for non-subblock-aligned effective_Sk
   - `reduce_c_row_group` with ROW_STRIDE enables narrower reduce on reduced path
   - First K-chunk: `do_eltwise_max=false`
   - On last K chunk: pop Q early
2. Phase 2: Row-by-row QKT@V with interleaved SALAD
   - Pipelined: V matmul for row N while SALAD-correcting row N-1
   - Overlap drain loop: last row's softmax interleaved with first V matmul
   - Non-split drain for partial subblock cases
3. On last K-chunk: per-row normalization streamed directly to output
   - Double-buffered normalized_out CB with batched writer NoC writes

---

## 10. Missing Features for Numerical Correctness <a id="10-missing-features"></a>

### 10.1 Causal Masking (HIGH PRIORITY)

The reference SDPA adds a causal mask to the attention scores before softmax. This is a critical correctness feature for decoder-only models. Without it, the kernel computes bidirectional attention (correct for encoder models, incorrect for GPT/LLaMA).

### 10.2 Sliding Window Attention

The reference supports `sliding_window_size > 0` for local attention patterns. Not implemented here.

### 10.3 Attention Sink

The reference supports attention sink logits. Not implemented here.

### 10.4 Padded Mask Support (DONE)

Tile-aligned padded K masking is fully implemented via the skip-padding optimization (Section 6). On the standard path (non-last K chunks), masking is handled by `apply_padded_mask()`. On the reduced path (last K chunk with padding), masking is eliminated entirely — the reduced instantiation computes only on valid tiles.

### 10.5 bfp8_b K/V Support (DONE)

When `kv_bf8b=1`, K and V use bfloat8_b data format (1 KB/tile vs 2 KB for bf16), halving K/V L1 footprint. Enabled via host-side data format configuration.

---

## 11. Suggested Next Steps <a id="11-next-steps"></a>

### Step 1: Add Causal Masking

This is the single most important feature for practical SDPA.

### Step 2: Multi-Head Extension

Wrap the current single-core kernel in a multi-core dispatch that assigns different heads to different cores.

### Step 3: Performance Optimizations

1. **Reader-side skip-padding:** Eliminate DRAM reads for padded K/V tiles on the last chunk. For pad=8/Sk=16 with head_dim_t=4: saves 64 tile reads (131 KB DRAM bandwidth).
2. **Granularity tuning:** The reference uses configurable granularities for tile-processing. Adding similar configurability could help tune for different problem sizes.
3. **Further Blackhole LLK optimizations:** Blocked reduce, additional MOP bypasses.

### Step 4: Ring Attention (Future)

For multi-device scenarios, implement the ring attention pattern from the reference.

---

## Appendix A: Helper Function Reference

| Function | Purpose |
|----------|---------|
| `cb_push_back_hold_wr_ptr` | Push tiles (UNPACK sees them) but rewind PACK's wr_ptr, keeping pack_tile\<true\> offsets stable |
| `sub_exp_block_bcast_cols` | In-place on inout_cb: subtract max (bcast cols), apply exp, pack back + L1 accum reduce. On Blackhole with `blocked_pack=true`: uses blocked sub/pack for fewer HW calls. |
| `sub_exp_first_col_blocks` | Subtract prev_max from cur_max, apply exp (column 0 only) → correction factor |
| `mul_bcast_cols_l1_acc` | Multiply + bcast_cols, L1 accumulate into output |
| `mul_block_bcast_cols_acc` | Block multiply + bcast_cols, L1 accumulate (for output rescaling) |
| `blocked_matmul_and_pack` | Blocked matmul with pack to CB via pack_tile\<true\> at absolute offsets. `KT_DIM_MATMUL` template param controls in0 row stride (defaults to INNER_DIM). On Blackhole with `blocked_pack=true`: blocked packing. |
| `reduce_c_row_group` | Max reduce across rows using reduce_block_max_row + eltwise_max. `ROW_STRIDE` template param separates physical row stride from reduce width. |
| `normalize_row_streaming` | Per-row normalization: matmul_reduce + recip + bcast multiply |
| `normalize_row` (lambda) | Push sum/out tiles, call normalize_row_streaming (decoupled from SALAD) |
| `apply_padded_mask` | L1-accumulate -inf onto padded K tile positions (batched, q_subblock-aware) |
| `calculate_exponential_polynomial` | Custom polynomial exp (degree 1-4) |

## Appendix B: Compile-Time Arguments

### Compute kernel

| Index | Name | Default | Description |
|-------|------|---------|-------------|
| 0 | `Sq_chunk_t` | 7 | Query chunk height in tiles (must be even when sbh=2) |
| 1 | `Sk_chunk_t` | 16 (or 8, 4) | Key chunk width in tiles |
| 2 | `Sv_chunk_t` | 16 (or 8, 4) | Value chunk height in tiles (= Sk_chunk_t) |
| 3 | `head_dim_t` | 4 | Head dimension in tiles (128/32 = 4) |
| 4 | `num_q_chunks` | 2 | Number of Q chunks |
| 5 | `num_k_chunks` | 3 | Number of K/V chunks |
| 6 | `scale_fp32` | 1/sqrt(128) as uint32 | Scale factor (bit-cast float) |
| 7 | `subblock_h` | 1 | Subblock height (1 or 2) |
| 8 | `padded_k_tiles` | 0 | Zero-padded K tiles in last chunk (0 = no masking) |

### Writer kernel

| Index | Name | Default | Description |
|-------|------|---------|-------------|
| 0-5 | (same as compute 0-5) | | |
| 6 | `identity_scalar_packed` | 1.0 packed | Two bfloat16 1.0s packed into uint32 |
| 7 | `subblock_h` | 1 | Subblock height (must match compute) |
| 8 | `padded_k_tiles` | 0 | Zero-padded K tiles in last chunk |
| 9+ | TensorAccessorArgs | | Output DRAM buffer accessor |

## Appendix C: Defines

| Define | Effect |
|--------|--------|
| `EXP_APPROX_MODE` | 0=polynomial exp, 1=Schraudolph piecewise approximation |
| `PROFILE_KERNEL` | Enable Tracy profiling zones |
| `MM_THROTTLE` | Matmul throttle level (1-5) |

## Appendix D: Blackhole-Specific LLK Functions

These functions are used via `#ifdef ARCH_BLACKHOLE` guards and provide performance optimizations specific to the Blackhole architecture:

| Function | Purpose |
|----------|---------|
| `matmul_block_no_mop` | Matmul bypassing MOP for reduced overhead |
| `mm_no_mop_init_short` | Initialize matmul HW without MOP setup |
| `mm_no_mop_reinit_short` | Re-initialize matmul HW after operation switch (lighter than full init) |
| `sub_bcast_cols_init_short_custom` | Initialize blocked column-broadcast subtraction |
| `sub_tiles_bcast_cols_custom` | Blocked column-broadcast subtraction (processes `tiles_per_column` tiles per call) |
| `llk_pack_mop_config` | Configure packing MOP for a specific tile width (called at operation transitions) |
