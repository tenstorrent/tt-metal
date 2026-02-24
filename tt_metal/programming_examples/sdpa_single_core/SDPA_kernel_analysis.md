# SDPA Single-Core Kernel Analysis

Comprehensive analysis of the in-progress single-core Scaled Dot-Product Attention (SDPA) implementation at `tt_metal/programming_examples/sdpa_single_core/`. Intended as a starting point for evolving this perf-benchmarking playground into a numerically correct SDPA.

---

## Table of Contents
1. [Algorithm Overview: Online Softmax / Flash Attention](#1-algorithm-overview)
2. [Kernel Structure and Roles](#2-kernel-structure)
3. [Circular Buffer Map](#3-circular-buffer-map)
4. [Data Flow: End-to-End](#4-data-flow)
5. [Compute Kernel Detailed Walkthrough](#5-compute-kernel-walkthrough)
6. [CB Usage Static Analysis (sbh=1)](#6-cb-usage-static-analysis)
7. [Indexing Correctness Audit](#7-indexing-correctness-audit)
8. [Comparison with Reference SDPA (compute_common.hpp)](#8-comparison-with-reference)
9. [Missing Features for Numerical Correctness](#9-missing-features)
10. [Suggested Next Steps](#10-next-steps)

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
- Streams normalized output tiles from `cb_normalized_out` (c9) to DRAM, one row of tiles at a time

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
| c9 | `cb_normalized_out` | 4 | Normalized output streaming | Compute | Writer |
| c10 | `cb_recip_scratch` | 1 | 1/sum scratch | Compute | Compute |
| c25 | `cb_out_A` | 7×4 = 28 | Ping-pong output accumulator A | Compute | Compute |
| c26 | `cb_out_B` | 7×4 = 28 | Ping-pong output accumulator B | Compute | Compute |
| c27 | `cb_max_A` | 7 | Ping-pong max A | Compute | Compute |
| c28 | `cb_max_B` | 7 | Ping-pong max B | Compute | Compute |
| c29 | `cb_sum_A` | 7 | Ping-pong sum A | Compute | Compute |
| c30 | `cb_sum_B` | 7 | Ping-pong sum B | Compute | Compute |
| c31 | `cb_exp_max_diff` | 7 | exp(prev_max - cur_max) | Compute | Compute |

**Total L1 usage (tiles):** 56 + 128 + 112 + 128 + 1 + 1 + 4 + 1 + 28 + 28 + 7 + 7 + 7 + 7 + 7 = **521 tiles** × 2048 bytes = **~1.04 MB** (well within 1.5 MB L1)

**Note:** CB indices c4 and c6 are unused. The row buffers that previously occupied these slots were eliminated by the `cb_push_back_hold_wr_ptr` optimization (see Section 5.2), saving 32 tiles (64 KB) of L1.

---

## 4. Data Flow: End-to-End <a id="4-data-flow"></a>

### Outer Loop Structure
```
for each Q chunk (q = 0..num_q_chunks-1):
    for each K chunk (k = 0..num_k_chunks-1):
        is_first = (k == 0)
        is_last  = (k == num_k_chunks - 1)
        sdpa_inner_loop_step(..., is_last, is_first)
        # Post-iteration: pop prev buffers (skip on first iter), swap ping-pong aliases
        # On last K: per-row normalize → cb_normalized_out → writer drains to DRAM
    pop Q
```

No `InitPrevBuffers` is needed — the first K-chunk iteration uses `is_first_iter` gating to skip all prev-buffer reads (eltwise_max, SALAD corrections, exp_max_diff). This follows the same pattern as the reference SDPA's `processed_k_chunks > 0` guard.

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
```

### Ping-Pong Buffer Strategy

The kernel uses three pairs of ping-pong buffers:
- **max**: `cb_max_A` / `cb_max_B` — running row maximums
- **sum**: `cb_sum_A` / `cb_sum_B` — running row sums (partial softmax denominator)
- **out**: `cb_out_A` / `cb_out_B` — running weighted-value accumulator

After each K-chunk, the "current" buffers become "previous" via pointer swaps. The "previous" buffers are popped (freed) after being consumed by the SALAD correction of the next iteration. On the first K-chunk (`is_first_iter`), no prev buffers exist to pop — the cleanup is skipped.

---

## 5. Compute Kernel Detailed Walkthrough <a id="5-compute-kernel-walkthrough"></a>

### 5.1 First-Iteration Gating (`is_first_iter`)

Instead of pre-filling dummy "previous" buffers (zeros for sum/out, -inf for max), the kernel uses an `is_first_iter` flag to skip all operations that read from prev buffers on the first K-chunk:

1. **Phase 1 — `reduce_c_row_group`**: passes `do_eltwise_max = !is_first_iter`. When false, the max-reduce writes directly into DST (zeroed by `tile_regs_acquire`) without comparing against prev_max.
2. **Phase 2 — exp_max_diff + SALAD**: entirely skipped on first iteration. The V matmul result is the cur_out directly, and the sub_exp row sums are the cur_sum directly — no rescaling correction needed.
3. **Post-iteration cleanup**: prev buffer pops are skipped (they were never filled).

This matches the reference SDPA's `processed_k_chunks > 0` pattern and eliminates the per-Q-chunk `InitPrevBuffers` overhead and the `cb_neginf` tile.

### 5.2 sdpa_inner_loop_step — Phase 1: Q@KT with Softmax

**Derived constants:**
- `qkt_subblock_w` = 8/sbh (8 when sbh=1, 4 when sbh=2)
- `q_num_subblocks` = Sq_chunk_t/sbh
- `kt_num_subblocks` = Sk_chunk_t/qkt_subblock_w (1-4 depending on sbh and Sk)
- `row_tiles` = sbh×Sk_chunk_t

| sbh | Sk | qkt_subblock_w | kt_num_subblocks | DST tiles |
|-----|-----|----------------|------------------|-----------|
| 1   | 16  | 8              | 2                | 1×8=8     |
| 1   | 8   | 8              | 1                | 1×8=8     |
| 2   | 16  | 4              | 4                | 2×4=8     |
| 2   | 8   | 4              | 2                | 2×4=8     |
| 2   | 4   | 4              | 1                | 2×4=8     |

**Flow for each q_subblock (0..6):**

All Q@KT matmul output is written directly to `cb_qkt_im` at absolute tile offsets via `pack_tile<true>`. The entire `cb_qkt_im` (Sq_chunk_t × Sk_chunk_t tiles) is reserved upfront. After each row is written, `cb_push_back_hold_wr_ptr` makes the row visible to UNPACK (for reading) while rewinding the PACK `wr_ptr` back — so all subsequent `pack_tile<true>` offsets remain relative to the same stable base.

1. **Wait for Q tiles** (cumulative: already loaded by reader)
2. **For each kt_subblock (0..kt_num_subblocks-1):**
   - If q_subblock > 0: in-place sub_exp of the *previous* row for this kt_subblock
     - `sub_exp_block_bcast_cols()`: reads from cb_qkt_im at the previous row's absolute position (already pushed/fronted via held push), subtracts cur_max, applies exp with scale, writes back to the same position via `pack_tile<true>`, and reduces (L1 accum) to cur_sum. L1 acc is reset at the start of each sub-row (i) when at the first kt_subblock, ensuring the first write to each reduce position overwrites rather than accumulating with stale data.
   - `blocked_matmul_and_pack()`: Q × KT for this subblock, packs to cb_qkt_im at absolute offset `(q_subblock * sbh + r) * Sk_chunk_t + kt_subblock * qkt_subblock_w + c`.
3. **If padded_k_tiles > 0 and last K chunk:** `apply_padded_mask()` — L1-accumulates -inf onto padded tile positions in cb_qkt_im at `(q_subblock * sbh + row) * Sk_chunk_t + col` (still in reserved state). Processes in batches of up to 8 tiles (DST capacity).
4. **`cb_push_back_hold_wr_ptr(cb_qkt_im, row_tiles)`** — makes the row visible to UNPACK for reading while keeping PACK's wr_ptr at the original base.
5. **Max reduce:** reads from cb_qkt_im at `in0_row_group_index = q_subblock`, writes to cur_max
   - `reduce_c_row_group()`: uses `reduce_block_max_row` (block-based, not per-tile reduce_tile), plus eltwise_max with prev_max (skipped on first K-chunk via `do_eltwise_max = !is_first_iter`)

**Key optimization:** The in-place sub_exp of the *previous* row overlaps with the matmul of the *current* row's subblock. Since sub_exp uses SFPU (exp) while matmul uses FPU, they can overlap.

**Key optimization (L1 savings):** By writing directly to cb_qkt_im and using `cb_push_back_hold_wr_ptr` to keep wr_ptr stable, the two ping-pong row buffers (c4/c6, 32 tiles = 64 KB) are eliminated. The PACK side always sees wr_ptr at the beginning of cb_qkt_im, so absolute offsets address the correct positions regardless of how many rows have been pushed to the UNPACK side.

### 5.3 sdpa_inner_loop_step — Phase 2: QKT@V + SALAD

**Derived constants:**
- `qktv_subblock_w` = 4 (V matmul produces tiles in groups of 4)
- `qktv_v_num_subblocks` = head_dim_t / 4 = 1

**q_subblock 0 (drain + first V matmul):**

The drain performs in-place sub_exp on the last row of cb_qkt_im (row N-1, which has raw matmul output from Phase 1), interleaved with QKT@V matmul (FPU) for overlap. All rows are already pushed (fronted) via `cb_push_back_hold_wr_ptr` from Phase 1, so sub_exp reads the already-visible tiles and writes back to the same positions. No push/pop of row buffers is needed. The strategy depends on sbh:

**sbh=1:** Split matmul along inner dimension. Loops `kt_num_subblocks` times, each iteration draining one sub_exp block (in-place on cb_qkt_im) then running a matmul with `inner_dim = Sv_chunk_t / kt_num_subblocks`. L1 accumulate for iterations after the first.
- kt_num_subblocks=2 (Sk=16): 2 iterations, each with inner dim 8
- kt_num_subblocks=1 (Sk=8): 1 iteration with full inner dim 8

**sbh>1:** Can't split along inner dimension because `matmul_block` uses `INNER_DIM` as the in0 row stride, which must equal `Sk_chunk_t` for multi-row subblocks. Instead, drains all sub_exp in-place first, then runs a single full-inner-dim matmul. SFPU/FPU overlap still occurs between the last sub_exp's EXP phase and the matmul's FPU phase.

**q_subblocks 1..6 (SALAD interleaved with V matmul):**

For each q_subblock:
1. **exp_max_diff** for *previous* row (**skipped on first K-chunk**): `sub_exp_first_col_blocks(prev_max, cur_max)` → cb_exp_max_diff
2. **Full V matmul** for *current* row: QKT_im[row_q, :] @ V → cur_out[row_q]
3. **SALAD corrections** for *previous* row (**skipped on first K-chunk**):
   - `mul_bcast_cols_l1_acc(prev_sum, exp_max_diff, cur_sum)`: rescale prev_sum into cur_sum
   - `mul_block_bcast_cols_acc(prev_out, exp_max_diff, cur_out)`: rescale prev_out into cur_out
4. If `is_last_iter`: `normalize_row()` pushes cur_sum/cur_out and calls `normalize_row_streaming()`:
   - For each tile row: matmul_reduce(sum × col_identity) → 1/sum → multiply output row → cb_normalized_out

On the first K-chunk, steps 1 and 3 are skipped entirely. If it is also the last K-chunk (num_k_chunks == 1), step 4 still runs — normalization is decoupled from SALAD via a separate `normalize_row` lambda.

**Pipeline drain:** SALAD + normalization for the last row (no more matmuls to overlap with). Also gated on `!is_first_iter` for SALAD; normalization runs unconditionally on `is_last_iter`.

### 5.4 normalize_row_streaming (last K iteration only)

For each of the Sq_chunk_t tile rows:
1. `matmul_reduce`: sum_tile × col_identity → scratch (collapses partial row sums to a single value per row; this is needed because the row sum accumulated via L1 accum across K subblocks is still a full tile with partial sums)
2. `recip_tile_first_column`: scratch = 1/sum (computed directly in DST, fused with matmul)
3. `mul_tiles_bcast_cols`: output_tiles × bcast_cols(1/sum) → normalized_out (streamed to writer)

**Important:** recip and normalize only operate on column 0:8 of each face (VectorMode::C), consistent with the reduce_block_max_row output format where meaningful values are only in column 0.

### 5.5 Custom exp Implementation

The kernel uses `exp_packthread_tile` (PACK-thread variant of exp) and `calculate_exponential_polynomial` — both forked from the standard `exp_tile` / `ckernel_sfpu_exp.h`. The polynomial path (used when EXP_APPROX_MODE=false) computes:
1. Range reduction: x → k, r where x = k·ln(2) + r
2. Polynomial evaluation: exp(r) ≈ c0 + c1·r + c2·r² (degree 2 for fp16b, degree 4 for fp32)
3. Reconstruction: exp(x) = exp(r) · 2^k

Packer ReLU is enabled during exp to clamp any negative results from approximation errors (InputClamping::None mode).

---

## 6. CB Usage Static Analysis (sbh=1) <a id="6-cb-usage-static-analysis"></a>

### 6.1 Phase 1 CB Protocol Trace

**cb_qkt_im (direct write + held push):**
```
reserve_back(cb_qkt_im, 112)              # reserved at start of inner_loop_step
  -- for each q_subblock (0..6):
  [matmul writes to cb_qkt_im at absolute offsets via pack_tile<true>]
  [sub_exp overwrites prev row in-place at absolute offsets via pack_tile<true>]
  cb_push_back_hold_wr_ptr(cb_qkt_im, 16) # push row (UNPACK sees it), rewind wr_ptr (PACK base stable)
  [max_reduce reads from cb_qkt_im at q_subblock position]
  -- Phase 2: sub_exp drain writes last row in-place --
  -- Phase 2: V matmul reads via cumulative cb_wait_front --
pop_front(cb_qkt_im, 112)                # freed at end of Phase 2
```
Protocol: correct. Reservation is 112 tiles upfront; rows are exposed incrementally via `cb_push_back_hold_wr_ptr` (16 tiles per q_subblock, 7 pushes = 112 tiles total). PACK's wr_ptr stays at the start throughout, so all `pack_tile<true>` offsets address absolute positions. UNPACK sees rows appear one at a time for cumulative `cb_wait_front`. Phase 2 drain writes the last row in-place (already pushed/fronted). Phase 2 pops all 112.

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
Protocol: correct. The reservation of 7 tiles upfront creates the write region for L1 accumulate. On first K-chunk, only sub_exp contributes to cur_sum (no SALAD L1 accum from prev_sum).

### 6.2 Phase 2 CB Protocol Trace

**cur_out (c25 or c26):**
```
reserve_back(cur_out, 28)                 # Sq_chunk_t * head_dim_t = 7*4
  [V matmul writes via pack_tile<true> at absolute offsets]
  [SALAD correction L1-accumulates via pack_tile<true>]
  -- if last_iter: push_back per row (sbh*head_dim_t = 4), normalize, pop --
  -- if not last_iter: push_back(cur_out, 28) at end --
pop_front(cur_out, 28)                    # in next iteration's cleanup
```
Protocol: correct. The entire 28-tile region stays reserved until all SALAD corrections complete.

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
Protocol: correct. On first K-chunk, exp_max_diff is never produced or consumed. On subsequent iterations: 7 reserve+push calls, then one bulk pop.

### 6.3 Normalization CB Protocol

**cb_recip_scratch (c10):**
```
reserve_back(1) → matmul+recip → pack_tile → push_back(1)
  → mul_bcast_cols reads → pop_front(1)
```
Protocol: correct. 1-tile scratch, used and freed per row.

**cb_normalized_out (c9):**
```
reserve_back(head_dim_t=4) → pack head_dim_t tiles → push_back(4)
  → writer reads 1 tile at a time: wait_front(1) → write → pop_front(1)
```
Protocol: correct. Writer consumes tiles as they become available.

### 6.4 One-Off Push/Pop Operations

| Operation | Location | Description | Assessment |
|-----------|----------|-------------|------------|
| `cb_identity_scale_in` push (writer) | writer.cpp:19 | generate_reduce_scaler pushes 1, never popped | **Correct** — permanent scaler |
| `cb_col_identity` push (writer) | writer.cpp:23 | generate_bcast_col_scalar pushes 1, never popped | **Correct** — permanent tile |

### 6.5 L1 Accumulation Discipline

All L1 accumulate uses follow the pattern:
1. `cb_reserve_back` creates the target region
2. First tile written without L1 accum (or with accum onto zeros)
3. `llk_pack_reconfig_l1_acc(1)` enabled after first tile
4. Subsequent tiles accumulated
5. `llk_pack_reconfig_l1_acc(0)` disabled after batch
6. `cb_push_back` publishes

In `sub_exp_block_bcast_cols`: The L1 accum enable/disable logic for the reduce is handled per-row:
- Lines 511-519: For `global_col_base > 0`, enables L1 accum before the first tile. For `global_col_base == 0`, enables after the first tile (first tile overwrites, subsequent tiles accumulate). This is **correct** — the first kt_subblock of each row writes fresh, subsequent subblocks accumulate.

---

## 7. Indexing Correctness Audit <a id="7-indexing-correctness-audit"></a>

### 7.1 Q@KT Matmul Indexing

```cpp
// blocked_matmul_and_pack — always uses pack_tile<true> at absolute offsets
in0_index_start = q_index_offset = q_subblock * sbh * head_dim_t
in1_index_start = kt_index_offset = kt_subblock * qkt_subblock_w
// Output: pack_tile<true> at (q_subblock * sbh + r) * Sk_chunk_t + kt_subblock * qkt_subblock_w + c
```

For sbh=1, q_subblock=2: `in0_index_start = 2 * 1 * 4 = 8` → reads Q tiles [8..11]
For kt_subblock=1: `in1_index_start = 1 * 8 = 8` → reads KT tiles starting at 8

Inside blocked_matmul_and_pack:
```cpp
for inner = 0..head_dim_t-1:
    matmul_block(cb_q, cb_kt, in0_index, in1_index, ...)
    in0_index++       // Q: 8, 9, 10, 11
    in1_index += Sk_chunk_t  // KT: 8, 24, 40, 56 (stride by Sk_chunk_t=16)
```

KT layout in CB: `[d=4 rows × Sk=16 cols]` transposed from `[Sk × d]`. So tile at `(d_idx, sk_idx)` is at `d_idx * Sk_chunk_t + sk_idx`. Reading KT[8] = (0, 8), KT[24] = (1, 8), KT[40] = (2, 8), KT[56] = (3, 8). This is correct for accumulating over the inner dimension.

Pack output for q_subblock=2, kt_subblock=1, r=0: `out_row_offset = (0 + 2*1) * 16 = 32`, plus `out_col_offset = 1*8 = 8` → writes at tile 40. ✓

**Verdict: CORRECT**

### 7.2 sub_exp_block_bcast_cols Indexing (in-place on cb_qkt_im)

```cpp
in0_tile_index = (max_row_base + i) * cols_in_row + global_col_base + j
max_tile = max_row_base + i
```

For q_subblock=2, kt_subblock=1 (sbh=1):
- `max_row_base = 2 * 1 = 2`
- `global_col_base = 1 * 8 = 8`
- `in0_tile_index = (2 + 0) * 16 + 8 + j` for j=0..7 → tiles [40..47] ✓ (row 2, second half)
- `max_tile = 2 * 1 + 0 = 2` → reads max tile for row 2 ✓

**Pack back in-place to cb_qkt_im:**
```cpp
pack_tile<true>(dst_index++, inout_cb, in0_tile_index)  // same absolute position
```
Writes back to the same position it was read from. ✓

**Reduce L1 accum to reduce_cb (cur_sum):**
```cpp
pack_tile<true>(dst_index++, reduce_cb, max_row_base + i)
```
For i=0: `max_row_base + 0 = 2` → writes at absolute offset 2 in cur_sum. ✓

**Verdict: CORRECT**

### 7.3 QKT@V Matmul Indexing

```cpp
qktv_in0_index_offset advances by qktv_subblock_h * qktv_in0_block_w = 1 * 16 = 16 per q_subblock
```

For q_subblock=0: reads cb_qkt_im tiles [0..15] (first row of softmax'd data)
For q_subblock=1: reads [16..31], etc.

**V matmul output with pack_tile<true>:**
```cpp
out_row_offset = (r + q_subblock * SUBBLOCK_H) * OUT_NUM_COLS
              = (0 + w_q * 1) * 4 = w_q * 4
```
Where `w_q = q_subblock - pushed_rows`.

On non-last iterations: `pushed_rows = 0`, so `w_q = q_subblock`.
For q_subblock=3: `out_row_offset = 3 * 4 = 12`, writes at [12..15]. ✓

On last iteration: `pushed_rows` increments as rows are normalized and pushed. The `w_salad` and `w_q` adjust offsets relative to the advancing wr_ptr.

**Verdict: CORRECT** — the pushed_rows adjustment correctly accounts for wr_ptr advancing during per-row normalization.

### 7.4 SALAD Correction Indexing

```cpp
mul_bcast_cols_l1_acc<sbh>(prev_sum, cb_exp_max_diff, cur_sum, salad_row, w_salad)
```

Reads: `prev_sum[(salad_row+1)*1]` tiles cumulatively, `cb_exp_max_diff[(salad_row+1)*1]` tiles cumulatively.
Writes: L1 accumulate at `cur_sum[w_salad]`.

```cpp
mul_block_bcast_cols_acc<sbh, head_dim_t>(prev_out, cb_exp_max_diff, cur_out, salad_row, w_salad)
```

Reads: `prev_out[(salad_row+1) * 1 * 4]` tiles from prev_out, `cb_exp_max_diff[(salad_row+1)*1]` from exp_max_diff.
Writes: L1 accumulate at `cur_out[w_salad * 4 + j]` for j=0..3.

**Verdict: CORRECT**

### 7.5 Potential Issue: DST Capacity

With sbh=1:
- Q@KT: sbh × qkt_subblock_w = 1 × 8 = 8 tiles → **at limit** (8 tiles max with fp16b double-buffer)
- sub_exp_block_bcast_cols: sbh × qkt_subblock_w = 1 × 8 = 8 tiles → **at limit**
- QKT@V: qktv_subblock_h × qktv_subblock_w = 1 × 4 = 4 tiles → OK
- mul_bcast_cols_l1_acc: sbh = 1 tile → OK
- mul_block_bcast_cols_acc: sbh × head_dim_t = 1 × 4 = 4 tiles → OK
- normalize: head_dim_t = 4 tiles → OK

**Verdict: All within 8-tile DST limit for fp16b double-buffer mode.**

### 7.6 Potential Issue: cb_wait_front Cumulative Counts

The kernel uses cumulative `cb_wait_front` calls extensively. Checking:

- `cb_wait_front(cb_q_in, q_wait_tiles)` where `q_wait_tiles` starts at `q_subblock_num_tiles` (=4) and grows by 4 each iteration: 4, 8, 12, ..., 28. Total Q tiles = 28. CB capacity = 56 (double-buffered). ✓
- `cb_wait_front(cb_kt_in, head_dim_t * Sk_chunk_t)` = 64. CB capacity = 128. ✓
- `cb_wait_front(cb_qkt_im, qktv_in0_wait_tiles)` grows from 16 to 112. CB capacity = 112. ✓ (at limit on last subblock)

**Verdict: CORRECT — no overruns.**

---

## 8. Comparison with Reference SDPA (compute_common.hpp) <a id="8-comparison-with-reference"></a>

### 8.1 Structural Differences

| Feature | Reference SDPA | This Implementation |
|---------|---------------|---------------------|
| **Q@KT matmul** | Single `matmul_blocks()` call, full Q@KT at once | Row-by-row directly into cb_qkt_im via `pack_tile<true>` |
| **Softmax** | `sub_exp_block_bcast_cols_inplace()` — in-place on cb_qkt_im | `sub_exp_block_bcast_cols()` — in-place on cb_qkt_im (reads/writes same absolute positions) |
| **Max reduce** | `reduce_c()` on full cb_qkt_im | `reduce_c_row_group()` per row, interleaved with matmul |
| **SALAD correction** | Separate steps: `sub_exp_block`, `mul_tiles_bcast_cols_inplace`, `add_block_inplace`, `mul_block_bcast_cols` | Fused: `sub_exp_first_col_blocks` + `mul_bcast_cols_l1_acc` + `mul_block_bcast_cols_acc` |
| **QKT@V matmul** | Single `matmul_blocks()` call | Subblock-by-subblock with overlap |
| **Final normalization** | `matmul_reduce` + `recip_block_inplace` + `mul_block_bcast_cols` (all in-place) | `normalize_row_streaming()` — per-row streaming with fused matmul+recip |
| **Row buffering** | Full cb_qkt_im written at once (no row buffers needed) | `cb_push_back_hold_wr_ptr` keeps wr_ptr stable (no row buffers needed) |
| **Causal masking** | Full support via `add_block_inplace(qk, mask)` | **NOT IMPLEMENTED** |
| **Attention sink** | Full support | **NOT IMPLEMENTED** |
| **Ring attention** | Full support (RING type) | **NOT IMPLEMENTED** |
| **Provided mask** | Full support | **NOT IMPLEMENTED** |
| **K-chunk transposition** | Done by compute kernel (matmul with `transpose=true`) | Done by reader kernel (tile grid transpose during DMA); matmul still uses `transpose=true` for within-tile transpose |

### 8.2 Algorithmic Differences

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
   - Pipelined: matmul row N (writing to cb_qkt_im at absolute offsets) while softmax'ing row N-1 (in-place on cb_qkt_im)
   - `cb_push_back_hold_wr_ptr` exposes each row to UNPACK without advancing PACK's wr_ptr — all absolute offsets stay valid
   - Max reduce per row immediately after push, reading from cb_qkt_im at the row's position
   - First K-chunk: `do_eltwise_max=false` — no prev_max comparison (same as reference's `processed_k_chunks > 0`)
2. Phase 2: Row-by-row QKT@V with interleaved SALAD
   - Pipelined: V matmul for row N while SALAD-correcting row N-1
   - Overlap drain loop: last row's softmax done in-place on cb_qkt_im (all rows already pushed), interleaved with first V matmul
   - First K-chunk: exp_max_diff and SALAD entirely skipped (same as reference's `if (processed_k_chunks > 0)`)
3. On last K-chunk: per-row normalization streamed directly to output (decoupled from SALAD, runs even when first == last)

This approach is significantly more pipelined than the reference, overlapping FPU (matmul) with SFPU (exp) and minimizing idle time. The `cb_push_back_hold_wr_ptr` trick eliminates the need for separate row buffers, saving 64 KB of L1.

### 8.3 Correctness Gap: The correction_block / fused_max_sub_exp_add_tile

The reference SDPA has a `correction_block()` function that uses `fused_max_sub_exp_add_tile` — a custom SFPI kernel that simultaneously:
1. Computes `cur_max = max(prev_max, worker_max)`
2. Computes `exp(prev_max - cur_max)` and `exp(worker_max - cur_max)`
3. Computes `cur_sum = exp(worker_max - cur_max) * worker_sum + exp(prev_max - cur_max) * prev_sum`

This is specifically used in **ring attention** (RING type) for combining results from different workers. The standard SDPA path does NOT use this — it does the same operations decomposed into separate steps as this implementation does.

---

## 9. Missing Features for Numerical Correctness <a id="9-missing-features"></a>

### 9.1 Causal Masking (HIGH PRIORITY)

The reference SDPA adds a causal mask to the attention scores before softmax:
```cpp
if (apply_mask) {
    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
}
```

This is a critical correctness feature for decoder-only models. The mask is a triangular matrix of -inf values that prevents attending to future tokens.

**Impact:** Without causal masking, the kernel computes bidirectional attention (correct for encoder models, incorrect for decoder models like GPT/LLaMA).

**Implementation plan:**
1. Add a mask CB (reader loads mask tiles from DRAM or generates them)
2. After Q@KT matmul output but before max_reduce, add mask tiles to the row buffer
3. For the row-by-row approach: could generate causal mask on-the-fly by computing which tiles need -inf based on q_subblock and kt_subblock indices

### 9.2 Sliding Window Attention

The reference supports `sliding_window_size > 0` for local attention patterns. Not implemented here.

### 9.3 Attention Sink

The reference supports attention sink logits (additional softmax denominator contribution without corresponding V output). Not implemented here.

### 9.4 Provided Mask Support

The reference supports user-provided masks (not just causal). Not implemented here.

### 9.5 Padded Mask Support (DONE)

Tile-aligned padded K masking is implemented via `apply_padded_mask()`. When `padded_k_tiles > 0`, the writer generates a single -inf tile in `cb_mask_in` (c7). On the last K chunk, the compute kernel L1-accumulates -inf onto the padded tile positions in the row buffer before push and max_reduce. This uses `pack_tile<true>` with L1 acc while tiles are still in reserved state — no row buffer doubling needed. The -inf tile stays fronted (never popped) for reuse across all Q subblocks.

### 9.6 Multi-Head / Multi-Batch

The current implementation processes a single head on a single core. The reference SDPA distributes heads across cores. Extension to multi-head is an architectural decision (multi-core parallelism).

### 9.7 Scale as Separate Operation vs. Fused

Both implementations fuse the scale into exp. This is correct.

### 9.8 K/V Layout: Reader Transpose vs. Compute Transpose

The current implementation transposes K tiles at the reader level (tile grid transpose during DMA), then also uses `transpose=true` in the matmul call. The reference uses `transpose=true` in the matmul call only.

**Note:** The reader's tile grid transpose changes `[Sk_chunk_t × head_dim_t]` → `[head_dim_t × Sk_chunk_t]`, which means each tile's position in the grid is transposed. The matmul `transpose=true` then transposes within each 32×32 tile. Together, this achieves the full K^T operation.

The reference does NOT do tile grid transpose at the reader — it relies solely on the matmul's `transpose=true`. This works because the reference uses different indexing in `matmul_blocks()` with `in1_index += N` stride.

**Both approaches are mathematically equivalent.** The current approach may be slightly more efficient because the reader can overlap the transpose DMA with compute work on the previous chunk.

---

## 10. Suggested Next Steps <a id="10-next-steps"></a>

### Step 1: Validate Numerical Correctness (DONE)

Bidirectional attention is numerically correct. Test results against PyTorch `F.scaled_dot_product_attention(Q, K, V, is_causal=False)`:

| Test | PCC | Max Abs Error | RMSE |
|------|-----|--------------|------|
| `1q_1k-zeros-sk16` | 1.000000 | 0.000000 | 0.000000 |
| `1q_1k-ones-sk16` | 1.000000 | 0.007812 | 0.007812 |
| `1q_1k-random-sk16` | 0.999805 | 0.039930 | 0.003270 |
| `1q_5k-random-sk16` | 0.999853 | 0.081168 | 0.003782 |
| `3q_5k-random-sk16` | 0.999851 | 0.493403 | 0.003575 |
| `1q_1k-random-sk8` | 0.999714 | 0.103503 | 0.004356 |
| `1q_5k-random-sk8` | 0.999865 | 0.094849 | 0.003997 |
| `3q_5k-random-sk8` | 0.999847 | 0.072169 | 0.002991 |
| `1q_5k-random-sk16-pad4` | 0.999855 | 0.159379 | 0.003332 |
| `1q_5k-random-sk8-pad2` | 0.999799 | 0.081001 | 0.003905 |
| `3q_5k-random-sk16-pad8` | 0.999752 | 0.518033 | 0.004621 |
| `1q_1k-random-sk4-sbh2` | 0.999819 | 0.100782 | 0.006158 |
| `1q_5k-random-sk4-sbh2` | 0.999821 | 0.083144 | 0.003283 |
| `1q_5k-random-sk8-sbh2` | 0.999862 | 0.068657 | 0.004281 |
| `3q_5k-random-sk4-sbh2-pad1` | 0.999763 | 0.102592 | 0.003703 |

The `1q_1k` tests exercise `is_first && is_last` (single K chunk — no SALAD). The multi-K tests exercise the full ping-pong loop with SALAD corrections. Tests run with Sk_chunk_t=16/8 (sbh=1) and Sk_chunk_t=4/8 (sbh=2). Padded tests (`-padN`) verify that zero-padded K tiles are correctly masked out via `apply_padded_mask`. Using `EXP_APPROX_MODE=0` (polynomial exp, degree 2).

### Step 2: Add Causal Masking

This is the single most important feature for practical SDPA:

1. **Option A (simple):** Have the reader generate causal mask tiles and load them into a new mask CB. Compute kernel adds mask to Q@KT scores before max_reduce.
2. **Option B (efficient):** Skip K-chunks entirely where all scores would be masked (i.e., where `k_chunk_start > q_chunk_end`). This is what the reference does via `k_chunk_end = (q_high_idx + Sk_chunk_t - 1) / Sk_chunk_t`.
3. For mixed chunks (partially masked), generate the mask tile on-the-fly or load from DRAM.

### Step 3: Multi-Head Extension

Wrap the current single-core kernel in a multi-core dispatch that assigns different heads to different cores. This is the standard SDPA parallelization strategy.

### Step 4: Performance Optimizations

1. **Double-buffering K/V:** The reader already double-buffers, but the compute kernel processes one K-chunk at a time. Consider prefetching the next K-chunk while computing on the current one.
2. **Granularity tuning:** The reference uses `SUB_EXP_GRANULARITY`, `DHT_GRANULARITY`, `STATS_GRANULARITY`, `REDUCE_GRANULARITY` for tile-processing granularity. Adding similar configurability here could help tune for different problem sizes.
3. **In-place operations:** The current implementation uses `cb_push_back_hold_wr_ptr` for in-place sub_exp on cb_qkt_im, eliminating the need for separate row buffers. Further in-place opportunities exist for SALAD corrections.

### Step 5: Ring Attention (Future)

For multi-device scenarios, implement the ring attention pattern from the reference (correction_block, sigmoid, logsigmoid operations).

---

## Appendix A: Helper Function Reference

| Function | Purpose |
|----------|---------|
| `cb_push_back_hold_wr_ptr` | Push tiles (UNPACK sees them) but rewind PACK's wr_ptr, keeping pack_tile\<true\> offsets stable |
| `sub_exp_block_bcast_cols` | In-place on inout_cb: subtract max (bcast cols), apply exp, pack back + L1 accum reduce |
| `sub_exp_first_col_blocks` | Subtract prev_max from cur_max, apply exp (column 0 only) → correction factor |
| `mul_bcast_cols_l1_acc` | Multiply + bcast_cols, L1 accumulate into output |
| `mul_block_bcast_cols_acc` | Block multiply + bcast_cols, L1 accumulate (for output rescaling) |
| `blocked_matmul_and_pack` | Blocked matmul with pack to CB via pack_tile\<true\> at absolute offsets |
| `reduce_c_row_group` | Max reduce across rows using reduce_block_max_row + eltwise_max |
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
