# Plan: Integrate Streaming SDPA Compute Kernel for Non-Causal Cases

## Context

The programming example at commit `54ba9951c42ae7548d350b8cb12703a6877116b1` implements a fundamentally different SDPA compute loop that pipelines operations row-by-row instead of processing full blocks monolithically. The key speedup mechanisms are:

1. **Alternating row buffers** for Q@KT: sub_exp of previous row overlaps with matmul of current row (hides SFPU latency behind FPU)
2. **SALAD correction overlap**: previous row's output/sum rescaling overlaps with current row's SV matmul
3. **Streaming normalization**: output tiles pushed per-row on the last K iteration, avoiding the separate `matmul_reduce` + `recip_block_inplace` + `mul_block_bcast_cols` pass

This plan integrates the new compute path for **non-causal SDPA only** (no causal mask, no attention_sink, no sliding window), gated by a compile-time define so the existing path is untouched.

---

## Architecture Comparison

### Current Production (`sdpa_inner_loop` in `compute_common.hpp`)

For each Q chunk, for each K chunk:
```
1. QK = matmul_blocks(Q, K^T)               ← full Sq_chunk_t × Sk_chunk_t block at once
2. QK += mask (if applicable)
3. cur_max = reduce_c<MAX>(QK)               ← full block reduce
4. QK = exp((QK - cur_max) * scale)          ← sub_exp_block_bcast_cols_inplace, full block
5. OUT = matmul_blocks(QK, V)                ← full Sq_chunk_t × vDHt block at once
6. correction: rescale prev_out, prev_sum by exp(prev_max - cur_max)
After all K chunks:
7. sum = matmul_reduce(sum, col_identity)    ← final row reduction
8. sum = 1/sum                               ← recip_block_inplace
9. out = out * (1/sum)                       ← mul_block_bcast_cols, final normalize
```

**Bottleneck**: Operations are strictly sequential — FPU and SFPU never overlap within a K iteration.

### New Streaming Kernel (`sdpa_inner_loop_step` from example)

For each Q chunk, for each K chunk:
```
PHASE 1 — Q@KT with alternating row buffers:
  For each Q subblock row (sbh tiles tall):
    a. matmul QK row → row_buffer_cur           ← FPU busy
    b. sub_exp(row_buffer_prev) → qk_im         ← SFPU busy (overlaps with next row's matmul)
    c. reduce_max(row_buffer_cur) → cur_max      ← per-row, not full block
    d. swap(row_buffer_cur, row_buffer_prev)

PHASE 2 — QKT@V + SALAD corrections:
  Drain last row: sub_exp(row_buffer_prev) → qk_im
  For each Q subblock row:
    a. exp(prev_max - cur_max) → exp_max_diff    ← SFPU for SALAD
    b. matmul QK_row @ V → cur_out               ← FPU (overlaps with SALAD above)
    c. SALAD: cur_sum += prev_sum * exp_max_diff  ← L1 accumulation
    d. SALAD: cur_out += prev_out * exp_max_diff  ← L1 accumulation
    e. IF last K iter: normalize_row_streaming    ← fused recip+bcast per row
```

**Key advantage**: FPU and SFPU overlap in both phases. Normalization is streaming (no separate pass).

---

## Files to Modify

| File | Change |
|------|--------|
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp` | Add ~8 new helper functions + `sdpa_inner_loop_step` + `sdpa_standard_v2` wrapper |
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp` | Gate between `sdpa_standard` and `sdpa_standard_v2` via `#ifdef USE_STREAMING_COMPUTE` |
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp` | Add 3 new CBs, pass `qk_subblock_h` compile-time arg, set `USE_STREAMING_COMPUTE` define for non-causal |

Reader (`reader_interleaved.cpp`) and writer (`writer_interleaved.cpp`) are **unaffected** — they push Q/K/V into c_0/c_1/c_2 and read output from c_16, same as before.

---

## CB Mapping (Production indices preserved)

```
Existing (unchanged):
  c_0  = Q input           c_5  = identity_scale
  c_1  = K input           c_7  = col_identity  (reused for normalize)
  c_2  = V input           c_16 = final output  (= normalized_out target)
  c_24 = QK intermediate   c_25/c_26 = output im A/B
  c_27/c_28 = max A/B      c_29/c_30 = sum A/B
  c_31 = exp_max_diff

New (streaming path only):
  c_4  = QK row buffer A   (reuses attention_sink slot; non-causal doesn't use it)
  c_6  = QK row buffer B
  c_10 = recip scratch     (1 tile)
```

---

## Implementation Steps

### Step 1: Add New Helper Functions to `compute_common.hpp`

Add the following functions (adapted from the example, remapped to production CB indices):

#### 1.1 `blocked_matmul_and_pack`
```cpp
template <bool transpose, uint32_t sbw, uint32_t sbh, uint32_t in0_block_w,
          uint32_t in1_N, uint32_t out_N, bool init_short = true>
void blocked_matmul_and_pack(
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb,
    uint32_t in0_index_offset, uint32_t in1_index_offset,
    uint32_t out_row_group, uint32_t out_col_offset);
```
Subblock-granular matmul that packs with explicit row/col offset. Used for both QK (transpose=true) and SV (transpose=false). Key difference from `matmul_blocks`: packs one subblock at a time with `pack_tile<true>(dst, out_cb, row*out_N + col)`.

#### 1.2 `reduce_c_row_group`
```cpp
template <PoolType pool_type, ReduceDim reduce_dim, uint32_t scale_cb,
          uint32_t cols, uint32_t sbh>
void reduce_c_row_group(
    uint32_t in_cb, uint32_t out_cb, uint32_t prev_cb,
    uint32_t row_group_idx, bool do_eltwise_max, uint32_t in0_row_group_index);
```
Per-row-group max reduction from row buffer. Reads `sbh` rows, reduces, optionally eltwise_max with prev, packs at explicit offset into `out_cb`.

#### 1.3 `sub_exp_block_bcast_cols` (NOT in-place)
```cpp
template <bool PROFILING, uint32_t scale_fp32, uint32_t sbh,
          uint32_t subblock_w, bool write_to_qk_im>
void sub_exp_block_bcast_cols(
    uint32_t row_buf_cb, uint32_t max_cb, uint32_t qk_im_cb,
    uint32_t sum_cb, uint32_t Sk_chunk_t,
    uint32_t q_subblock_idx, uint32_t kt_subblock_idx);
```
Reads raw QK tiles from row buffer, subtracts cur_max, applies exp with ReLU clamping, writes softmax'd tiles to `cb_qk_im` at explicit offset. Accumulates partial row sums into `cur_sum` via L1 accumulation.

#### 1.4 `sub_exp_first_col_blocks`
```cpp
template <bool PROFILING, uint32_t scale_fp32, uint32_t sbh>
void sub_exp_first_col_blocks(
    uint32_t prev_max_cb, uint32_t cur_max_cb,
    uint32_t out_cb, uint32_t row_group_idx);
```
Computes `exp((prev_max - cur_max) * scale)` for SALAD corrections. Column-only (VectorMode::C).

#### 1.5 `mul_bcast_cols_l1_acc`
```cpp
template <uint32_t sbh>
void mul_bcast_cols_l1_acc(
    uint32_t prev_sum_cb, uint32_t exp_diff_cb, uint32_t cur_sum_cb,
    uint32_t row_group_idx, uint32_t write_offset);
```
SALAD sum correction: `cur_sum += prev_sum * exp_max_diff`. Uses L1 accumulation at explicit tile offset.

#### 1.6 `mul_block_bcast_cols_acc`
```cpp
template <uint32_t sbh, uint32_t head_dim_t>
void mul_block_bcast_cols_acc(
    uint32_t prev_out_cb, uint32_t exp_diff_cb, uint32_t cur_out_cb,
    uint32_t row_group_idx, uint32_t write_row_offset);
```
SALAD output correction: `cur_out += prev_out * exp_max_diff`. L1 accumulation at explicit row offset.

#### 1.7 `normalize_row_streaming`
```cpp
template <bool PROFILING, uint32_t sbh, uint32_t head_dim_t>
void normalize_row_streaming(
    uint32_t cur_sum_cb, uint32_t cur_out_cb, uint32_t col_identity_cb,
    uint32_t scratch_cb, uint32_t normalized_out_cb);
```
Per-row normalization pipeline:
- `matmul_reduce`: sum × col_identity → scratch (collapse partial sums to column 0)
- `recip` directly in DST (avoids pack/unpack roundtrip via `recip_tile_first_column`)
- `mul_bcast_cols`: output × (1/sum) → normalized output
- Consumes (pops) sum and output tiles, pushes to `cb_normalized_out` (= c_16)

**Note**: Requires `head_dim_t <= 8` (DST capacity). For WAN shapes with head_dim=128, head_dim_t=4 — OK.

---

### Step 2: Add `sdpa_inner_loop_step` to `compute_common.hpp`

This is the core ~350-line function implementing the two-phase algorithm. Template signature:
```cpp
template <bool PROFILING_ENABLED,
          uint32_t Sq_chunk_t, uint32_t Sk_chunk_t, uint32_t Sv_chunk_t,
          uint32_t head_dim_t,
          uint32_t cb_q_in, uint32_t cb_kt_in, uint32_t cb_v_in,
          uint32_t cb_qkt_im, uint32_t cb_identity_scale_in,
          uint32_t cb_exp_max_diff, uint32_t scale_fp32,
          uint32_t subblock_h,
          uint32_t cb_qkt_row_A, uint32_t cb_qkt_row_B,
          uint32_t cb_col_identity, uint32_t cb_recip_scratch,
          uint32_t cb_normalized_out>
void sdpa_inner_loop_step(
    const uint32_t prev_max, const uint32_t cur_max,
    const uint32_t prev_sum, const uint32_t cur_sum,
    const uint32_t prev_out, const uint32_t cur_out,
    const bool is_last_iter, const bool is_first_iter);
```

**Derived compile-time constants** (computed inside):
```cpp
constexpr uint32_t sbh = subblock_h;
constexpr uint32_t in0_block_w = head_dim_t;
constexpr uint32_t qkt_subblock_w = 8 / sbh;           // DST tiles for QK subblock width
constexpr uint32_t q_num_subblocks = Sq_chunk_t / sbh;  // How many row groups per Q chunk
constexpr uint32_t kt_num_subblocks = Sk_chunk_t / qkt_subblock_w;
constexpr uint32_t row_tiles = sbh * Sk_chunk_t;        // Tiles per row in QK
```

**Phase 1 pseudocode:**
```
alias_cur_qkt_row = cb_qkt_row_A
alias_prev_qkt_row = cb_qkt_row_B
cb_reserve_back(cb_qkt_im, Sq_chunk_t * Sk_chunk_t)
cb_wait_front(cb_kt_in, head_dim_t * Sk_chunk_t)
cb_reserve_back(cur_sum, Sq_chunk_t)

for q_subblock = 0..q_num_subblocks-1:
    cb_wait_front(cb_q_in, ...)
    cb_reserve_back(alias_cur_qkt_row, row_tiles)

    for kt_subblock = 0..kt_num_subblocks-1:
        // OVERLAP: sub_exp previous row while matmul current row
        if q_subblock > 0:
            sub_exp_block_bcast_cols(prev_row → qk_im, cur_max, cur_sum)
        blocked_matmul_and_pack<transpose=true>(Q, K → cur_row)

    if q_subblock > 0:
        push softmax'd prev row to qk_im, pop prev row buffer

    push raw matmul row
    reduce_max(cur_row → cur_max, with eltwise_max if !first_iter)
    swap(cur_row, prev_row)

pop K
```

**Phase 2 pseudocode:**
```
drain: sub_exp(last_prev_row → qk_im)
push last softmax'd row, pop last row buffer

cb_wait_front(cb_v_in, ...)
cb_reserve_back(cur_out, Sq_chunk_t * head_dim_t)

for q_subblock = 0..qktv_q_num_subblocks-1:
    cb_wait_front(cb_qkt_im, ...)

    if q_subblock > 0 && !first_iter:
        sub_exp_first_col_blocks(prev_max, cur_max → exp_max_diff)

    // SV matmul for current row
    blocked_matmul_and_pack<transpose=false>(QK_row, V → cur_out)

    if q_subblock > 0 && !first_iter:
        // SALAD: correct previous row's sum and output
        mul_bcast_cols_l1_acc(prev_sum * exp_max_diff → cur_sum)
        mul_block_bcast_cols_acc(prev_out * exp_max_diff → cur_out)
        if is_last_iter: normalize_row_streaming(sum, out → normalized_out)
    elif q_subblock > 0 && is_last_iter:
        normalize_row_streaming(sum, out → normalized_out)

// Pipeline drain: SALAD + normalize for last row
if !first_iter: SALAD correct last row
if is_last_iter: normalize last row

// Bulk push (skip on last iter — consumed by normalize)
if !is_last_iter: push cur_sum, cur_out

pop V, pop qk_im
```

---

### Step 3: Add `sdpa_standard_v2` Wrapper

Wraps `sdpa_inner_loop_step` in the same outer Q-chunk / K-chunk loop with ping-pong buffer swapping:

```cpp
template <uint32_t cb_qk_im, uint32_t cb_identity_scale_in,
          uint32_t Sq_chunk_t, uint32_t Sk_chunk_t, uint32_t DHt,
          uint32_t scale_fp32, uint32_t subblock_h,
          uint32_t cb_qkt_row_A, uint32_t cb_qkt_row_B,
          uint32_t cb_col_identity, uint32_t cb_recip_scratch>
void sdpa_standard_v2(
    const uint32_t k_num_chunks,
    const uint32_t iter_q_start, const uint32_t iter_q_end,
    const uint32_t local_q_start,
    /* CB handles */ cb_q_in, cb_k_in, cb_v_in,
    cb_out_im_A, cb_out_im_B, cb_max_A, cb_max_B,
    cb_sum_A, cb_sum_B, cb_exp_max_diff, cb_out);
```

Outer loop:
```
for q_iter = iter_q_start..iter_q_end:
    ping-pong: prev_sum/cur_sum, prev_max/cur_max, prev_out/cur_out
    for k_chunk = 0..k_num_chunks:
        sdpa_inner_loop_step<...>(prev/cur buffers, is_first, is_last)
        if !first: pop prev buffers
        if last: pop cur_max
        else: swap prev/cur
    pop Q
```

---

### Step 4: Gate in `sdpa.cpp`

In the compute kernel entry point, add after existing compile-time arg parsing:

```cpp
#ifdef USE_STREAMING_COMPUTE
    constexpr uint32_t subblock_h = get_compile_time_arg_val(30);  // New arg index
    // ... call sdpa_standard_v2 with appropriate CB indices
#else
    // ... existing sdpa_standard call (unchanged)
#endif
```

The new path remaps example CB indices to production:
- `cb_kt_in` = c_1 (same as `cb_k_in`)
- `cb_v_in` = c_2
- `cb_qkt_im` = c_24
- `cb_qkt_row_A` = c_4
- `cb_qkt_row_B` = c_6
- `cb_col_identity` = c_7
- `cb_recip_scratch` = c_10
- `cb_normalized_out` = c_16

---

### Step 5: Modify `sdpa_program_factory.cpp`

**Gating condition** (non-causal AND no padding needed):
```cpp
bool use_streaming_compute = !is_causal && !use_provided_mask &&
    !use_attention_sink && sliding_window_size == 0 && !use_padded_mask;
```

When `use_streaming_compute`:

1. **Set define**: `defines["USE_STREAMING_COMPUTE"] = "1"`

2. **Compute subblock_h**: Use the QK subblock height from `determine_largest_subblock_size(Sq_chunk_t, Sk_chunk_t, dst_size)`.

3. **Add compile-time arg**: Append `subblock_h` to compute kernel compile-time args.

4. **Allocate 3 new CBs**:
   ```cpp
   // c_4 = QK row buffer A (reuses attention_sink slot)
   CircularBufferConfig cb_qkt_row_A_config =
       CircularBufferConfig(qk_subblock_h * Sk_chunk_t * tile_size, {{CBIndex::c_4, cb_data_format}})
           .set_page_size(CBIndex::c_4, tile_size);

   // c_6 = QK row buffer B
   CircularBufferConfig cb_qkt_row_B_config =
       CircularBufferConfig(qk_subblock_h * Sk_chunk_t * tile_size, {{CBIndex::c_6, cb_data_format}})
           .set_page_size(CBIndex::c_6, tile_size);

   // c_10 = recip scratch (1 tile)
   CircularBufferConfig cb_recip_scratch_config =
       CircularBufferConfig(tile_size, {{CBIndex::c_10, cb_data_format}})
           .set_page_size(CBIndex::c_10, tile_size);
   ```

5. **c_16 already allocated** as final output CB — serves as `cb_normalized_out`.

**Important constraints** (validated at host):
- `Sq_chunk_t % subblock_h == 0`
- `subblock_h * (8 / subblock_h) <= 8` (DST capacity)
- `Sk_chunk_t % (8 / subblock_h) == 0`

---

### Step 6: Padded Mask Fallback

When `use_padded_mask=true` (K tile count not a multiple of `Sk_chunk_t`), fall back to the existing `sdpa_standard` path. The streaming path gating condition in Step 5 already excludes this case.

**Test coverage analysis** — which `test_sdpa_accuracy` cases hit the new path:

| Shape | k_chunk | Skt (tiles) | Sk_chunk_t | k_num_chunks | Padded? | Path |
|-------|---------|-------------|------------|--------------|---------|------|
| 9472  | 128     | 296         | 4          | 74           | No      | **v2** |
| 9472  | 256     | 296         | 8          | 37           | No      | **v2** |
| 9472  | 512     | 296         | 16         | 19 (pad=304) | Yes     | v1   |
| 2368  | 128     | 74          | 4          | 19 (pad=76)  | Yes     | v1   |
| 2368  | 256     | 74          | 8          | 10 (pad=80)  | Yes     | v1   |
| 2368  | 512     | 74          | 16         | 5 (pad=80)   | Yes     | v1   |

With 3 Q chunk sizes each: **6 out of 18** accuracy tests exercise the new streaming path (shape 9472 × k128/k256). The remaining 12 fall back to the existing path unchanged. Padded mask support in v2 can be added as a follow-up to get full coverage.

---

## Verification

```bash
# Build
./build_metal.sh --release

# Clear JIT cache
rm -rf /localdev/pjosipovic/tt-metal/built/*

# Run the target tests
PYTHONPATH="/localdev/pjosipovic/tt-metal/tools:/localdev/pjosipovic/tt-metal:/localdev/pjosipovic/tt-metal/ttnn" \
  pytest tests/tt_eager/python_api_testing/unit_testing/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy -v

# Verify determinism
PYTHONPATH="/localdev/pjosipovic/tt-metal/tools:/localdev/pjosipovic/tt-metal:/localdev/pjosipovic/tt-metal/ttnn" \
  pytest tests/tt_eager/python_api_testing/unit_testing/test_scaled_dot_product_attention_sprint.py::test_sdpa_determinism -v
```

Expected: PCC >= 0.9997, RMSE <= 4e-2 for all shape/chunk combinations.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| CB protocol deadlock in new compute path | Carefully trace CB push/pop/wait/reserve for every path through the two phases. Add DPRINT traces for debugging. |
| L1 memory overflow from additional row buffer CBs | Row buffers are `sbh * Sk_chunk_t` tiles each (small). Verify total L1 fits on BH. |
| DST register pressure in normalize_row_streaming | head_dim_t <= 8 required (128/32=4 for WAN shapes — OK). |
| Accuracy regression from different operation ordering | Same FlashAttention algorithm, different scheduling. Should be bitwise-similar but verify PCC. |
| Fallback path not exercised | Keep existing `sdpa_standard` for causal/masked/chunked cases — those tests remain on old path. |
