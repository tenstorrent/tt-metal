# Plan: Integrate Streaming SDPA v2 — Row Buffer Elimination + sbh=2

## Context

Four new commits on `origin/dnijemcevic/sdpa_benchmark` (since `12dfcf4de8a`, our v1 integration base):

| Commit | Summary |
|--------|---------|
| `a17ef4aec14` | Support subblock_h=2 to enable Sk_chunk_t=4 |
| `f3900572417` | Eliminate row buffers c_4/c_6 via `cb_push_back_hold_wr_ptr` |
| `63ee91e1c9b` | Docs only (README/analysis update) |
| `e29a40dc146` | Minor instrumentation fix (`SUB_EXP_BLOCK_INIT` zone) |

These changes **remove the need for c_4/c_6 row buffer CBs** (saving 64 KB L1) and **fix the subblock_h=2 correctness bug** that forced our v1 to constrain `qk_out_subblock_h == 1`.

## What This Fixes in v1

| v1 Limitation | Root Cause | v2 Fix |
|---------------|-----------|--------|
| `qk_out_subblock_h == 1` constraint | Three sbh=2 bugs (pack layout, reduce L1 acc init, QKT matmul output) | Commit `a17ef4aec14` fixes all three |
| L1 overflow for Sk_chunk_t=16 (k_chunk=512) | Two 16-tile row buffers at 2048 B/tile = 64 KB | Row buffers eliminated entirely via `cb_push_back_hold_wr_ptr` |
| Only k_chunk=256 hits streaming path | Above two constraints combined | All k_chunk values can use streaming |
| `mul_block_bcast_cols_acc<sbh, vDHt>` DST overflow with sbh>1 | sbh=3 × vDHt=4 = 12 > 8 DST tiles | sbh=2 × vDHt=4 = 8 ≤ 8 (sbh=2 works; sbh=3 still excluded but rare) |

## Key Architectural Change: `cb_push_back_hold_wr_ptr`

The v1 design used two alternating row buffers (c_4, c_6) because `pack_tile<true>` computes addresses relative to `wr_ptr`, and `cb_push_back` advances `wr_ptr` — causing address drift on subsequent writes.

The new `cb_push_back_hold_wr_ptr` utility pushes tiles (making them visible to UNPACK) while **rewinding `wr_ptr` back**, keeping all `pack_tile<true>` offsets relative to a stable base. This allows:
- Q@KT matmul output writes **directly into `cb_qkt_im`** (no intermediate row buffers)
- In-place `sub_exp` on `cb_qkt_im` (was read_cb → write_cb, now single `inout_cb`)
- Incremental row exposure via held push

```cpp
ALWI void cb_push_back_hold_wr_ptr(uint32_t cb_id, uint32_t num_tiles) {
    cb_push_back(cb_id, num_tiles);
    PACK(({
        auto& intf = get_local_cb_interface(cb_id);
        intf.fifo_wr_ptr -= num_tiles * intf.fifo_page_size;
        uint32_t fifo_start = intf.fifo_limit - intf.fifo_size;
        if (intf.fifo_wr_ptr < fifo_start) {
            intf.fifo_wr_ptr += intf.fifo_size;
        }
    }));
}
```

## Files to Modify

| File | Changes |
|------|---------|
| `compute_common.hpp` | Update 4 helper functions, update `sdpa_inner_loop_step`, update `sdpa_standard_v2` |
| `sdpa.cpp` | Remove c_4/c_6 CB declarations, remove `cb_recip_scratch` alias, simplify `sdpa_standard_v2` template args |
| `sdpa_program_factory.cpp` | Remove c_4/c_6 CB allocation, remove L1 budget check, relax `qk_out_subblock_h == 1` to `qk_out_subblock_h <= 2` |

Reader/writer kernels remain **unchanged**.

## Implementation Steps

### Step 1: Add `cb_push_back_hold_wr_ptr` to `compute_common.hpp`

Add the utility function near the top of the streaming helpers section. It's a simple ALWI function that calls `cb_push_back` then rewinds `wr_ptr` on the PACK thread.

### Step 2: Update `sub_exp_block_bcast_cols`

Change from 3-CB (read_cb, write_cb, reduce_cb) to 2-CB in-place (inout_cb, reduce_cb):
- Remove `write_cb` parameter, replace `read_cb` with `inout_cb`
- Read positions: `(max_row_base + i) * cols_in_row + global_col_base + j` (absolute in cb_qkt_im)
- Write positions: same absolute positions (in-place overwrite via `pack_tile<true>`)
- Wait: `cb_wait_front(inout_cb, (q_subblock + 1) * tiles_per_row * cols_in_row)` (cumulative)
- Add `SUB_EXP_BLOCK_INIT` profiling zone around init (from `e29a40dc146`)

### Step 3: Update `blocked_matmul_and_pack`

- Remove `SEQUENTIAL_OUTPUT` template parameter (always use absolute offsets now)
- All pack uses `pack_tile<true>` with row-major `(r + q_subblock * SUBBLOCK_H) * OUT_NUM_COLS + out_col_offset + c`

### Step 4: Update `apply_mask_to_row_buffer` → convert to updated `apply_padded_mask` style

Replace our `apply_mask_to_row_buffer` (which accumulated the full mask) with the updated `apply_padded_mask` from the example:
- Add `SBH` template parameter for multi-row support
- Add `q_subblock` runtime parameter for cb_qkt_im offset: `row_offset = (q_subblock * SBH + row) * num_cols`
- This is more efficient than our v1 approach (only masks padded positions, not full row)

**Note:** This reverts to using a single `-inf` tile (like the example) rather than the full padded mask from c_3. We need to decide whether to keep the c_3 full mask approach or switch to the `-inf` tile approach. The `-inf` tile approach requires allocating a small 1-tile CB for the `-inf` value. The c_3 approach works but wastes cycles accumulating zero tiles. **Recommendation:** Keep c_3 approach for now (simpler integration with existing writer), but use the `q_subblock` offset to only accumulate the current row group's tiles instead of the full mask.

### Step 5: Update `sdpa_inner_loop_step`

Major restructuring:
1. **Remove `cb_qkt_row_A` and `cb_qkt_row_B` template parameters**
2. **Remove alternating buffer aliases** (`alias_cur_qkt_row` / `alias_prev_qkt_row`)
3. **Phase 1 rewrites:**
   - Remove `cb_reserve_back(alias_cur_qkt_row, row_tiles)` — matmul writes directly to `cb_qkt_im`
   - `blocked_matmul_and_pack` now writes to `cb_qkt_im` at `q_subblock` offset
   - `sub_exp_block_bcast_cols` operates in-place on `cb_qkt_im` (single `inout_cb`)
   - Remove `cb_push_back(cb_qkt_im, row_tiles)` / `cb_pop_front(alias_prev_qkt_row, row_tiles)` pair → replace with `cb_push_back_hold_wr_ptr(cb_qkt_im, row_tiles)`
   - `reduce_c_row_group` reads from `cb_qkt_im` at `q_subblock` position (was reading from row buffer at position 0)
   - Remove `std::swap(alias_cur_qkt_row, alias_prev_qkt_row)`
4. **Phase 2 drain rewrites:**
   - `sub_exp_block_bcast_cols` in-place on `cb_qkt_im` (remove `alias_prev_qkt_row`)
   - Remove `cb_push_back(cb_qkt_im, row_tiles)` / `cb_pop_front(alias_prev_qkt_row, row_tiles)` from drain
   - sbh==1 path: keep split-matmul overlap (unchanged logic, just different CB sources)
   - **Add sbh>1 path:** drain all sub_exp first, then single full-inner-dim matmul (can't split inner dim because `matmul_block` uses `INNER_DIM` as in0 row stride, which must equal Sk_chunk_t for multi-row subblocks)
5. **Relax static_assert** on `kt_num_subblocks` from `1 || 2` to `1-4`

### Step 6: Update `sdpa_standard_v2`

- Remove `cb_qkt_row_A` and `cb_qkt_row_B` template parameters
- Update `sdpa_inner_loop_step` call to match new signature

### Step 7: Update `sdpa.cpp`

- Remove `cb_qkt_row_A`, `cb_qkt_row_B`, `cb_recip_scratch` constexpr declarations
- Update `sdpa_standard_v2` template instantiation to remove row buffer CB args
- `cb_recip_scratch` for `normalize_row_streaming` needs a new home — use one of the existing small CBs or keep c_4 as a 1-tile scratch (allocated only for streaming, much smaller than before)

### Step 8: Update `sdpa_program_factory.cpp`

1. **Remove c_4 and c_6 row buffer CB allocation** under `use_streaming_compute`
2. **Remove L1 budget constraint** (`streaming_row_buffer_l1 <= streaming_l1_budget`)
3. **Relax subblock_h constraint** from `qk_out_subblock_h == 1` to `qk_out_subblock_h <= 2`
   - sbh=2 now works correctly (bugs fixed in `a17ef4aec14`)
   - sbh=3+ still excluded by `sbh * vDHt <= dst_size` (3×4=12>8)
4. **Optionally allocate a 1-tile c_4 CB** for `cb_recip_scratch` (used by `normalize_row_streaming`). This is 1 tile × 2048 bytes = 2 KB, negligible L1 impact.

### Step 9: Verify constraints relaxation

After changes, the gating condition becomes:
```cpp
const bool use_streaming_compute =
    !is_causal && !use_provided_mask && !use_attention_sink &&
    sliding_window_size.value_or(0) == 0 && !is_chunked &&
    qk_out_subblock_h * vDHt <= dst_size &&
    qk_out_subblock_h <= 2 &&
    Sk_chunk_t % (8 / qk_out_subblock_h) == 0 &&
    vDHt <= 8;
```

Expected streaming enablement for WAN shapes:

| S | q_chunk | k_chunk | Sq_chunk_t | Sk_chunk_t | sbh | sbh×vDHt | Streaming? |
|---|---------|---------|-----------|-----------|-----|----------|-----------|
| 9472 | 224 | 256 | 7 | 8 | 1 | 4 | **Yes** |
| 9472 | 288 | 256 | 9 | 8 | 1 | 4 | **Yes** |
| 9472 | 224 | 512 | 7 | 16 | 1 | 4 | **Yes** (L1 no longer blocks) |
| 9472 | 288 | 512 | 9 | 16 | 1 | 4 | **Yes** |
| 2368 | 224 | 256 | 7 | 8 | 1 | 4 | **Yes** |
| 2368 | 288 | 256 | 9 | 8 | 1 | 4 | **Yes** |
| 9472 | 224 | 128 | 7 | 4 | 2 | 8 | **Yes** (sbh=2 now works) |
| 9472 | 288 | 128 | 9 | 4 | 2* | 8* | No (9%2≠0, sbh falls to 1, then 4%8≠0) |
| 2368 | 224 | 128 | 7 | 4 | 2 | 8 | **Yes** |
| 2368 | 288 | 128 | 9 | 4 | 2* | 8* | No (same issue) |

*For Sq_chunk_t=9, Sk_chunk_t=4: `determine_largest_subblock_size(9, 4, 8)` → sbh=3 (9%3=0, 3×(4/3)... actually 8/3=2, 3×2=6≤8). But sbh=3 > 2, so excluded. Falls to sbh=1, but Sk_chunk_t=4, 4%(8/1)=4%8≠0 → streaming disabled. This is expected — Sq_chunk_t=9 with Sk_chunk_t=4 doesn't divide cleanly.

## Test Plan

```bash
./build_metal.sh --release
rm -rf /localdev/pjosipovic/tt-metal/built/*

# Accuracy
PYTHONPATH="..." pytest tests/.../test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy -v

# Performance comparison
PYTHONPATH="..." pytest tests/.../test_scaled_dot_product_attention_sprint.py::test_sdpa_create_perf_table -v -s

# Determinism
PYTHONPATH="..." pytest tests/.../test_scaled_dot_product_attention_sprint.py::test_sdpa_determinism -v
```

Expected PCC >= 0.9997 for all streaming-enabled cases (matching example kernel's reported PCC >= 0.999714).

## Risks

| Risk | Mitigation |
|------|------------|
| `cb_push_back_hold_wr_ptr` uses internal CB interface | Function is simple (wr_ptr rewind). Verify with DPRINT that pushed tile count matches expected. |
| In-place sub_exp on cb_qkt_im | Already validated in example (15/15 tests pass). Read positions == write positions eliminates data aliasing. |
| sbh=2 SALAD correction DST pressure | sbh=2 × vDHt=4 = 8 = dst_size. Exactly at limit but works (validated in example). |
| Phase 2 drain divergence for sbh>1 | Two code paths (sbh==1 split-matmul vs sbh>1 full-matmul). Both validated in example. |
