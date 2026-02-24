# Plan: Integrate Streaming SDPA Compute Kernel for Non-Causal Cases

## Context

Branch `origin/dnijemcevic/sdpa_benchmark` (latest: `12dfcf4de8a`) implements a row-by-row streaming SDPA compute kernel with:
- Alternating row buffers (FPU/SFPU overlap in Phase 1 and Phase 2)
- Streaming per-row normalization (no separate matmul_reduce + recip pass)
- Padded-K mask support via single -inf tile + L1 accumulation
- Flexible Sk_chunk_t (8 and 16 — kt_num_subblocks 1 or 2)

Goal: integrate this into the production SDPA for **non-causal** cases so all 18 `test_sdpa_accuracy` cases exercise the new path.

> **Note**: All example line references below are to `origin/dnijemcevic/sdpa_benchmark:tt_metal/programming_examples/sdpa_single_core/kernels/compute/sdpa.cpp` (1249 lines). Fetch that branch to follow along.

## Files to Modify

| File | Change |
|------|--------|
| `ttnn/.../sdpa/device/kernels/compute/compute_common.hpp` | Add 8 new helper functions + `sdpa_inner_loop_step` + `sdpa_standard_v2` wrapper |
| `ttnn/.../sdpa/device/kernels/compute/sdpa.cpp` | Gate between `sdpa_standard` and `sdpa_standard_v2` via `constexpr bool use_streaming_compute` compile-time arg |
| `ttnn/.../sdpa/device/sdpa_program_factory.cpp` | Add compile-time args, allocate 2 new CBs for non-causal streaming path |

Reader (`reader_interleaved.cpp`) and writer (`writer_interleaved.cpp`) are **unchanged** — writer already generates the padded mask in c_3 via `generate_mask`, and streaming compute reuses it directly.

## CB Remapping (Example → Production)

The example uses different CB indices than production. Remapping happens in `sdpa.cpp` constants.

### Existing CBs (index remap only — already allocated in production)

| Role | Example CB | Production CB | Notes |
|------|-----------|---------------|-------|
| Q input | c_0 | **c_0** | Same |
| K input | c_1 | **c_1** | Same |
| V input | c_3 | **c_2** | Reader pushes V to c_2 |
| QK intermediate | c_2 | **c_24** | Already allocated unconditionally |
| identity_scale | c_5 | **c_5** | Already allocated unconditionally |
| col_identity | c_8 | **c_7** | Already allocated unconditionally |
| normalized_out | c_9 | **c_16** | Production output CB, already allocated |
| output im A/B | c_25/c_26 | **c_25/c_26** | Same |
| max A/B | c_27/c_28 | **c_27/c_28** | Same |
| sum A/B | c_29/c_30 | **c_29/c_30** | Same |
| exp_max_diff | c_31 | **c_31** | Same |

### New CB allocations (under `use_streaming_compute`)

| Role | Example CB | Production CB | Size | Notes |
|------|-----------|---------------|------|-------|
| QK row buffer A | c_4 | **c_4** | `subblock_h * Sk_chunk_t` tiles, Float16_b | Reuses attention_sink slot (safe: gating excludes `use_attention_sink`). Also reused as recip scratch in Phase 2. |
| QK row buffer B | c_6 | **c_6** | `subblock_h * Sk_chunk_t` tiles, Float16_b | Reuses page_table slot (safe: gating excludes `is_chunked`) |

### Removed from example (no production equivalent needed)

| Role | Example CB | Notes |
|------|-----------|-------|
| -inf mask tile | c_7 | Replaced by full padded mask in c_3 (see `apply_mask_to_row_buffer`) |
| recip scratch | c_10 | c_4 serves this role in Phase 2 (row buffers are free then) |

## Implementation Steps

### Step 1: Add helper functions to `compute_common.hpp`

Port 8 functions from the example kernel, keeping them as standalone additions that don't modify existing functions:

1. **`blocked_matmul_and_pack`** (lines 594-659) — Subblock matmul with explicit offset packing. Supports both sequential and absolute-offset output modes via `SEQUENTIAL_OUTPUT` template.

2. **`reduce_c_row_group`** (lines 529-592) — Per-row-group max reduction with optional eltwise_max. Uses `reduce_block_max_row` + cumulative wait pattern.

3. **`sub_exp_block_bcast_cols`** (lines 422-527) — NOT in-place. Reads from row buffer, subtracts max, applies exp with ReLU clamping, writes to QK intermediate. L1-accumulates partial row sums.

4. **`sub_exp_first_col_blocks`** (lines 301-342) — Column-only exp(prev_max - cur_max) for SALAD corrections.

5. **`mul_bcast_cols_l1_acc`** (lines 343-382) — SALAD sum correction with L1 accumulate at explicit offset.

6. **`mul_block_bcast_cols_acc`** (lines 384-420) — SALAD output correction with L1 accumulate.

7. **`apply_mask_to_row_buffer`** (NEW — not from example) — L1-accumulates mask tiles from c_3 onto the row buffer. Reads `sbh * Sk_chunk_t` mask tiles at the correct row group offset, accumulates onto row buffer tiles in reserved state. Replaces the example's `apply_padded_mask` (which used a single -inf tile) by reusing the existing full padded mask.
   **Format note**: When `!use_provided_mask`, the mask CB c_3 is allocated as **Bfp4_b** (see `sdpa_program_factory.cpp:526`) while row buffers are Float16_b. The L1-accumulate path handles this via unpack (Bfp4_b→DST) + add + pack (DST→Float16_b). Ensure the unpack source format is configured for c_3's Bfp4_b before the accumulate loop, and restored afterward.

8. **`normalize_row_streaming`** (lines 730-794) — Per-row pipeline: matmul_reduce + recip-in-DST + mul_bcast_cols. Requires `head_dim_t <= 8`.

**Note**: The SFPI functions (`calculate_exponential_polynomial`, `calculate_exponential_first_column`, `exp_tile_first_column`, `calculate_recip_first_column`, `recip_tile_first_column`) and `sdpa_reduce_copy_tile_to_dst_init_short` already exist in production `compute_common.hpp` with identical implementations. Do NOT re-add them.

### Step 2: Add `sdpa_inner_loop_step` to `compute_common.hpp`

Port lines 796-1127 from the example, with one key adaptation: replace `apply_padded_mask` call with `apply_mask_to_row_buffer` (Step 1 #7). Template parameters include CB indices (for remapping) plus `use_padded_mask` (bool) and `cb_mask_in`.

The function is ~330 lines implementing two-phase algorithm:
- **Phase 1**: Q@KT row-by-row with alternating buffers, interleaved sub_exp. On last K chunk when `use_padded_mask`: L1-accumulate mask tiles from c_3 onto each row buffer before reduce_max. Pop c_3 after all rows processed.
- **Phase 2**: Drain + QKT@V with SALAD corrections, streaming normalization on last K iter

### Step 3: Add `sdpa_standard_v2` wrapper

Wraps `sdpa_inner_loop_step` in the Q-chunk / K-chunk outer loop with ping-pong buffer management. Port from example `kernel_main()` lines 1129-1249, parameterized on CB indices.

### Step 4: Gate in `sdpa.cpp`

Read `use_streaming_compute` as a `constexpr bool` from compile-time args (new arg index 30). Then use `if constexpr (use_streaming_compute)` to:
1. Use the existing `qk_subblock_h` (already at arg index 12) as the row-group size — no new arg needed since the streaming kernel's `subblock_h` is identical to the existing QK subblock height.
2. Set remapped CB constants (V→c_2, QK_im→c_24, mask→c_3, col_identity→c_7, normalized_out→c_16)
3. Call `sdpa_standard_v2` instead of `sdpa_standard`

When `use_streaming_compute` is false (causal, masked, chunked, etc.), the existing `sdpa_standard` path is taken — zero impact on those codepaths since `if constexpr` eliminates the dead branch at compile time.

### Step 5: Modify `sdpa_program_factory.cpp`

**Gating condition** (vanilla non-causal SDPA):
```cpp
bool use_streaming_compute = !is_causal && !use_provided_mask &&
    !use_attention_sink && sliding_window_size == 0 && !is_chunked;
```

When `use_streaming_compute`:

1. **Reuse existing `qk_out_subblock_h`** (already computed at factory line ~307 via `determine_largest_subblock_size(Sq_chunk_t, Sk_chunk_t, dst_size)`). This is already passed as arg 12. No new `subblock_h` arg needed.

2. **Append one compile-time arg** to compute kernel (after existing 30 args):
   - Arg 30: `use_streaming_compute` (bool)
   - (`use_padded_mask` remains via existing arg 25 — writer still generates full padded mask in c_3)
   - TensorAccessorArgs follow at index 31 (shifted by 1 from current index 30; the compute kernel does not read TensorAccessorArgs, so this is safe).

3. **Allocate 2 new CBs** (see table above):
   - c_4 (row buffer A): `qk_out_subblock_h * Sk_chunk_t` tiles, Float16_b
   - c_6 (row buffer B): `qk_out_subblock_h * Sk_chunk_t` tiles, Float16_b

When `!use_streaming_compute`: still append `false` as arg 30 (keeps TensorAccessorArgs at constant index 31 in both paths).

**Constraints** (validated at host):
- `Sq_chunk_t % qk_out_subblock_h == 0`
- `qk_out_subblock_h <= 8` (ensures the DST-tiles-per-row-group ratio below is well-defined)
- `Sk_chunk_t % (8 / qk_out_subblock_h) == 0` (ensures each row group's K tiles divide evenly into DST-sized batches for the streaming matmul; `8 / subblock_h` is the number of K-tile columns that fit in DST alongside `subblock_h` output rows)
- `head_dim_t <= 8`

## Test Coverage

All 18 non-causal test cases hit the streaming path:

| Sk | k_chunk | Skt | padded_k_tiles | Path |
|----|---------|-----|----------------|------|
| 9472 | 128 | 296 | 0 | v2 |
| 9472 | 256 | 296 | 0 | v2 |
| 9472 | 512 | 296 | 8 (304-296) | v2 (padded) |
| 2368 | 128 | 74 | 2 (76-74) | v2 (padded) |
| 2368 | 256 | 74 | 6 (80-74) | v2 (padded) |
| 2368 | 512 | 74 | 6 (80-74) | v2 (padded) |

The 3 Q-dimension variants per shape produce identical K-padding behavior. However, `use_padded_mask` is also triggered by Q padding (`padded_Sq != Sq` — see `sdpa_program_factory.cpp:148`). If any Q-dimension variant produces `Sq % q_chunk != 0`, the mask will include Q-padded rows in addition to (or instead of) K-padded columns. The mask is a full 2D `Sq_chunk_t × Sk_chunk_t` block, so `apply_mask_to_row_buffer` handles both Q and K padding uniformly — Q-padded rows get -inf across all K positions, zeroing their softmax output. Verify the actual Sq values in the test to confirm which cases trigger Q padding.

All 18 cases (6 shapes × 3 Q chunk sizes) exercise the streaming path.

## Verification

```bash
./build_metal.sh --release
rm -rf /localdev/pjosipovic/tt-metal/built/*

PYTHONPATH="/localdev/pjosipovic/tt-metal/tools:/localdev/pjosipovic/tt-metal:/localdev/pjosipovic/tt-metal/ttnn" \
  pytest tests/tt_eager/python_api_testing/unit_testing/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy -v

PYTHONPATH="/localdev/pjosipovic/tt-metal/tools:/localdev/pjosipovic/tt-metal:/localdev/pjosipovic/tt-metal/ttnn" \
  pytest tests/tt_eager/python_api_testing/unit_testing/test_scaled_dot_product_attention_sprint.py::test_sdpa_determinism -v
```

Expected: PCC >= 0.9997, RMSE <= 4e-2 for all combinations.

**If a test hangs**:
- Reset the device: `tt-smi -r`
- Diagnose with: `./tools/tt-triage.py`
- For kernel-level debugging (DPRINT): see `docs/source/tt-metalium/tools/kernel_print.rst`

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| CB protocol deadlock | Trace all push/pop/wait/reserve paths. Add DPRINT for debugging. |
| L1 overflow from row buffers | Row buffers are `sbh * Sk_chunk_t` tiles (small). Verify total L1 budget. |
| DST pressure in normalize | head_dim_t=4 for WAN shapes (well under 8 limit). |
| Mask format mismatch | Padded mask may be Bfp4_b, row buffer is Float16_b. L1 accumulate handles conversion via unpack→DST→pack. Verify with DPRINT. |
| Causal/masked tests regress | `use_streaming_compute` is false, gated by `if constexpr` — dead code eliminated at compile time. |
