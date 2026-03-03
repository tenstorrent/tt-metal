# Plan: Skip Padded K/V Tile Computation (Approach B)

## Context

When K sequence length doesn't divide evenly into Sk_chunk_t-sized chunks, the last chunk is zero-padded. Currently, all computation proceeds on the full chunk — matmuls on zero K tiles, masking with -inf, exp(-inf)=0, multiplying 0×V — contributing nothing to output. For pad=6/Sk=16, this wastes 37.5% of the last K chunk's compute.

**Approach B** dispatches to a second `sdpa_inner_loop_step` instantiation with `effective_Sk = Sk_chunk_t - padded_k_tiles` on the last K chunk. All matmul, sub_exp, and reduce operations use the reduced dimension. No reader or host changes needed.

Reference: `tt_metal/programming_examples/sdpa_single_core/kernels/skip_padding.md`

## File to Modify

`tt_metal/programming_examples/sdpa_single_core/kernels/compute/sdpa.cpp`

(No changes to reader, writer, or host code.)

## Implementation Steps

### Step 1: Add `KT_stride` template parameter to `sdpa_inner_loop_step`

Add a defaulted template parameter `KT_stride` (default = `Sk_chunk_t`) to `sdpa_inner_loop_step`. This separates the KT CB indexing stride (physical layout from reader) from the compute Sk dimension.

- Template line ~897: add `uint32_t KT_stride = Sk_chunk_t` after `cb_mask_in`
- The standard path passes default (KT_stride = Sk_chunk_t = same)
- The reduced path passes KT_stride = original Sk_chunk_t

### Step 2: Replace fixed subblocking with partial-subblock-aware constants

Replace the existing Phase 1 constants (lines ~910-917):

```cpp
// Current:
constexpr uint32_t kt_num_subblocks = Sk_chunk_t / qkt_subblock_w;
static_assert(Sk_chunk_t % qkt_subblock_w == 0, ...);

// New:
constexpr uint32_t kt_num_full_subblocks = Sk_chunk_t / qkt_subblock_w;
constexpr uint32_t kt_remainder = Sk_chunk_t % qkt_subblock_w;
constexpr bool has_partial_subblock = (kt_remainder > 0);
```

Remove the `static_assert(Sk_chunk_t % qkt_subblock_w == 0, ...)`. The standard path always has `kt_remainder == 0` (the existing constraint is enforced by the host), so the partial-subblock code compiles out via `if constexpr`.

### Step 3: Modify Phase 1 loop for partial last subblock

After the `for (kt_subblock = 0; kt_subblock < kt_num_full_subblocks; ++kt_subblock)` loop, add an `if constexpr (has_partial_subblock)` block that:

1. Calls `sub_exp_block_bcast_cols<..., kt_remainder, ...>` for the prev row (if q_subblock > 0)
2. Re-inits matmul HW with `ct_dim = kt_remainder`
3. Calls `blocked_matmul_and_pack<true, kt_remainder, sbh, in0_block_w, KT_stride, Sk_chunk_t, ...>` — note `IN1_STRIDE = KT_stride` (original Sk for KT CB layout), `OUT_NUM_COLS = Sk_chunk_t` (effective_Sk for cb_qkt_im)
4. On Blackhole: reconfigure `llk_pack_mop_config` for the narrow width before, restore after

Also update the existing full-subblock matmul call to use `KT_stride` for IN1_STRIDE (line 980): change `Sk_chunk_t` to `KT_stride` in the first stride position. `OUT_NUM_COLS` remains `Sk_chunk_t` (which IS effective_Sk in the reduced instantiation).

### Step 4: Remove `apply_padded_mask` from reduced path

The `if constexpr (padded_k_tiles > 0)` block at line ~998 stays as-is. On the reduced path, the caller passes `padded_k_tiles = 0`, so this block compiles out — no mask needed since there are no padded tiles.

### Step 5: Modify Phase 2 drain for partial subblock

The drain loop (sbh=1 path, line ~1075) currently splits the V matmul by `kt_num_subblocks`. With a partial subblock, the inner dims are uneven (8 + 2 for effective_Sk=10).

**Approach:** When `has_partial_subblock` is true, use the sbh>1-style drain (drain all sub_exp first, then one full V matmul with `INNER_DIM = effective_Sk`). Guard with `if constexpr`:

```cpp
if constexpr (sbh == 1 && !has_partial_subblock) {
    // Existing split-drain loop (unchanged)
} else {
    // Non-split drain: all sub_exp first (full + partial subblocks), then full V matmul
    for (kt_sub = 0; kt_sub < kt_num_full_subblocks; ++kt_sub) {
        sub_exp_block_bcast_cols<..., qkt_subblock_w, ...>(..., kt_sub);
    }
    if constexpr (has_partial_subblock) {
        sub_exp_block_bcast_cols<..., kt_remainder, ...>(..., kt_num_full_subblocks);
    }
    // waits + full V matmul with INNER_DIM = Sv_chunk_t (= effective_Sk)
}
```

When `has_partial_subblock` is false, the standard sbh=1 path is taken (zero-overhead for existing code). When true (only on reduced path), the non-split drain runs.

### Step 6: Dispatch in `kernel_main`

In the K-chunk loop (line ~1358), add dispatch for the reduced path:

```cpp
constexpr uint32_t effective_Sk = Sk_chunk_t - padded_k_tiles;

for (uint32_t k_chunk = 0; k_chunk < num_k_chunks; k_chunk++) {
    bool is_first = (k_chunk == 0);
    bool is_last = (k_chunk == num_k_chunks - 1);

    if (is_last && padded_k_tiles > 0) {
        // Reduced path: compute with effective_Sk, no padding mask
        call_step_reduced(profiling_tag);  // instantiated with effective_Sk
        // Compensation pops: reader loaded full Sk_chunk_t, reduced path popped effective_Sk
        cb_pop_front(cb_kt_in, head_dim_t * padded_k_tiles);
        cb_pop_front(cb_v_in, padded_k_tiles * head_dim_t);
    } else {
        call_step(profiling_tag);  // standard path
    }
    // ... post-iteration cleanup unchanged
}
```

The `call_step_reduced` lambda calls `sdpa_inner_loop_step` with:
- `Sk_chunk_t = effective_Sk`
- `Sv_chunk_t = effective_Sk`
- `padded_k_tiles = 0` (no masking needed)
- `KT_stride = Sk_chunk_t` (original, for KT CB indexing)

**Compensation pops correctness:** After `sdpa_inner_loop_step` returns, the K/V CBs have `padded_k_tiles * head_dim_t` leftover tiles. The compensation pops remove them from the front of the FIFO before the compute reads new data for the next Q iteration. Even if the reader has already pushed new K tiles (due to double-buffering), the FIFO ordering ensures stale tiles are at the front and get correctly removed.

### Step 7: Verify `kt_num_subblocks` usage in the outer scope

The variable `kt_num_subblocks` is used in Phase 2 drain. After Step 2, the name changes to `kt_num_full_subblocks`. Audit all references in sdpa_inner_loop_step:
- Phase 1 loop bound → `kt_num_full_subblocks`
- Phase 2 drain loop → guarded by `if constexpr` (Step 5)
- `matmul_inner = qktv_in0_block_w / kt_num_subblocks` → only in the non-partial path

## Template Instantiation Summary

Standard path (no padding or non-last chunks):
- `sdpa_inner_loop_step<..., Sk_chunk_t, Sv_chunk_t, ..., padded_k_tiles, ..., Sk_chunk_t /*KT_stride*/>`
- `kt_remainder = 0`, `has_partial_subblock = false` → all partial-subblock code compiles out

Reduced path (last chunk with padding):
- `sdpa_inner_loop_step<..., effective_Sk, effective_Sk, ..., 0 /*padded=0*/, ..., Sk_chunk_t /*KT_stride*/>`
- `kt_remainder` may be > 0 → narrow sub_exp + matmul instantiations

New template instantiations generated:
- `blocked_matmul_and_pack<true, kt_remainder, sbh, head_dim_t, Sk_chunk_t, effective_Sk>` (narrow Q@KT)
- `sub_exp_block_bcast_cols<..., kt_remainder, ...>` (narrow sub_exp)
- `reduce_block_max_row<effective_Sk>` (narrower reduce — verified: no alignment restrictions, any value 1-127 works)

## Verification

```bash
# Build
./build_metal.sh --build-programming-example

# Reset device
pkill -9 -f pytest || true
tt-smi -r $(tt-smi -ls 2>&1 | grep -oP '^\│ \K\d+' | head -1)

# Run the motivating test case (pad=6, non-aligned)
timeout 120 pytest tt_metal/programming_examples/sdpa_single_core/generate_and_test_sdpa.py -v -k "3q_5k-random-sk16-pad6"

# Run all padded test cases (regression check)
timeout 300 pytest tt_metal/programming_examples/sdpa_single_core/generate_and_test_sdpa.py -v -k "pad"

# Run full test suite
timeout 600 pytest tt_metal/programming_examples/sdpa_single_core/generate_and_test_sdpa.py -v
```

All tests must pass with PCC > 0.99, matching baseline values. No PCC regression expected — the optimization computes the exact same mathematical result, just skipping zero-contribution work.
