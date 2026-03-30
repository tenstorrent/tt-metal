# Fake Tilization Analysis & Next Steps

## Summary

Your `deepseek_moe_post_combine_reduce` implementation **already uses fake tilization** correctly, similar to the `rm_scaled_add` reference. Both implementations:

1. Read ROW_MAJOR data page-by-page without tilization
2. Write directly to L1 circular buffers
3. Treat the ROW_MAJOR data as "tiles" (1024 elements each) for compute operations
4. Process using tile-based math kernels (mul_tiles_bcast, binary_dest_reuse_tiles)

## Key Comparison

| Aspect | rm_scaled_add | Your reduce implementation |
|--------|---------------|---------------------------|
| **Reader Strategy** | Reads all pages in loop | Reads one expert per iteration |
| **CB Setup** | tile_size=2048, full tensor size | tile_size=2048, one expert at a time |
| **Compute Batching** | Processes N tiles in DST[0..7] | Processes 7 tiles in DST[0..6] |
| **Data Flow** | Read all → Compute all → Write all | Read expert → Compute → Repeat |
| **Memory Footprint** | Higher (holds all data) | Lower (one expert at a time) |

## Critical Issue: Page Indexing

Your reader kernel (line 94) assumes:
```cpp
uint32_t expert_output_start_page = global_token_idx * num_experts + expert_idx;
```

This works **IF** ROW_MAJOR buffer pages follow this layout:
- Page 0: token 0, expert 0
- Page 1: token 0, expert 1
- ...
- Page 7: token 0, expert 7
- Page 8: token 1, expert 0
- ...

**To Verify**: Run `test_fake_tiles_data_layout` and `test_page_layout_simple.py` to confirm page mapping.

## Potential Issues

### 1. Buffer Page Size Mismatch
```cpp
// Your reader line 46 - compile-time constant
constexpr uint32_t combine_page_size = emb_dim * 2;
```

**Risk**: If actual buffer page_size ≠ `emb_dim * 2`, reads will be misaligned.

**Fix**: Either:
- Assert buffer page size matches expectation
- OR pass actual buffer page size as runtime arg (like rm_scaled_add does)

### 2. Multi-dimensional Shape Handling

Your program factory supports arbitrary shapes via `expert_dim` parameter, but the reader assumes expert dimension immediately precedes embedding dimension.

**Example shapes**:
- `[1, 8, 7168]` - works if expert_dim=1
- `[1, 1, 2, 8, 7168]` - works if expert_dim=3
- `[1, 1, 8, 2, 7168]` - **BREAKS** if expert_dim=2 (emb_dim not last)

**Current limitation**: `emb_dim = combine_shape[-1]` assumes embedding is last dimension.

### 3. Weight Buffer Layout

Weights are loaded as tiles (2048 bytes) but only first element is used for SCALAR broadcast.

```cpp
// Reader line 78-79
// Read weight tile (only first element will be used for SCALAR broadcast)
noc_async_read_page(weight_page_idx, weight_addrg, weight_write_addr);
```

**Verify**: Weight page indexing matches token/expert structure:
```cpp
uint32_t weight_page_idx = global_token_idx * num_experts + expert_idx;
```

## Debugging Action Plan

### Phase 1: Validate Assumptions (Before Code Changes)

1. **Run test_fake_tiles_data_layout**
   ```bash
   pytest tests/ttnn/unit_tests/operations/test_fake_tiles_concept.py::test_fake_tiles_data_layout -v -s
   ```
   Confirms: ROW_MAJOR `[1, 8, 7168]` has 8 pages × 14336 bytes

2. **Run test_page_layout_simple.py**
   ```bash
   python test_page_layout_simple.py
   ```
   Confirms: Page indexing formula `page_idx = token_idx * num_experts + expert_idx`

3. **Add Debug Logging**
   ```cpp
   // In reader after line 101
   DPRINT << "  READ: page=" << expert_output_start_page
          << " token=" << global_token_idx
          << " expert=" << expert_idx << ENDL();
   ```

### Phase 2: Fix Issues

1. **Add Buffer Validation** (in program_factory.cpp after line 102):
   ```cpp
   TT_FATAL(
       combine_buffer->page_size() == emb_dim * 2,
       "Buffer page size {} != expected {} (emb_dim={} * 2)",
       combine_buffer->page_size(), emb_dim * 2, emb_dim);
   ```

2. **Pass Actual Buffer Metadata** (like rm_scaled_add):
   ```cpp
   // In SetRuntimeArgs for reader
   uint32_t combine_buffer_page_size = combine_buffer->page_size();
   uint32_t combine_num_pages = combine_buffer->num_pages();

   SetRuntimeArgs(program, reader_kernel_id, core, {
       combine_buffer->address(),
       weight_buffer->address(),
       tokens_for_this_core,
       token_start,
       combine_buffer_page_size,  // NEW
       combine_num_pages          // NEW
   });
   ```

3. **Test with Smaller Shapes First**:
   ```python
   # In test file, start simple
   test_known_values_simple(seq_len=1, num_experts=1, emb_dim=1024)
   test_known_values_simple(seq_len=1, num_experts=2, emb_dim=2048)
   test_known_values_simple(seq_len=2, num_experts=4, emb_dim=2048)
   # Then scale up to full size
   test_known_values_simple(seq_len=3200, num_experts=8, emb_dim=7168)
   ```

### Phase 3: Optimization (After It Works)

1. **Sparsity Optimization** (Issue #39640 Approach 2)
   - Skip experts with zero weights
   - Early exit in reader/compute loops

2. **Multi-core Scaling**
   - Distribute tokens across cores (framework already in place lines 168-200)

3. **Double Buffering**
   - Pipeline reader/compute with ping-pong CBs

## Key Differences from rm_scaled_add

| Aspect | rm_scaled_add | Your reduce |
|--------|---------------|-------------|
| **Operation** | `A + B * scalar` | `Σ(expert_i * weight_i)` over experts |
| **Data Flow** | Bulk read → Compute → Bulk write | Streaming per expert |
| **Memory Trade-off** | Higher footprint, simpler control | Lower footprint, more orchestration |
| **CB Size** | Holds all tiles (`n_tiles * tile_size`) | Holds one expert (`emb_dim_tiles * tile_size`) |
| **Best For** | Element-wise ops on full tensors | Reductions with streaming |

## Why Your Approach is Better for Reduce

Your streaming approach (process one expert at a time) is **superior** for reduction operations because:

1. **Lower Memory**: Only 7 tiles in CB at once vs 7 × num_experts
2. **Better Cache Locality**: Process expert completely before moving to next
3. **Scalable**: Works regardless of num_experts (rm_scaled_add style needs CB for all data)

## Conclusion

You're on the right track! The fake tilization is correctly implemented. The issues are likely:
- Buffer page indexing assumptions (verify with tests)
- Shape handling edge cases (multi-dimensional tensors)
- Missing validations (buffer metadata checks)

Run the diagnostic tests, add the suggested validations, and you should have it working soon! 🚀
