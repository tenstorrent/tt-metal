# Depthwise Conv2D Bias Implementation Plan

## Build & Test Commands

```bash
# Build
./build_metal.sh --release

# Test (with 15s timeout)
source python_env/bin/activate && \
TT_METAL_CLEAR_L1=0 \
TT_METAL_DPRINT_ONE_FILE_PER_RISC=1 \
TT_METAL_DPRINT_CORES="(0,0)" \
timeout 15 pytest tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py::test_groups_vs_pool2 -v

# Recovery if hung
tt-smi -r 1
```

---

## Step Transition Rules

1. **DO NOT proceed to the next step until the current step is fully completed**
2. **DO NOT advance steps automatically** - the user must explicitly trigger moving to the next step
3. **Wait for user confirmation** before marking a step as complete and starting the next one
4. **Each step contains**: What needs to be done, Why, Code changes, How to test, Key takeaways

---

## Progress Tracker

| Step | Status | Description |
|------|--------|-------------|
| 1 | COMPLETE | Add bias CB allocation in depthwise program factory |
| 2 | COMPLETE | Pass bias tensor to reader kernel via TensorAccessor |
| 3 | COMPLETE | Add bias reading logic to reader kernel |
| 4 | PENDING | Add bias addition to compute kernel |
| 5 | PENDING | Update shared_variables_t and runtime argument overrides |
| 6 | PENDING | Integration testing with bias |
| 7 | PENDING | Edge cases and validation |

---

## Current State Analysis

### How Conventional Conv2D Implements Bias

From `BIAS_IMPLEMENTATION.md`, conventional conv2d:

1. **Preparation** (`prepare_conv_bias_internal`):
   - Input shape: `[1, 1, 1, out_channels]` ROW_MAJOR
   - Pads to `[1, 1, 32, out_channels_padded]`
   - Converts to TILE layout
   - Converts to weights_dtype

2. **Device-side**:
   - Bias CB allocated with `per_core_out_matrix_width_ntiles` pages
   - Writer kernel reads bias tiles from DRAM, multicasts to receivers
   - Compute kernel uses `add_tiles_bcast_rows()` after matmul

3. **Key code locations**:
   - `conv2d_op_program_factory_common.cpp:259-264` - CB allocation
   - `conv_bmm_tilize.cpp:529-581` - Compute kernel bias fusion
   - `writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp:293-350` - Bias reading

### Current Depthwise Conv2D Implementation

From `conv2d_op_depthwise_program_factory.cpp`:

1. **Pool-based approach**: Uses pool kernels (`reader_pool_2d.cpp`, `compute_pool_2d.cpp`)
2. **Bias signature exists** but is unused (line 36: `const std::optional<const Tensor>& bias`)
3. **No bias CB** is allocated
4. **No bias reading/addition** in kernels

### Mathematical Correctness

Depthwise conv with bias:
```
output[n, h, w, c] = bias[c] + Σ(input[n, h+kh, w+kw, c] * weight[kh, kw, c])
```

Since bias is added **after** the reduction, we can add bias after the pool-style reduction completes.

### Key Architectural Difference from Conventional Conv2D

**Conventional Conv2D (matmul-based)**:
- Works on full tiles in DST
- Bias added via `add_tiles_bcast_rows` directly after matmul
- Activation applied per-tile in DST
- Single pack to output

**Depthwise Conv2D (pool-based)**:
- Processes stick-by-stick (one output row at a time)
- Results accumulated in DST, then `pack_untilize_dest` to `pre_tilize_cb`
- After 32 sticks: `fast_tilize_block` creates full tiles in `out_cb`
- **Bias must be added AFTER tilization** for efficiency (32x speedup)
- Activation also moves to post-tilization to maintain correct order

**Optimized flow for depthwise**:
```
reduce → pack_untilize_dest → collect 32 sticks → tilize → out_cb
      → load from out_cb → add_bias → activation → pack back → out_cb
```

---

## Step 1: Add Bias CB Allocation in Depthwise Program Factory

### What Needs to Be Done

Add a circular buffer for bias in `conv2d_op_depthwise_program_factory.cpp`. The bias CB should follow the same pattern as conventional conv2d.

### Location

File: `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_depthwise_program_factory.cpp`
After line ~278 (after `clear_value_cb` allocation), add bias CB.

### Code Changes

```cpp
// Bias CB - only allocated if has_bias is true
uint32_t bias_cb_id = 32;  // Invalid CB ID by default
if (has_bias) {
    bias_cb_id = next_cb_index++;
    // Bias uses same data format as weights
    const tt::DataFormat bias_data_format = weight_data_format;
    const uint32_t bias_cb_pagesize = tt::tile_size(bias_data_format);
    // One tile row of bias per output channel tile
    const uint32_t bias_cb_npages = params.in_ntiles_c;  // out_channels = in_channels for depthwise
    tt::tt_metal::create_cb(
        bias_cb_id, program, parallel_config.grid, bias_cb_pagesize, bias_cb_npages, bias_data_format);
    log_debug(tt::LogOp, "CB {} (bias_cb) :: PS = {}, NP = {}", bias_cb_id, bias_cb_pagesize, bias_cb_npages);
}
```

### Why This Approach

- Follows the pattern from conventional conv2d
- Bias tiles align with output channel tiles
- Uses weight dtype for consistency
- Only allocates when `has_bias` is true

### How to Test

Build and run existing tests - should not break anything since bias CB is only allocated when `has_bias=true`.

### Key Takeaways

- CB allocation is conditional on `has_bias`
- `bias_cb_npages = in_ntiles_c` because for depthwise, `out_channels == in_channels`
- Track `bias_cb_id` for use in kernel compile args

---

## Step 2: Pass Bias Tensor to Reader Kernel via TensorAccessor

### What Needs to Be Done

Add bias tensor information to the reader kernel compile-time and runtime args using TensorAccessor pattern (similar to weight tensor).

### Location

File: `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_depthwise_program_factory.cpp`
After line ~558 (after weight TensorAccessor args), add bias accessor args.

### Code Changes

1. **Add compile-time args for bias** (extend `reader0_ct_args`):

```cpp
// Add at position 48+ (after weight-related args)
reader0_ct_args.push_back(has_bias);           // 48 - has_bias flag
reader0_ct_args.push_back(bias_cb_id);         // 49 - bias_cb_id
reader0_ct_args.push_back(params.in_ntiles_c); // 50 - bias_ntiles (same as out channel tiles)

// Add tensor accessor args for bias buffer (after weight accessor)
if (has_bias && bias.has_value()) {
    tt::tt_metal::TensorAccessorArgs(bias.value().buffer()).append_to(reader0_ct_args);
}
```

2. **Update the function signature** to ensure `bias` optional tensor is available in scope.

### Why This Approach

- TensorAccessor pattern is already used for weights
- Compile-time flag `has_bias` allows kernel to conditionally include bias code
- Bias tiles are read similarly to weight tiles

### How to Test

Build - no runtime impact yet since kernel doesn't use these args.

### Key Takeaways

- `has_bias` compile-time arg enables conditional compilation
- TensorAccessorArgs provides safe DRAM access from kernel
- Keep bias args optional/conditional to maintain backward compatibility

---

## Step 3: Add Bias Reading Logic to Reader Kernel

### What Needs to Be Done

Modify `reader_pool_2d.cpp` to read bias tiles from DRAM when `has_bias` is true. Alternatively, create a new reader kernel for depthwise conv with bias support.

### Location

File: `ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/reader_pool_2d.cpp`

### Approach Options

**Option A: Modify reader_pool_2d.cpp (Recommended for minimal changes)**
- Add compile-time args for bias
- Add bias reading logic gated by `has_bias` constexpr

**Option B: Create new reader_depthwise_conv2d.cpp**
- Fork from reader_pool_2d.cpp
- Add depthwise-specific bias handling

### Code Changes (Option A)

Add to compile-time args parsing (after line ~300):
```cpp
// Bias-related compile-time args
constexpr bool has_bias = get_compile_time_arg_val(48);
constexpr uint32_t bias_cb_id = get_compile_time_arg_val(49);
constexpr uint32_t bias_ntiles = get_compile_time_arg_val(50);
```

Add bias reading logic (once, at start of processing):
```cpp
if constexpr (has_bias) {
    // Read bias tiles from DRAM into bias CB
    // Use TensorAccessor to read bias_ntiles tiles
    cb_reserve_back(bias_cb_id, bias_ntiles);
    uint32_t bias_l1_addr = get_write_ptr(bias_cb_id);

    // Read all bias tiles (one pass, reused for all output positions)
    for (uint32_t tile_idx = 0; tile_idx < bias_ntiles; ++tile_idx) {
        noc_async_read_tile(tile_idx, s_bias, bias_l1_addr);
        bias_l1_addr += bias_tile_size;
    }
    noc_async_read_barrier();
    cb_push_back(bias_cb_id, bias_ntiles);
}
```

### Why This Approach

- Bias is read once per core (per-channel values are reused)
- Similar pattern to conventional conv2d writer kernel
- Uses existing TensorAccessor infrastructure

### How to Test

Add print statements in reader kernel to verify bias CB is populated correctly.

### Key Takeaways

- Bias is read once at the start, not per output position
- For width/block sharding, each core reads its own bias shard
- The `cb_push_back` makes bias available to compute kernel

---

## Step 4: Add Bias Addition and Activation to Compute Kernel (Optimized)

### What Needs to Be Done

Modify `compute_pool_2d.cpp` to add bias **after tilization** for optimal performance. This requires:
1. Removing activation from the per-stick loop
2. After tilization, loading tiles back from `out_cb`
3. Adding bias using `add_tiles_bcast_rows`
4. Applying activation on full tiles in DST
5. Packing back to `out_cb`

### Why This Approach is 32x More Efficient

**Per-stick bias addition (naive approach)**:
- For 32 sticks × N channel tiles = 32×N pack/unpack cycles
- Very expensive due to round-trip per stick

**Post-tilization bias addition (optimized approach)**:
- Tilization collects 32 sticks into tiles
- Add bias to full tiles: only N pack/unpack cycles
- **32x more efficient!**

### Location

File: `ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp`

### Code Changes

#### 1. Add compile-time args (after line ~76):
```cpp
constexpr bool has_bias = get_compile_time_arg_val(33);
constexpr uint32_t bias_cb_id = get_compile_time_arg_val(34);
constexpr uint32_t bias_ntiles = get_compile_time_arg_val(35);
```

#### 2. Add bias initialization (after SFPU init, around line ~129):
```cpp
if constexpr (has_bias) {
    // Wait for bias to be available (read by reader kernel)
    cb_wait_front(bias_cb_id, bias_ntiles);
}
```

#### 3. Add header include:
```cpp
#include "compute_kernel_api/bcast.h"
```

#### 4. CONDITIONALLY move activation from per-stick loop (lines 359-369):

**Key insight**: Only move activation to post-tilization when `has_bias=true`. If no bias, keep activation in the original location for better performance (avoids extra load/pack cycle).

**BEFORE**:
```cpp
#ifdef SFPU_OP_FUNC_ACTIVATION
            if (last_c_block) {
                for (uint32_t i = 0; i < partial_iter_output_tiles; ++i) {
                    SFPU_OP_FUNC_ACTIVATION
                }
            } else {
                for (uint32_t i = 0; i < max_tiles_per_iter; ++i) {
                    SFPU_OP_FUNC_ACTIVATION
                }
            }
#endif
```

**AFTER** (activation only skipped when has_bias):
```cpp
#ifdef SFPU_OP_FUNC_ACTIVATION
    // Only apply activation here if NO bias - otherwise it's applied post-tilization
    if constexpr (!has_bias) {
        if (last_c_block) {
            for (uint32_t i = 0; i < partial_iter_output_tiles; ++i) {
                SFPU_OP_FUNC_ACTIVATION
            }
        } else {
            for (uint32_t i = 0; i < max_tiles_per_iter; ++i) {
                SFPU_OP_FUNC_ACTIVATION
            }
        }
    }
    // If has_bias, activation will be applied after tilization with bias
#endif
```

#### 5. Add bias + activation after tilization (after `fast_tilize_block`, around line 410):

Add bias+activation handling **only when has_bias is true**. When no bias, the original code path is unchanged (activation already applied per-stick).

```cpp
fast_tilize_block(pre_tilize_cb_id, in_ntiles_c, out_cb_id);
fast_tilize_uninit(pre_tilize_cb_id, out_cb_id);

// ============ BIAS + ACTIVATION (post-tilization) ============
// Only needed when has_bias - otherwise activation already applied per-stick
if constexpr (has_bias) {
    // Load tiles from out_cb into DST
    cb_wait_front(out_cb_id, in_ntiles_c);

    tile_regs_acquire();

    // Add bias using broadcast row addition
    add_bcast_rows_init_short(out_cb_id, bias_cb_id);
    for (uint32_t i = 0; i < in_ntiles_c; ++i) {
        add_tiles_bcast_rows(out_cb_id, bias_cb_id, i, i, i);
    }

    // Apply activation function on tiles with bias
#ifdef SFPU_OP_FUNC_ACTIVATION
    for (uint32_t i = 0; i < in_ntiles_c; ++i) {
        SFPU_OP_FUNC_ACTIVATION
    }
#endif

    tile_regs_commit();

    // Remove tiles from out_cb (we'll repack them)
    cb_pop_front(out_cb_id, in_ntiles_c);

    // Pack back to out_cb
    tile_regs_wait();
    cb_reserve_back(out_cb_id, in_ntiles_c);
    for (uint32_t i = 0; i < in_ntiles_c; ++i) {
        pack_tile(i, out_cb_id);
    }
    tile_regs_release();
}
// ============ END BIAS + ACTIVATION ============
// Note: When !has_bias, activation was already applied per-stick, no extra work needed here

cb_push_back(out_cb_id, in_ntiles_c);
```

### Flow Diagram

```
NO BIAS (unchanged - best performance):
  reduce → activation (per stick) → pack_untilize_dest → collect 32 sticks → tilize → out_cb

WITH BIAS (post-tilization):
  reduce → pack_untilize_dest → collect 32 sticks → tilize → out_cb
       → load from out_cb → add_bias → activation → pack back to out_cb
```

**Key**: No bias path has zero overhead. Bias path only pays the extra load/pack cost when bias is actually used.

### Mathematical Correctness

Standard neural network computation: `output = activation(conv_result + bias)`

- Bias must be added BEFORE activation (e.g., ReLU)
- By moving activation to post-tilization, we maintain correct order:
  1. Reduction (depthwise conv)
  2. Tilization (collect 32 sticks)
  3. Bias addition
  4. Activation

### Performance Analysis

| Approach | Operations per 32 sticks |
|----------|-------------------------|
| Per-stick bias | 32 × N × (pack + add + unpack) |
| Post-tilization | N × (load + add + activation + pack) |

For N=2 channel tiles: **32x speedup**

### Consideration: ROW_MAJOR Output Path

For ROW_MAJOR output (no tilization), bias must still be added per-stick since there's no tile accumulation. This path is less common for depthwise conv but should be handled:

```cpp
// ROW_MAJOR path (lines 435-449) - keep per-stick bias addition
if constexpr (!is_output_tiled) {
    // ... existing pack_untilize_dest code ...

    // For ROW_MAJOR, bias addition happens per-stick (less efficient but necessary)
    // This path typically not used for depthwise conv with TILE output
}
```

### How to Test

1. Run depthwise conv test with bias=True and activation (e.g., ReLU)
2. Compare output against PyTorch reference: `F.relu(F.conv2d(..., bias=bias))`
3. Verify numerical accuracy within tolerance

### Key Takeaways

- **No bias**: Activation stays in per-stick loop (original behavior, zero overhead)
- **With bias**: Activation moves to post-tilization (after bias addition)
- `if constexpr (!has_bias)` guards per-stick activation
- `if constexpr (has_bias)` guards post-tilization bias+activation block
- `add_tiles_bcast_rows` loads from `out_cb`, broadcasts bias rows, stores to DST
- 32x more efficient than per-stick bias approach
- Maintains correct mathematical order: conv → bias → activation

---

## Step 5: Update shared_variables_t and Runtime Argument Overrides

### What Needs to Be Done

Update the program factory's shared variables and override functions to handle bias tensor properly during execution.

### Location

File: `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_depthwise_program_factory.hpp`
File: `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_depthwise_program_factory.cpp`

### Code Changes

1. **Update shared_variables_t** (in .hpp):
```cpp
struct shared_variables_t {
    std::vector<CoreCoord> cores_vec;
    tt::tt_metal::KernelHandle reader_id{};
    tt::tt_metal::KernelHandle writer_id{};
    tt::tt_metal::KernelHandle compute_id{};
    tt::tt_metal::CBHandle cb_input{};
    tt::tt_metal::CBHandle cb_output{};
    tt::tt_metal::CBHandle cb_weight{};
    tt::tt_metal::CBHandle cb_mul{};
    tt::tt_metal::CBHandle cb_bias{};  // NEW: Add bias CB handle
    bool has_bias = false;
};
```

2. **Store bias CB in shared_vars** (in .cpp, in create function):
```cpp
shared_vars.cb_bias = has_bias ? bias_cb_id : 0;
```

3. **Update override_runtime_arguments** if bias buffer address needs updating:
```cpp
void Conv2dDepthwiseProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {

    auto& shared_vars = cached_program.shared_variables;

    if (shared_vars.has_bias && tensor_args.bias.has_value()) {
        // Update bias buffer address if needed
        // Similar pattern to how weights are handled
    }
}
```

### Why This Approach

- Shared variables track CB handles for runtime updates
- Override function handles tensor address changes between invocations
- Follows existing pattern from conventional conv2d

### How to Test

Run test multiple times to verify cached program reuse works correctly.

### Key Takeaways

- Shared variables enable program caching with runtime updates
- Bias CB handle needs to be tracked for runtime address updates
- Follow existing weight handling pattern

---

## Step 6: Integration Testing with Bias

### What Needs to Be Done

Create or update tests to verify depthwise conv2d with bias works correctly.

### Test Cases

1. **Basic depthwise conv2d with bias**
   - Input: [1, 8, 8, 4], Weight: [3, 3, 4], Bias: [1, 1, 1, 4]
   - Verify output matches PyTorch reference

2. **Different sharding modes**
   - Height sharded with bias
   - Width sharded with bias
   - Block sharded with bias

3. **Different data types**
   - bf16 input/weights/bias
   - bfp8 weights with bf16 bias

4. **Edge cases**
   - Single channel
   - Large channel count (>32)
   - Non-tile-aligned channel count

### Test File Location

`tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py`

### Example Test

```python
def test_depthwise_conv2d_with_bias():
    device = ttnn.open_device(0)

    # Create input tensor [N, H, W, C]
    input_shape = [1, 8, 8, 4]
    input_tensor = torch.randn(input_shape)

    # Depthwise weight [kernel_h, kernel_w, 1, C] (groups=C)
    weight_shape = [3, 3, 1, 4]
    weight_tensor = torch.randn(weight_shape)

    # Bias [C]
    bias_shape = [4]
    bias_tensor = torch.randn(bias_shape)

    # PyTorch reference
    torch_output = F.conv2d(
        input_tensor.permute(0, 3, 1, 2),  # NHWC -> NCHW
        weight_tensor.permute(3, 2, 0, 1),  # HWiC -> CoiHW
        bias=bias_tensor,
        groups=4,
        padding=1
    ).permute(0, 2, 3, 1)  # NCHW -> NHWC

    # TTNN execution
    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, device=device)
    tt_weight = ttnn.from_torch(weight_tensor, dtype=ttnn.bfloat16, device=device)
    tt_bias = ttnn.from_torch(bias_tensor.reshape(1, 1, 1, 4), dtype=ttnn.bfloat16, device=device)

    tt_output = ttnn.conv2d(
        tt_input, tt_weight, bias=tt_bias,
        groups=4, padding=(1, 1), stride=(1, 1)
    )

    # Compare
    tt_output_torch = ttnn.to_torch(tt_output)
    assert torch.allclose(tt_output_torch, torch_output, atol=0.1)

    ttnn.close_device(device)
```

### How to Test

```bash
pytest tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py::test_depthwise_conv2d_with_bias -v
```

### Key Takeaways

- Test both functional correctness and numerical accuracy
- Cover all sharding modes
- Verify backward compatibility (no bias still works)

---

## Step 7: Edge Cases and Validation

### What Needs to Be Done

Add proper validation and handle edge cases.

### Validation Checks

1. **Bias shape validation** (in conv2d.cpp or prepare_conv_bias):
```cpp
if (is_depthwise && bias.has_value()) {
    TT_FATAL(bias.value().logical_shape()[3] == output_channels,
             "Depthwise conv bias must have {} channels, got {}",
             output_channels, bias.value().logical_shape()[3]);
}
```

2. **Bias dtype validation**:
```cpp
TT_ASSERT(bias_tensor.dtype() == weight_dtype,
          "Bias tensor should match weights dtype for depthwise conv");
```

### Edge Cases to Handle

1. **Non-tile-aligned channels**: Bias padding must match output channel padding
2. **Width sharding**: Each core needs its bias shard
3. **Block sharding**: Bias multicast from row-0 cores (like weights)

### Potential Issues

1. **Activation function ordering**: ✅ RESOLVED - Activation moved to post-tilization, applied after bias
2. **FP accumulation**: May need fp32_dest_acc_en for numerical stability
3. **Bias multicast**: For multi-core cases, bias may need multicast pattern
4. **ROW_MAJOR output**: Less efficient path - no tilization means per-stick bias addition

### How to Test

Create targeted unit tests for each edge case.

### Key Takeaways

- Validation prevents silent numerical errors
- Edge cases often reveal sharding/layout bugs
- Follow conventional conv2d patterns for consistency

---

## Implementation Order and Dependencies

```
Step 1 (CB allocation)
    ↓
Step 2 (Kernel args)
    ↓
Step 3 (Reader kernel) ←→ Step 4 (Compute kernel)
    ↓                           ↓
Step 5 (Runtime overrides)
    ↓
Step 6 (Integration tests)
    ↓
Step 7 (Edge cases)
```

Steps 3 and 4 can be developed in parallel but must both be complete before integration testing.

---

## Files to Modify

| File | Changes |
|------|---------|
| `conv2d_op_depthwise_program_factory.cpp` | CB allocation, kernel args, shared_vars |
| `conv2d_op_depthwise_program_factory.hpp` | shared_variables_t struct |
| `reader_pool_2d.cpp` | Bias reading from DRAM |
| `compute_pool_2d.cpp` | Bias addition after reduction |
| `test_conv2d.py` | Test cases for depthwise + bias |

---

## Reference: Key Code Patterns from Conventional Conv2D

### CB Allocation Pattern
```cpp
// From conv2d_op_program_factory_common.cpp:259-264
cb_info.emplace_back(CBInfo{
    .name = Conv2dCb::BIAS,
    .num_pages = enable_bias ? per_core_out_matrix_width_ntiles : 0,
    .page_size = bias_tile_size,
    .data_format = bias_df});
```

### Bias Addition Pattern (Compute Kernel)
```cpp
// From conv_bmm_tilize.cpp:538-551
reconfig_data_format(in1_cb_id, matmul_partials_cb, mm_in0_cb_id, bias_cb_id);
add_bcast_rows_init_short(matmul_partials_cb, bias_cb_id);
cb_wait_front(bias_cb_id, bias_ntiles_w);

for (uint32_t w = 0; w < out_subblock_w; ++w) {
    add_tiles_bcast_rows(matmul_partials_cb, bias_cb_id, i, bcast_tile_i, i);
    ++bcast_tile_i;
    ++i;
}
```

### Bias Reading Pattern (Writer Kernel)
```cpp
// From writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp:373-388
cb_reserve_back(bias_cb_id, bias_ntiles);
uint32_t bias_l1_addr = get_write_ptr(bias_cb_id);

for (uint32_t bias_tile = bias_tile_offset; bias_tile < bias_tile_offset + bias_ntiles; ++bias_tile) {
    noc_async_read_tile(bias_tile, s_bias, bias_l1_addr);
    bias_l1_addr += bias_pagesize;
}
noc_async_read_barrier();
cb_push_back(bias_cb_id, bias_ntiles);
```
