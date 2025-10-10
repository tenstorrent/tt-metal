# Plan: Replace `estimate_halo_output_elems` with Precise Calculation

## Overview
Currently, the halo operation uses an **estimation** function `estimate_halo_output_elems()` to calculate the memory requirements for the output tensor per core. This estimation can be conservative (over-allocate) or inaccurate in certain edge cases. The goal is to replace this with a **precise calculation** that leverages the exact shard shape information already computed during the halo operation.

## Current State Analysis

### Where Estimation is Used
The estimation function `estimate_halo_output_elems()` is currently used in two places:

1. **`ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp`** (2 occurrences):
   - Line ~1380: During L1 memory allocation size calculation for conv operations
   - Line ~2390: During conv2d slice configuration to estimate halo output size

### Current Estimation Logic
Located in `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp`, the function uses approximations:

```cpp
uint32_t estimate_halo_output_elems(
    std::array<uint32_t, 2> halo_input_shard_shape,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> dilation,
    std::array<uint32_t, 4> padding)
```

**Estimation approach:**
- Calculates approximate shard height based on input dimensions
- Adds conservative padding for kernel overlap and dilation
- Includes batch boundary multipliers for multi-batch scenarios
- Returns: `approx_max_halo_num_sticks * halo_input_shard_shape[1]`

### Precise Calculation Already Exists
In `ttnn/cpp/ttnn/operations/sliding_window/halo/device/halo_device_operation.cpp` **lines 278-287**, the exact value is already computed:

```cpp
auto sliding_window_hash = config.get_hash();
if (!HaloDeviceOperation::sliding_window_max_out_nsticks_per_core.contains(sliding_window_hash)) {
    auto op_trace_metadata = sliding_window::generate_op_trace_metadata(config);
    auto shard_boundaries = sliding_window::generate_shard_boundaries(config, op_trace_metadata);
    HaloDeviceOperation::sliding_window_max_out_nsticks_per_core.emplace(
        sliding_window_hash, sliding_window::generate_max_out_nsticks_per_core(shard_boundaries));
}

uint32_t max_out_nsticks_per_core =
    HaloDeviceOperation::sliding_window_max_out_nsticks_per_core.at(sliding_window_hash);
```

**Precise calculation in `sliding_window.cpp` lines 391-398:**
```cpp
uint32_t generate_max_out_nsticks_per_core(const std::vector<ShardBoundary>& shard_boundaries) {
    uint32_t max_out_nsticks_per_core = 0;
    for (auto [_, in_shard] : shard_boundaries) {
        auto [in_start, in_end] = in_shard;
        max_out_nsticks_per_core = std::max(max_out_nsticks_per_core, in_end - in_start + 1);
    }
    return max_out_nsticks_per_core;
}
```

## Key Insight
The **precise shard shape** calculation (`max_out_nsticks_per_core`) is already available in the halo device operation. This value represents the exact maximum number of output sticks (rows) per core across all cores, taking into account:
- Actual sliding window configuration
- Precise shard boundaries
- Exact padding, stride, dilation, and kernel size effects
- Per-core distribution of work

## Important Clarification: Why Does Stride Matter for Halo?

**Question**: If the halo operation is just gathering overlapping data, why does stride matter?

**Answer**: The stride parameter is critical because it determines **which input positions** are accessed by the sliding window operation. Here's why:

1. **Op Trace Metadata Generation** (sliding_window.cpp:293-295):
   ```cpp
   uint32_t input_index = (b * padded_input_h * padded_input_w) +
                          (h * config.stride_hw.first * padded_input_w) +
                          (w * config.stride_hw.second);
   ```

   The stride determines the spacing between consecutive output positions. For stride=2, the halo only needs to gather every 2nd row/column of input data plus the kernel overlap region.

2. **Impact on Shard Boundaries**:
   - With stride=1: Every input position is accessed → larger halo region
   - With stride=2: Only every 2nd position is accessed → potentially smaller halo region
   - The shard boundaries calculation uses `op_trace_metadata` to determine the exact input range needed for each output shard

3. **Example**:
   - 3x3 kernel, stride=1: output[0,0] needs input[0:2, 0:2], output[0,1] needs input[0:2, 1:3]
   - 3x3 kernel, stride=2: output[0,0] needs input[0:2, 0:2], output[0,1] needs input[0:2, 2:4]

   The stride affects which input elements are gathered, thus affecting the precise shard size.

**Conclusion**: Always use the actual convolution stride when constructing `SlidingWindowConfig` for halo calculations. The stride is NOT `{1,1}` unless the convolution itself has stride `{1,1}`.

## Implementation Plan

### Phase 1: Refactor Precise Calculation into Reusable Function
**File:** `ttnn/cpp/ttnn/operations/sliding_window/sliding_window.hpp`

1. Add new public function declaration:
```cpp
uint32_t calculate_precise_halo_output_elems(
    const SlidingWindowConfig& config,
    const std::array<uint32_t, 2>& shard_shape);
```

**File:** `ttnn/cpp/ttnn/operations/sliding_window/sliding_window.cpp`

2. Implement the function that encapsulates the precise calculation:
```cpp
uint32_t calculate_precise_halo_output_elems(
    const SlidingWindowConfig& config,
    const std::array<uint32_t, 2>& shard_shape) {
    // Generate metadata for precise calculation
    auto op_trace_metadata = generate_op_trace_metadata(config);
    auto shard_boundaries = generate_shard_boundaries(config, op_trace_metadata);

    // Get precise max sticks per core
    uint32_t max_out_nsticks_per_core = generate_max_out_nsticks_per_core(shard_boundaries);

    // Return total elements: max_sticks * stick_width
    return max_out_nsticks_per_core * shard_shape[1];
}
```

### Phase 2: Replace Estimation Calls with Precise Calculation

#### Location 1: Conv2D Utils - L1 Memory Calculation
**File:** `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp` (~line 1380)

**Current code:**
```cpp
auto halo_input_shard_shape = halo_input_memory_config.shard_spec().value().shape;
uint32_t approx_input_size_per_core = estimate_halo_output_elems(
    halo_input_shard_shape, batch_size, input_height, input_width, kernel_size, dilation, padding);
```

**Replace with:**
```cpp
auto halo_input_shard_shape = halo_input_memory_config.shard_spec().value().shape;

// Create SlidingWindowConfig from conv parameters
// Note: The halo operation itself doesn't have a stride parameter - it always gathers with stride=1
// The stride used here is from the convolution operation and affects which input elements
// are needed for the sliding window computation (via generate_op_trace_metadata)
SlidingWindowConfig halo_config;
halo_config.batch_size = batch_size;
halo_config.input_hw = {input_height, input_width};
halo_config.window_hw = {kernel_size[0], kernel_size[1]};
halo_config.stride_hw = {stride[0], stride[1]};  // Use actual conv stride
halo_config.padding = padding;
halo_config.dilation_hw = {dilation[0], dilation[1]};
halo_config.num_cores_nhw = input_parallel_config.num_cores_nhw;
halo_config.num_cores_c = input_parallel_config.num_cores_c;
halo_config.core_range_set = input_parallel_config.grid;

uint32_t precise_input_size_per_core =
    sliding_window::calculate_precise_halo_output_elems(halo_config, halo_input_shard_shape);
```

#### Location 2: Conv2D Slice Configuration
**File:** `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp` (~line 2390)

**Current code:**
```cpp
uint32_t approx_max_halo_bytes = estimate_halo_output_elems(
    shard_shape,
    params.batch_size,
    input_slice_height,
    input_slice_width,
    params.kernel_size,
    params.dilation,
    params.padding_n4) * input_datum_size;
```

**Replace with:**
```cpp
// Create SlidingWindowConfig from slice parameters
SlidingWindowConfig slice_halo_config;
slice_halo_config.batch_size = params.batch_size;
slice_halo_config.input_hw = {input_slice_height, input_slice_width};
slice_halo_config.window_hw = {params.kernel_size[0], params.kernel_size[1]};
slice_halo_config.stride_hw = {params.stride[0], params.stride[1]};  // Use actual conv stride
slice_halo_config.padding = params.padding_n4;
slice_halo_config.dilation_hw = {params.dilation[0], params.dilation[1]};
slice_halo_config.num_cores_nhw = /* extract from parallel config */;
slice_halo_config.num_cores_c = /* extract from parallel config */;
slice_halo_config.core_range_set = /* extract from parallel config */;

uint32_t precise_max_halo_bytes =
    sliding_window::calculate_precise_halo_output_elems(slice_halo_config, shard_shape) * input_datum_size;
```

### Phase 3: Update Halo Device Operation (Optional Optimization)
**File:** `ttnn/cpp/ttnn/operations/sliding_window/halo/device/halo_device_operation.cpp`

The halo operation already computes this value correctly. Consider refactoring to use the new public function for consistency:

```cpp
// Replace lines 280-283 with:
if (!HaloDeviceOperation::sliding_window_max_out_nsticks_per_core.contains(sliding_window_hash)) {
    uint32_t precise_max_nsticks = sliding_window::calculate_precise_halo_output_elems(
        config, {input_tensor.memory_config().shard_spec()->shape[0],
                 input_tensor.memory_config().shard_spec()->shape[1]}) /
        input_tensor.memory_config().shard_spec()->shape[1];

    HaloDeviceOperation::sliding_window_max_out_nsticks_per_core.emplace(
        sliding_window_hash, precise_max_nsticks);
}
```

### Phase 4: Remove Deprecated Function
**Files to update:**
1. `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.hpp` - Remove declaration
2. `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp` - Remove implementation

**Action:** Delete the `estimate_halo_output_elems()` function entirely once all call sites are migrated.

## Benefits of This Approach

1. **Accuracy**: Eliminates estimation errors and conservative over-allocation
2. **Memory Efficiency**: Uses exact memory requirements, reducing L1 memory pressure
3. **Consistency**: Uses the same calculation logic that the halo operation already trusts
4. **Maintainability**: Single source of truth for halo output size calculations
5. **Robustness**: Handles edge cases (batch boundaries, sharding patterns) correctly

## Testing Strategy

1. **Unit Tests**: Create tests comparing old estimation vs. new precise calculation
2. **Integration Tests**: Run existing conv2d tests to ensure no regressions
3. **Memory Tests**: Verify L1 memory usage is within bounds
4. **Performance Tests**: Ensure no performance degradation
5. **Edge Cases**: Test with:
   - Multi-batch inputs
   - Various kernel sizes and dilations
   - Different sharding schemes (height, width, block)
   - Boundary conditions (small/large inputs)

## Implementation Order

1. ✅ Analyze current code and create plan (this document)
2. Add `calculate_precise_halo_output_elems()` to `sliding_window.hpp/.cpp`
3. Update first call site in conv2d_utils (L1 memory calculation)
4. Update second call site in conv2d_utils (slice config)
5. Run comprehensive tests
6. Remove deprecated `estimate_halo_output_elems()` function
7. Update documentation and comments

## Potential Challenges

1. **SlidingWindowConfig Construction**: Need to properly construct `SlidingWindowConfig` from conv parameters at call sites
2. **Parallel Config Extraction**: May need to pass or extract parallel config info to the call sites
3. **Performance**: The precise calculation involves generating metadata and boundaries - profile to ensure no significant overhead
4. **Caching**: Consider adding caching similar to what halo_device_operation does if called frequently

## Dependencies

- `ttnn/cpp/ttnn/operations/sliding_window/sliding_window.hpp`
- `ttnn/cpp/ttnn/operations/sliding_window/sliding_window.cpp`
- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.hpp`
- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp`
- `ttnn/cpp/ttnn/operations/sliding_window/halo/device/halo_device_operation.cpp`

## Notes

- The key value is at line **286** in `halo_device_operation.cpp`: `max_out_nsticks_per_core`
- This represents the maximum number of rows (height) per core in the output shard
- Multiply by shard width (`shard_shape[1]`) to get total elements per core
- The calculation in `generate_max_out_nsticks_per_core()` (lines 391-398 in `sliding_window.cpp`) is the ground truth

## Critical Implementation Detail: Stride Parameter

**IMPORTANT**: When creating the `SlidingWindowConfig` for halo calculation, you MUST use the **actual convolution stride**, not `{1, 1}`.

The stride affects:
1. Which input elements are accessed via `generate_op_trace_metadata()`
2. The resulting `shard_boundaries` that determine the precise halo region size
3. The final `max_out_nsticks_per_core` value

Using stride `{1, 1}` when the actual conv stride is different will result in **incorrect memory calculations**.
