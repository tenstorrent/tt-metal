# Implementation Plan: `config_tensors_in_dram` for Pool Operations

## Overview

This plan describes how to implement the `config_tensors_in_dram` feature for pool operations (`ttnn.max_pool2d`, `ttnn.avg_pool2d`, and similar) to match the existing implementation in `ttnn.conv2d`.

### Problem Statement

L1_SMALL is persistent storage on Tenstorrent devices that gets quickly exhausted when running large CNNs with many layers. Currently, pool operations always store their config tensors (reader indices, scalar configs) in L1_SMALL, which contributes to this exhaustion.

### Solution

Add a `config_tensors_in_dram` optional argument to pool operations that, when enabled, stores config tensors in DRAM instead of L1_SMALL.

---

## Background: How `config_tensors_in_dram` Works in Conv2D

### Key Files and Mechanisms

1. **Configuration**: Simple bool parameter passed through the API (default `false`)

2. **Tensor Creation** (`sliding_window.cpp:997-1015`):
   - `construct_on_host_config_tensor`: When `store_in_dram=true`, does NOT replicate config across grid (factor=1)
   - `move_config_tensor_to_device`:
     - If `store_in_dram=true`: Uses `TensorMemoryLayout::INTERLEAVED` with `BufferType::DRAM`
     - If `store_in_dram=false`: Uses `TensorMemoryLayout::HEIGHT_SHARDED` with `BufferType::L1_SMALL`

3. **Program Factory** (`conv2d_op_sharded_program_factory.cpp:835-840`):
   - Adds `CONFIG_TENSOR_IN_DRAM` define to kernels
   - Passes buffer address and page size as compile-time arguments

4. **Reader Kernel Support** (`conv_reader_common.hpp:336-351`):
   ```cpp
   template <uint32_t dram_addr_index, uint32_t page_size_index, uint32_t tensor_args_index, uint32_t cb_reader_index>
   void load_config_tensor_if_in_dram(uint32_t core_index) {
   #ifdef CONFIG_TENSOR_IN_DRAM
       constexpr uint32_t config_dram_addr = get_compile_time_arg_val(dram_addr_index);
       constexpr uint32_t config_page_size = get_compile_time_arg_val(page_size_index);
       const auto config_tensor_args = TensorAccessorArgs<tensor_args_index>();
       const auto config_accessor = TensorAccessor(config_tensor_args, config_dram_addr, config_page_size);
       uint64_t src_noc_addr = get_noc_addr(core_index, config_accessor);

       noc_async_read(src_noc_addr, get_write_ptr(cb_reader_index), config_page_size);
       noc_async_read_barrier();
       cb_push_back(cb_reader_index, 1);
   #endif
   }
   ```

5. **Reader Kernel Usage** (`reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp:88-91`):
   ```cpp
   if constexpr (split_reader_enabled) {
   #ifdef CONFIG_TENSOR_IN_DRAM
       cb_wait_front(cb_reader_indices, 1);
   #endif
   }
   ```

---

## Implementation Steps

### Step 1: Update Pool2D Operation Attributes

**File**: `ttnn/cpp/ttnn/operations/pool/generic/device/pool_op.hpp`

Add `config_tensors_in_dram` to `operation_attributes_t`:

```cpp
struct operation_attributes_t {
    sliding_window::SlidingWindowConfig sliding_window_config_;
    Pool2DType pool_type_;
    DataType output_dtype_;
    Layout output_layout_;
    MemoryConfig memory_config_;
    std::optional<DeviceComputeKernelConfig> compute_kernel_config_;
    bool count_include_pad_;
    std::optional<int32_t> divisor_override_;
    bool return_indices_;
    uint32_t memory_used;
    bool config_tensors_in_dram_ = false;  // ADD THIS
};
```

Update the `invoke` function signature:

```cpp
static std::tuple<operation_attributes_t, tensor_args_t> invoke(
    const Tensor& input_tensor,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    Pool2DType pool_type,
    DataType output_dtype,
    Layout output_layout,
    MemoryConfig memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    bool count_include_pad,
    std::optional<int32_t> divisor_override,
    bool return_indices,
    uint32_t memory_used,
    bool config_tensors_in_dram = false);  // ADD THIS PARAMETER
```

### Step 2: Update Generic Pools High-Level API

**File**: `ttnn/cpp/ttnn/operations/pool/generic/generic_pools.hpp`

Update function signatures for `MaxPool2DOp` and `AvgPool2DOp`:

```cpp
struct MaxPool2DOp {
    static std::vector<Tensor> invoke(
        const Tensor& input_tensor,
        // ... existing parameters ...
        const Layout output_layout = Layout::ROW_MAJOR,
        bool config_tensors_in_dram = false);  // ADD THIS
};

struct AvgPool2DOp {
    static Tensor invoke(
        const Tensor& input_tensor,
        // ... existing parameters ...
        const Layout output_layout = Layout::ROW_MAJOR,
        bool config_tensors_in_dram = false);  // ADD THIS
};
```

### Step 3: Update Generic Pools Implementation

**File**: `ttnn/cpp/ttnn/operations/pool/generic/generic_pools.cpp`

#### 3a. Update `pool2d_invoke` function signature (line ~33):

```cpp
static std::vector<Tensor> pool2d_invoke(
    // ... existing parameters ...
    const Layout output_layout = Layout::ROW_MAJOR,
    bool config_tensors_in_dram = false) {  // ADD THIS
```

#### 3b. Pass `config_tensors_in_dram` to halo operation (line ~250):

```cpp
Tensor haloed_tensor = ttnn::halo(
    input_tensor_sharded,
    sliding_window_config,
    get_bf16_pool_init_value(pool_type),
    false,
    parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
    input_tensor_sharded.memory_config(),
    is_out_tiled,
    config_tensors_in_dram);  // ADD THIS PARAMETER
```

#### 3c. Pass to prim::pool2d (line ~273):

```cpp
std::vector<Tensor> output_tensors = ttnn::prim::pool2d(
    haloed_tensor,
    sliding_window_config,
    pool_type,
    dtype,
    output_layout,
    out_memory_config,
    compute_kernel_config,
    count_include_pad,
    divisor_override,
    return_indices,
    pre_allocate_size,
    config_tensors_in_dram);  // ADD THIS PARAMETER
```

#### 3d. Update MaxPool2DOp::invoke and AvgPool2DOp::invoke to pass the parameter.

### Step 4: Update Program Factory for Config Tensor Storage

**File**: `ttnn/cpp/ttnn/operations/pool/generic/device/pool_multi_core_program_factory.cpp`

#### 4a. Accept `config_tensors_in_dram` in function parameters

Update `pool2d_multi_core_sharded_with_halo_v2_impl_new` signature and `Pool2D::MultiCore::create` to pass through the flag.

#### 4b. Modify reader indices tensor creation (around line 893-895):

```cpp
// Current code:
Tensor reader_indices = sliding_window::construct_on_host_config_tensor(top_left_indices, parallel_config);
Tensor reader_indices_on_device =
    sliding_window::move_config_tensor_to_device(reader_indices, parallel_config, is_block_sharded, input.device());

// New code:
Tensor reader_indices = sliding_window::construct_on_host_config_tensor(
    top_left_indices, parallel_config, config_tensors_in_dram);
Tensor reader_indices_on_device = sliding_window::move_config_tensor_to_device(
    reader_indices, parallel_config, is_block_sharded, input.device(), config_tensors_in_dram);
```

#### 4c. Modify scalar config tensor creation (around line 567-568):

```cpp
// Current code:
const MemoryConfig memory_config{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1_SMALL, config_shard_spec};

// New code:
MemoryConfig memory_config;
if (config_tensors_in_dram) {
    memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
} else {
    memory_config = MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1_SMALL, config_shard_spec};
}
```

#### 4d. Add kernel defines and compile-time args when `config_tensors_in_dram=true`:

```cpp
std::map<std::string, std::string> reader_defines;

if (config_tensors_in_dram) {
    reader_defines["CONFIG_TENSOR_IN_DRAM"] = "1";

    // Add compile-time args for DRAM tensor access
    // Reader indices tensor
    reader0_ct_args.push_back(reader_indices_storage.get_buffer()->address());
    reader0_ct_args.push_back(reader_indices_storage.get_buffer()->page_size());

    // Scalar config tensor (if applicable)
    if (!one_scalar_per_core && scalar_config_storage.is_allocated()) {
        reader0_ct_args.push_back(scalar_config_storage.get_buffer()->address());
        reader0_ct_args.push_back(scalar_config_storage.get_buffer()->page_size());
    }
}

auto reader0_config = tt::tt_metal::DataMovementConfig{
    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
    .noc = tt::tt_metal::NOC::RISCV_0_default,
    .compile_args = reader0_ct_args,
    .defines = reader_defines};  // ADD defines
```

### Step 5: Update Pool Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/reader_pool_2d.cpp`

#### 5a. Add DRAM loading logic at the beginning of `kernel_main()`:

Add new compile-time argument indices for DRAM tensor addresses (after existing args):

```cpp
// Add after line 426:
#ifdef CONFIG_TENSOR_IN_DRAM
constexpr uint32_t reader_indices_dram_addr = get_compile_time_arg_val(46);
constexpr uint32_t reader_indices_page_size = get_compile_time_arg_val(47);
constexpr uint32_t config_dram_addr = get_compile_time_arg_val(48);  // only if !one_scalar_per_core
constexpr uint32_t config_page_size = get_compile_time_arg_val(49);  // only if !one_scalar_per_core
#endif
```

#### 5b. Add DRAM read logic before accessing config tensors:

```cpp
// Add before line 497 (before accessing reader indices):
#ifdef CONFIG_TENSOR_IN_DRAM
    // Load reader indices from DRAM into local CB
    {
        // Get core index for tensor accessor
        uint32_t core_x = get_arg_val<uint32_t>(0);  // May need adjustment based on how core index is passed
        uint32_t core_y = get_arg_val<uint32_t>(1);

        // Read from DRAM
        uint64_t src_noc_addr = get_noc_addr(reader_indices_dram_addr);  // Simplified - may need TensorAccessor
        noc_async_read(src_noc_addr, get_write_ptr(in_reader_indices_cb_id), reader_indices_page_size);
        noc_async_read_barrier();
        cb_push_back(in_reader_indices_cb_id, 1);
    }
    cb_wait_front(in_reader_indices_cb_id, 1);
#endif

    const uint32_t in_l1_read_base_addr = get_read_ptr(in_shard_cb_id);
    uint32_t reader_indices_l1_addr = get_read_ptr(in_reader_indices_cb_id);
```

#### 5c. Add DRAM read logic for scalar config tensor (when `!one_scalar_per_core`):

```cpp
// Add before line 511 (before accessing config_ptr):
#ifdef CONFIG_TENSOR_IN_DRAM
    if constexpr (!one_scalar_per_core) {
        // Load scalar config from DRAM into local CB
        uint64_t config_noc_addr = get_noc_addr(config_dram_addr);
        noc_async_read(config_noc_addr, get_write_ptr(config_cb_id), config_page_size);
        noc_async_read_barrier();
        cb_push_back(config_cb_id, 1);
        cb_wait_front(config_cb_id, 1);
    }
#endif
```

**Alternative approach**: Create a common header file similar to `conv_reader_common.hpp` with a `load_config_tensor_if_in_dram` template function that can be shared between conv and pool operations.

### Step 6: Update Python Bindings

**File**: `ttnn/cpp/ttnn/operations/pool/generic/generic_pools_pybind.cpp`

#### 6a. Update max_pool2d binding (around line 54-117):

```cpp
py::arg("output_layout") = Layout::ROW_MAJOR,
py::arg("config_tensors_in_dram") = false  // ADD THIS
```

Also update the lambda to accept and pass the new parameter.

#### 6b. Update avg_pool2d binding (around line 153-220):

```cpp
py::arg("output_layout") = Layout::ROW_MAJOR,
py::arg("config_tensors_in_dram") = false  // ADD THIS
```

Also update the lambda to accept and pass the new parameter.

#### 6c. Update docstrings:

Add documentation for the new parameter in both operations:

```cpp
R"doc(
    ...
    Keyword Args:
        ...
        config_tensors_in_dram (bool, optional): If true, config tensors are stored in DRAM
            instead of L1_SMALL. L1_SMALL is persistent storage and gets quickly used up for
            large CNNs. Defaults to `False`.
    ...
)doc"
```

### Step 7: Verify Halo Operation Interface

**File**: `ttnn/cpp/ttnn/operations/sliding_window/halo/halo.hpp` and `halo.cpp`

Verify that `ttnn::halo` accepts `config_tensors_in_dram` parameter. Based on `halo_device_operation_types.hpp:22`, the underlying operation already supports this. The high-level halo function signature should be checked to ensure it passes this parameter through.

---

## Files Summary

| File | Changes |
|------|---------|
| `pool_op.hpp` | Add `config_tensors_in_dram` to `operation_attributes_t`, update `invoke` signature |
| `pool_multi_core_program_factory.cpp` | Use flag for reader indices and scalar config tensor storage, add kernel defines and compile-time args |
| `generic_pools.hpp` | Update `MaxPool2DOp` and `AvgPool2DOp` function signatures |
| `generic_pools.cpp` | Pass flag to halo and pool2d operations |
| `generic_pools_pybind.cpp` | Expose parameter to Python, update docstrings |
| `reader_pool_2d.cpp` | Add `#ifdef CONFIG_TENSOR_IN_DRAM` blocks to load tensors from DRAM |
| `halo.hpp`/`halo.cpp` | Verify high-level interface accepts flag (likely already supported) |

---

## Reader Kernel Changes Detail

The pool reader kernel (`reader_pool_2d.cpp`) needs modifications similar to conv2d reader kernels:

### Current Flow (L1_SMALL):
1. Config tensors are sharded in L1_SMALL
2. Reader kernel directly accesses them via `get_read_ptr(cb_id)`

### New Flow (DRAM):
1. Config tensors are interleaved in DRAM
2. Reader kernel must:
   - Read from DRAM into local CB using `noc_async_read`
   - Wait for read to complete with `noc_async_read_barrier`
   - Push to CB with `cb_push_back`
   - Wait for data with `cb_wait_front` before accessing

### Key Differences from Conv2D:
- Pool reader accesses **two** config tensors:
  1. `in_reader_indices_cb_id` - Reader indices (always used)
  2. `config_cb_id` - Scalar config (only when `!one_scalar_per_core`)
- Both need DRAM loading when `CONFIG_TENSOR_IN_DRAM` is defined

---

## Testing Recommendations

1. **Unit Tests**: Add tests in `tests/ttnn/unit_tests/operations/pool/` to verify:
   - Default behavior unchanged (config tensors in L1_SMALL)
   - With `config_tensors_in_dram=True`, tensors are allocated in DRAM
   - Numerical correctness unchanged

2. **Memory Tests**: Verify L1_SMALL usage decreases when feature is enabled

3. **Performance Tests**: Measure any performance impact of DRAM vs L1_SMALL storage

4. **Integration Tests**: Test with large CNN models that previously exhausted L1_SMALL

---

## Example Usage (After Implementation)

```python
import ttnn

# Use DRAM for config tensors (recommended for large CNNs)
output = ttnn.max_pool2d(
    input_tensor,
    batch_size=batch,
    input_h=height,
    input_w=width,
    channels=channels,
    kernel_size=[2, 2],
    stride=[2, 2],
    padding=[0, 0],
    dilation=[1, 1],
    config_tensors_in_dram=True  # NEW PARAMETER
)

output = ttnn.avg_pool2d(
    input_tensor,
    batch_size=batch,
    input_h=height,
    input_w=width,
    channels=channels,
    kernel_size=[2, 2],
    stride=[2, 2],
    padding=[0, 0],
    config_tensors_in_dram=True  # NEW PARAMETER
)
```

---

## Notes

- The feature is already implemented in the underlying sliding_window infrastructure
- Pool operations call `ttnn::halo` which already supports this feature internally
- The main work is threading the parameter through all layers of the API and adding DRAM read support to the pool reader kernel
- The pool reader kernel changes mirror the pattern in `conv_reader_common.hpp:load_config_tensor_if_in_dram`
