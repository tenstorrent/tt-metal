# Program Factory Refactoring - Remaining Tasks

## Overview
This document tracks the remaining files in `ttnn/cpp/ttnn/operations` that contain multiple functions creating `tt::tt_metal::Program` objects. Each of these files needs refactoring to extract individual functions into separate files for better code organization and modularity.

## Current Status

### ✅ **COMPLETED** (8 files):
1. **topk_program_factory.cpp** - 2 functions → PR #25231
2. **argmax_program_factory.cpp** - 2 functions → PR #25232
3. **padded_slice_program_factory.cpp** - 2 functions → PR #25234
4. **reshard_program_factory.cpp** - 3 functions → PR #25235
5. **sort_program_factory.cpp** - 3 functions → PR #25236
6. **tilize_with_val_padding_program_factory.cpp** - 4 functions → PR #25237
7. **update_cache_op_multi_core.cpp** - 2 functions → PR #25238
8. **fill_cache_multi_core.cpp** - 2 functions → PR #25238

---

## 🔄 **PENDING REFACTORING** (11 files):

### **Data Movement Operations** (6 files):

#### 1. **gather_program_factory.cpp** - 2 functions
- **File**: `ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp`
- **Functions**:
  - `GatherProgramFactorySingleRowSingleCore::create()`
  - `GatherProgramFactorySingleRowMultiCore::create()`

#### 2. **concat_program_factory.cpp** - 6 functions 🔥
- **File**: `ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_program_factory.cpp`
- **Functions**:
  - `s2s_tiled_concat_two_tensors_height_multi_core()`
  - `s2s_rm_concat_two_tensors_height_multi_core()`
  - `s2s_concat_multi_core()`
  - `s2i_rm_concat_multi_core()`
  - `sharded_concat_multi_core()`
  - `concat_multi_core()`

#### 3. **slice_program_factory.cpp** - 4 functions
- **File**: `ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory.cpp`
- **Functions**:
  - `slice_rm_multi_core()`
  - `slice_rm_strided_single_core_n_dims()`
  - `slice_rm_multi_core_sharded()`
  - `slice_tile_multi_core()`

#### 4. **tilize_program_factory.cpp** - 4 functions
- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_program_factory.cpp`
- **Functions**:
  - `tilize_single_core()`
  - `tilize_multi_core_block()`
  - `tilize_multi_core_interleaved()`
  - `tilize_multi_core_sharded()`

#### 5. **transpose_program_factory.cpp** - 5 functions
- **File**: `ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_program_factory.cpp`
- **Functions**: (5 different transpose variants)

#### 6. **untilize_program_factory.cpp** - 6 functions 🔥
- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_program_factory.cpp`
- **Functions**:
  - `untilize_multi_core_sub_core_grids()`
  - `untilize_multi_core_parallelize_column()`
  - `untilize_multi_core_block()`
  - `untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical()`
  - `untilize_multi_core()`
  - `untilize_single_core()`

### **Normalization Operations** (2 files):

#### 7. **softmax_op_multi_core.cpp** - 2 functions
- **File**: `ttnn/cpp/ttnn/operations/normalization/softmax/device/multi_core/softmax_op_multi_core.cpp`
- **Functions**:
  - `scale_mask_softmax_multi_core()`
  - `scale_mask_softmax_sharded_multi_core()`

#### 8. **layernorm_op_multi_core.cpp** - 2 functions
- **File**: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/multi_core/layernorm_op_multi_core.cpp`
- **Functions**:
  - `layernorm_multi_core()`
  - `layernorm_multi_core_sharded()`

### **Eltwise Binary Operations** (1 file):

#### 9. **broadcast_height_and_width_multi_core_program_factory.cpp** - 2 functions
- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/broadcast_height_and_width_multi_core_program_factory.cpp`
- **Functions**:
  - `create()` method
  - `override_runtime_arguments()` method

### **Moreh Softmax Operations** (2 files):

#### 10. **softmax_c_large.cpp** - 2 functions
- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/softmax_c_large/softmax_c_large.cpp`
- **Functions**:
  - `create()` method
  - `override_runtime_arguments()` method

#### 11. **softmax_h_large.cpp** - 2 functions
- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/softmax_h_large/softmax_h_large.cpp`
- **Functions**:
  - `create()` method
  - `override_runtime_arguments()` method

---

## 🔍 **NEEDS FURTHER INVESTIGATION**

### **Potential Additional Files**:
The following categories likely contain more files with multiple Program-creating functions:

1. **Moreh Operations** - More softmax variants, softmax_backward operations
2. **Matmul Operations** - Various reuse and mcast variants
3. **Conv2D Operations** - Sharded program factories
4. **Experimental Operations** - CCL, transformer operations
5. **Pool Operations** - Upsample variants
6. **Eltwise Operations** - More binary and unary variants

### **Next Steps**:
1. Complete comprehensive search of all remaining files
2. Prioritize by complexity (files with 6+ functions first)
3. Focus on frequently used operations
4. Consider impact on build times and modularity

---

## 📊 **Impact Summary**

### **Current Status**:
- **Completed**: 8 files (20+ functions)
- **Pending**: 11 files (35+ functions)
- **Estimated Total**: 50+ functions across 19+ files

### **Most Complex Files** (6+ functions):
1. `concat_program_factory.cpp` - 6 functions
2. `untilize_program_factory.cpp` - 6 functions
3. `transpose_program_factory.cpp` - 5 functions

### **Benefits of Refactoring**:
- ✅ Better code organization and modularity
- ✅ Easier maintenance and testing
- ✅ Cleaner compilation units
- ✅ Reduced code duplication
- ✅ Improved build times

---

**Last Updated**: December 2024
**Total Files Identified**: 19+ files
**Total Functions**: 55+ functions
**Progress**: 8/19+ files completed (42%)
