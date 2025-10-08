# LayerNorm v2 Implementation Plan

## Overview

LayerNorm v2 is a simplified, single-chip implementation of Layer Normalization optimized for row-major tensors with DRAM input/output. This version removes the complexity of distributed operations, sharded memory layouts, and fused operations present in the current LayerNorm implementation.

## Goals

- **Simplicity**: Clean, maintainable implementation focused on core LayerNorm functionality
- **Performance**: Optimized for single-chip execution with row-major tensors
- **Correctness**: Numerically stable implementation matching PyTorch's LayerNorm behavior
- **Extensibility**: Foundation for future enhancements and optimizations

## Requirements and Constraints

### Core Requirements
- ✅ Single chip execution only (no distributed operations)
- ✅ Row-major input tensors from DRAM
- ✅ Row-major output tensors to DRAM
- ✅ Support standard LayerNorm mathematical operations: `y = (x - μ) / √(σ² + ε) * γ + β`
- ✅ Configurable epsilon parameter (default: 1e-5)
- ✅ Optional learnable affine parameters (gamma, beta)
- ✅ Support for bfloat16 data type only

### Constraints
- 🚫 No distributed LayerNorm stages
- 🚫 No sharded memory layout support initially
- 🚫 No fused pre-add operations
- 🚫 No RMSNORM variant (LayerNorm only)
- 🚫 No legacy compatibility modes
- 🚫 Only bfloat16 data type support (no float32, int8, etc.)
- 🚫 **Phase 1: Single Tensix core only** (multi-core in Phase 2)
- ✅ DRAM-to-DRAM operation only

### Performance Targets
**Phase 1 (Single-Core):**
- Functional correctness on single Tensix core
- Establish performance baseline vs. existing LayerNorm single-core
- Efficient single-core memory bandwidth utilization

**Phase 2 (Multi-Core):**
- Competitive performance with current LayerNorm for row-major tensors
- Linear scaling performance improvements with additional cores
- Optimized multi-core memory bandwidth utilization
- Low kernel launch overhead

## Architecture Design

### Directory Structure
```
ttnn/cpp/ttnn/operations/normalization/layernorm_v2/
├── layernorm_v2.hpp                    # Public API
├── layernorm_v2.cpp                    # Host implementation
├── layernorm_v2_pybind.cpp            # Python bindings
├── device/
│   ├── layernorm_v2_op.hpp            # Device operation header
│   ├── layernorm_v2_op.cpp            # Device operation implementation
│   └── kernels/
│       ├── compute/
│       │   └── layernorm_v2_compute.cpp
│       └── dataflow/
│           ├── layernorm_v2_reader.cpp
│           └── layernorm_v2_writer.cpp
└── test/
    └── test_layernorm_v2.py           # Python tests
```

### API Design

#### Public Interface
```cpp
namespace ttnn::operations::normalization {

struct ExecuteLayerNormV2 {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        float epsilon = 1e-5f,
        const std::optional<const ttnn::Tensor>& weight = std::nullopt,
        const std::optional<const ttnn::Tensor>& bias = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt
    );
};

constexpr auto layer_norm_v2 =
    ttnn::register_operation<"ttnn::layer_norm_v2", ExecuteLayerNormV2>();

}
```

### Implementation Strategy

#### Phase 1: Single Tensix Core Implementation
1. **Host Operation** (`layernorm_v2.cpp`)
   - Input validation (row-major, bfloat16 dtype)
   - Memory configuration setup
   - Dispatch to device operation

2. **Device Operation** (`layernorm_v2_op.cpp`)
   - Single-core tensor spec computation
   - Single-core kernel program creation
   - Single-core output tensor creation

3. **Single-Core Compute Kernel** (`layernorm_v2_compute.cpp`)
   - Row-wise mean calculation: `μ = (1/W) * Σ(x_i)` (single Tensix core)
   - Row-wise variance calculation: `σ² = (1/W) * Σ((x_i - μ)²)` (single Tensix core)
   - Normalization: `y_i = (x_i - μ) / √(σ² + ε)` (single Tensix core)
   - Optional affine transformation: `y_i = y_i * γ + β` (single Tensix core)

4. **Single-Core Dataflow Kernels**
   - **Reader**: Load input tensor data and parameters from DRAM using TensorAccessor (single core)
   - **Writer**: Store normalized output to DRAM using TensorAccessor (single core)

**Phase 1 Testing Milestones (Single Tensix Core):**
- ✅ **Compilation Tests**: All files compile without errors
- ✅ **Build Integration Tests**: CMakeLists.txt changes allow successful build
- ✅ **C++ Registration Tests**: `ttnn::layer_norm_v2` available in C++ namespace
- ✅ **Python Module Tests**: Operation properly registered in normalization module
- ✅ **Basic API Tests**: `ttnn::layer_norm_v2()` function can be called from C++
- ✅ **Python Bindings Tests**: `ttnn.layer_norm_v2()` callable from Python
- ✅ **Python API Tests**: Basic Python interface works with simple tensors
- ✅ **Golden Function Tests**: PyTorch reference function works correctly
- ✅ **Input Validation Tests**: Proper error handling for invalid inputs (wrong dtype, non-row-major)
- ✅ **Single-Core Small Tensor Tests**: Simple 2x4 bfloat16 tensors on one Tensix core
- ✅ **Single-Core Mathematical Correctness**: Compare against PyTorch LayerNorm (single core)
- ✅ **Single-Core Memory Tests**: Verify DRAM read/write operations work on single core
- ✅ **TensorAccessor Tests**: Confirm reader/writer kernels use TensorAccessor correctly
- ✅ **Single-Core Kernel Execution**: Confirm all kernels execute without crashes on single core
- ✅ **Single-Core Performance Baseline**: Establish performance baseline for single-core implementation

#### Phase 2: Multi-Core Scaling (After Single-Core Verification)
**Prerequisites**: Phase 1 single-core implementation must be fully working and tested

1. **Multi-Core Device Operation**: Extend device operation to support multiple Tensix cores
2. **Multi-Core Work Distribution**: Implement tensor partitioning across cores (row-wise distribution)
3. **Multi-Core Compute Kernels**: Adapt compute kernels for parallel execution on multiple cores
4. **Multi-Core Dataflow Coordination**: Coordinate reader/writer kernels across multiple cores
5. **Multi-Core Memory Management**: Optimize DRAM access patterns for parallel execution

**Phase 2 Testing Milestones (Multi-Core Transition):**
- 🎯 **Single-to-Multi-Core Migration**: Verify single-core functionality preserved in multi-core setup
- 🎯 **Multi-Core Correctness**: Confirm multi-core results match single-core results exactly
- 🎯 **Multi-Core Performance**: Measure performance improvements vs. single-core baseline
- 🎯 **Core Utilization Tests**: Verify all cores are utilized efficiently
- 🎯 **Multi-Core Memory Tests**: Test DRAM bandwidth utilization with multiple cores
- 🎯 **Multi-Core Scaling Tests**: Test with 2, 4, 8 cores to verify scaling behavior
- 🎯 **Multi-Core Python Integration**: Full Python workflow with multi-core execution
- 🎯 **Multi-Core Stress Tests**: Large tensor operations (e.g., 1024x4096) on multiple cores
- 🎯 **Regression Tests**: Ensure multi-core doesn't break single-core functionality

#### Phase 3: Advanced Features
1. **In-place Operations**: Memory-efficient variants
2. **Broadcasting**: Support for different gamma/beta shapes
3. **Extended Validation**: Additional edge cases and robustness testing

**Phase 3 Testing Milestones:**
- 🔬 **In-place Tests**: Verify memory-efficient operations produce correct results
- 🔬 **Broadcasting Tests**: Test gamma/beta with different shapes (1D, scalar, etc.)
- 🔬 **Edge Case Tests**: Zero variance, extreme epsilon values, NaN/inf handling
- 🔬 **Python Model Integration**: LayerNorm v2 within transformer models called from Python
- 🔬 **Python Workflow Tests**: End-to-end Python workflows with chained operations
- 🔬 **Robustness Tests**: Stress testing with various tensor shapes and sizes
- 🔬 **Python Error Handling**: Proper error propagation from C++ to Python
- 🔬 **Final Performance**: End-to-end benchmarking vs. current LayerNorm implementation

## Mathematical Implementation

### Algorithm Steps
```cpp
// For each row in the input tensor:
// 1. Compute mean: μ = (1/W) * Σ(x_i) where W is the width dimension
for (uint32_t w = 0; w < W_tiles; w++) {
    sum += input_tile[w];
}
mean = sum / W;

// 2. Compute variance: σ² = (1/W) * Σ((x_i - μ)²)
for (uint32_t w = 0; w < W_tiles; w++) {
    diff = input_tile[w] - mean;
    variance += diff * diff;
}
variance = variance / W;

// 3. Normalize: y_i = (x_i - μ) / √(σ² + ε)
inv_std = 1.0 / sqrt(variance + epsilon);
for (uint32_t w = 0; w < W_tiles; w++) {
    normalized[w] = (input_tile[w] - mean) * inv_std;
}

// 4. Apply affine transformation (if provided): y_i = y_i * γ + β
if (gamma) {
    for (uint32_t w = 0; w < W_tiles; w++) {
        normalized[w] = normalized[w] * gamma[w];
    }
}
if (beta) {
    for (uint32_t w = 0; w < W_tiles; w++) {
        normalized[w] = normalized[w] + beta[w];
    }
}
```

### Circular Buffer Usage
```cpp
constexpr auto cb_input = tt::CBIndex::c_0;      // Input tensor data
constexpr auto cb_mean = tt::CBIndex::c_1;       // Row means
constexpr auto cb_variance = tt::CBIndex::c_2;   // Row variances
constexpr auto cb_inv_std = tt::CBIndex::c_3;    // 1/sqrt(var + eps)
constexpr auto cb_normalized = tt::CBIndex::c_4; // Normalized values
constexpr auto cb_gamma = tt::CBIndex::c_5;      // Weight parameters (optional)
constexpr auto cb_beta = tt::CBIndex::c_6;       // Bias parameters (optional)
constexpr auto cb_output = tt::CBIndex::c_16;    // Final output
```

### TensorAccessor Implementation
```cpp
// Reader kernel (layernorm_v2_reader.cpp)
constexpr auto input_accessor_args = TensorAccessorArgs<4>(); // For NCHW tensor
const auto input_accessor = TensorAccessor(input_accessor_args, input_addr, page_size);

// Writer kernel (layernorm_v2_writer.cpp)
constexpr auto output_accessor_args = TensorAccessorArgs<4>(); // For NCHW tensor
const auto output_accessor = TensorAccessor(output_accessor_args, output_addr, page_size);

// Use accessor for efficient DRAM access patterns:
// input_accessor.read_page(page_id, l1_write_addr);
// output_accessor.write_page(page_id, l1_read_addr);
```

## Testing Strategy

### Unit Tests
1. **Correctness Tests**
   - Compare outputs with PyTorch LayerNorm
   - Test with various tensor shapes (bfloat16 only)
   - Edge cases (very small/large values, zero variance)

2. **Parameter Tests**
   - Different epsilon values
   - With/without gamma and beta parameters
   - Various parameter shapes and values

3. **Performance Tests**
   - Benchmark against current LayerNorm implementation
   - Memory bandwidth utilization analysis
   - Latency measurements

### Integration Tests
1. **Model Integration**: Test within transformer architectures
2. **Pipeline Tests**: Chaining with other operations
3. **Memory Tests**: DRAM access patterns and efficiency

### Test Cases
```python
# test_layernorm_v2.py
def test_python_api_callable():
    """Test that ttnn.layer_norm_v2 is callable from Python"""
    pass

def test_basic_layernorm_v2():
    """Test basic functionality against PyTorch"""
    pass

def test_with_affine_params():
    """Test with gamma and beta parameters"""
    pass

def test_bfloat16_support():
    """Test bfloat16 data type functionality"""
    pass

def test_python_parameter_validation():
    """Test Python API parameter validation and error messages"""
    pass

def test_python_tensor_conversion():
    """Test conversion between Python tensors and TT tensors"""
    pass

def test_edge_cases():
    """Test numerical edge cases"""
    pass

def test_python_integration_workflow():
    """Test LayerNorm v2 in end-to-end Python workflows"""
    pass

def test_performance():
    """Benchmark against existing implementation"""
    pass
```

## Operation Registration Requirements

To make `ttnn.layer_norm_v2()` available outside the layernorm_v2 folder, the following integrations are required:

### **C++ Registration** (Automatic via `register_operation`)
✅ **Already handled** in `layernorm_v2.hpp`:
```cpp
constexpr auto layer_norm_v2 =
    ttnn::register_operation<"ttnn::layer_norm_v2", ExecuteLayerNormV2>();
```
This automatically makes the operation available as `ttnn::layer_norm_v2` in C++.

### **Python Bindings** (Required Changes)
1. **Add pybind binding in `layernorm_v2_pybind.cpp`**:
```cpp
void bind_normalization_layernorm_v2_operation(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::layer_norm_v2,
        R"doc(LayerNorm v2 documentation...)doc",
        ttnn::pybind_arguments_t{...});
}
```

2. **Update `normalization_pybind.cpp`**:
```cpp
// Add include
#include "layernorm_v2/layernorm_v2_pybind.hpp"

// Add binding call in py_module()
void py_module(py::module& module) {
    // ... existing bindings ...
    detail::bind_normalization_layernorm_v2(module);  // ADD THIS LINE
}
```

### **Build System Integration** (Required Changes)
Update `ttnn/cpp/ttnn/operations/normalization/CMakeLists.txt`:

**Add to kernels glob:**
```cmake
file(
    GLOB_RECURSE kernels
    layernorm/device/kernels/*
    layernorm_v2/device/kernels/*  # ADD THIS LINE
    softmax/device/kernels/*
)
```

**Add to FILE_SET api:**
```cmake
target_sources(
    ttnn_op_normalization
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        FILES
            layernorm/layernorm.hpp
            layernorm_v2/layernorm_v2.hpp  # ADD THIS LINE
            # ... other headers
```

**Add to PRIVATE sources:**
```cmake
target_sources(
    ttnn_op_normalization
    PRIVATE
        # ... existing sources ...
        layernorm_v2/device/layernorm_v2_op.cpp      # ADD THESE LINES
        layernorm_v2/layernorm_v2.cpp                # ADD THESE LINES
)
```

### **Python Golden Function** (Optional Enhancement)
Add to `ttnn/ttnn/operations/normalization.py`:
```python
def _golden_function_v2(input_tensor: ttnn.Tensor, *, epsilon=1e-5, weight=None, bias=None, **_):
    import torch
    # LayerNorm v2 specific reference implementation
    return torch.nn.functional.layer_norm(input_tensor, (input_tensor.shape[-1],), weight, bias, eps=epsilon)

ttnn.attach_golden_function(ttnn.layer_norm_v2, golden_function=_golden_function_v2)
```

### **Registration Flow Summary**

1. **C++ Operation Registration**: `register_operation` in `layernorm_v2.hpp` → Available as `ttnn::layer_norm_v2` in C++

2. **Python Binding**: `bind_registered_operation` → Available as `ttnn._ttnn.operations.normalization.layer_norm_v2`

3. **Main Python Module**: Normalization module integration → Available as `ttnn.layer_norm_v2()` in Python

4. **Optional Golden Function**: Attach PyTorch reference → Enables testing framework integration

The key insight is that `register_operation` automatically handles C++ namespace exposure, but Python requires explicit pybind integration through the normalization module system.

## Implementation Timeline

### Week 1-2: Single-Core Foundation (Phase 1)
- [ ] Create directory structure and basic files
- [ ] Basic host operation with validation (single-core)
- [ ] Single-core device operation setup
- [ ] **Operation Registration Setup**: Update CMakeLists.txt and normalization_pybind.cpp
- [ ] Begin Phase 1 single-core testing milestones

### Week 3-4: Single-Core Kernel Implementation (Phase 1 continued)
- [ ] Implement single-core compute kernel with basic LayerNorm math
- [ ] Create single-core reader/writer dataflow kernels with TensorAccessor
- [ ] Complete single-core device operation integration
- [ ] **Python bindings implementation** (layernorm_v2_pybind.cpp)
- [ ] **Golden function attachment** for testing framework integration
- [ ] **Validate all Phase 1 single-core testing milestones**
- [ ] **Single-core implementation must be fully working before Phase 2**

### Week 5-6: Multi-Core Scaling (Phase 2)
- [ ] **Prerequisites verified**: Phase 1 single-core fully functional
- [ ] Extend device operation for multi-core support
- [ ] Implement multi-core work distribution and coordination
- [ ] Adapt kernels for parallel execution on multiple Tensix cores
- [ ] Complete Phase 2 multi-core testing milestones

### Week 7-8: Advanced Features & Documentation (Phase 3)
- [ ] In-place operations and broadcasting support
- [ ] Complete Phase 3 testing milestones
- [ ] Documentation and final benchmarking

## Success Criteria

### Phase 1 Success Criteria (Single-Core)
**Functional Requirements:**
- ✅ Passes all correctness tests against PyTorch LayerNorm on single Tensix core
- ✅ Supports required tensor shapes and bfloat16 data type on single core
- ✅ Handles edge cases robustly on single core
- ✅ Clean, maintainable single-core code structure

**Performance Requirements:**
- 🎯 Single-core functional correctness (performance baseline established)
- 🎯 Single-core memory operations working correctly
- 🎯 Single-core kernel execution without crashes

### Phase 2+ Success Criteria (Multi-Core)
**Functional Requirements:**
- ✅ Multi-core results match single-core results exactly
- ✅ Supports scaling from single-core to multi-core seamlessly
- ✅ Preserves all single-core functionality in multi-core mode

**Performance Requirements:**
- 🎯 Performance within 10% of current LayerNorm for row-major tensors
- 🎯 Efficient multi-core memory bandwidth utilization (>80% theoretical peak)
- 🎯 Linear scaling performance with additional cores
- 🎯 Low kernel launch overhead (<100μs)

### Quality Requirements (All Phases)
- 📚 Comprehensive documentation and examples
- 🧪 >95% test coverage
- 🔍 Code review approval from TT team
- 🚀 Ready for integration into main codebase

## Future Enhancements

### Short Term (Next 2-3 months)
1. **Multi-core Support**: Parallelize execution across cores
2. **Additional Data Types**: Support for float32, int8, float16
3. **In-place Operations**: Memory-efficient variants

### Medium Term (3-6 months)
1. **Kernel Fusion**: Fuse with other operations (GELU, etc.)
2. **Advanced Memory Layouts**: Support for other layouts if needed
3. **Auto-tuning**: Automatic parameter optimization

### Long Term (6+ months)
1. **Multi-chip Support**: If distributed requirements emerge
2. **Specialized Variants**: RMSNORM, GroupNorm, etc.
3. **Hardware-specific Optimizations**: Architecture-specific tuning

## Risk Assessment

### Technical Risks
- **Performance**: Risk of not meeting performance targets
  - *Mitigation*: Early benchmarking and incremental optimization
- **Numerical Stability**: Potential for numerical issues with edge cases
  - *Mitigation*: Comprehensive testing with diverse inputs
- **Memory Bandwidth**: DRAM access patterns may limit performance
  - *Mitigation*: Careful memory access pattern design

### Timeline Risks
- **Kernel Complexity**: Compute kernel may take longer than expected
  - *Mitigation*: Start with simple implementation, add optimizations incrementally
- **Integration Issues**: API integration may reveal unforeseen issues
  - *Mitigation*: Early integration testing and stakeholder feedback

### Quality Risks
- **Test Coverage**: May miss edge cases in testing
  - *Mitigation*: Systematic test case generation and review
- **Documentation**: May lack sufficient documentation for users
  - *Mitigation*: Documentation-driven development approach

## Conclusion

LayerNorm v2 represents a focused, simplified approach to implementing Layer Normalization on TT hardware. By removing distributed complexity and focusing on the core single-chip, row-major use case, we can deliver a clean, performant, and maintainable implementation that serves as a solid foundation for future enhancements.

The phased approach ensures we deliver value incrementally while maintaining high quality standards throughout the development process.
