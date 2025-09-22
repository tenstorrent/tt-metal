# Depthwise Conv2d Optimization Plan

## Progress Status
🔄 **CURRENT STATUS:** Planning Phase - Ready to Begin Implementation

### Progress Tracker
```
Phase 1: Analysis & Setup     [▓▓] 2/2 complete
Phase 2: Core Implementation  [ ] 0/4 complete
Phase 3: Integration         [ ] 0/3 complete
Phase 4: Performance & Tests [ ] 0/3 complete
Phase 5: Production Ready    [ ] 0/2 complete

Overall Progress: 2/14 steps (14%)
```

**Current Branch:** `depthwise_conv2d_optimized`
**Next Steps:** Phase 2, Step 3 - Reuse Pool Kernel for Depthwise Conv

**Important Setup Note:** Always run tests with: `source python_env/bin/activate && TEST_COMMAND`

---

## Project Overview

### Goal
Implement a high-performance depthwise conv2d operation that leverages pool kernels instead of the current fallback to regular conv2d, achieving 2-5x performance improvement for models like MobileNetV2 and YOLOv10.

### Current Problem
- Depthwise convolution currently expands weight matrices with many zeros
- Results in unnecessary multiplications and poor performance
- Critical bottleneck in efficient mobile architectures

### Solution Strategy
Use a pooling-like approach that:
1. Processes one spatial window per iteration
2. Produces one output "stick"
3. Eliminates zero-multiplications through targeted computation
4. Leverages existing optimized pool kernel infrastructure

---

## Phase 1: Analysis & Setup (Foundation)

### Step 1: Establish Performance Baseline & Current Analysis
**Goal:** Get baseline performance numbers and understand current depthwise conv implementation
**Achievement Method:**
- Run performance measurement using Tracy: `python -m tracy -r -m -p "pytest tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py::test_groups_vs_pool2"`
- Look at DEVICE KERNEL DURATION column on optimizedconvnew for baseline numbers
- Analyze how current groups=input_channels (depthwise) is handled in conv2d
- Study existing pool kernel implementation for reuse opportunities

**Changes Required:**
- No code changes, pure analysis
- Document current performance metrics
- Document findings in analysis notes

**Testing:**
- Run `test_groups_vs_pool2` with Tracy profiling
- Record baseline performance metrics for MobileNetV2 and YOLOv10 cases

**Status After:** Clear performance baseline established, understanding of current bottlenecks

**✅ COMPLETED - Baseline Performance Analysis:**
- Identified 6 depthwise conv operations from Tracy profiling
- DEVICE KERNEL DURATION range: 18.4ms - 232.1ms (total: ~767ms)
- Key bottlenecks: Large spatial dimensions (80×80) + high channels (320-640) = worst performance
- Memory layouts: HEIGHT_SHARDED and BLOCK_SHARDED L1 configurations
- Target for optimization: 2-5x improvement means reducing total time to ~153-383ms

**Detailed Performance Breakdown (DEVICE KERNEL DURATION):**

| Operation | Model Type | Input Shape | Channels | Duration (ms) | Duration (M cycles) | Memory Layout |
|-----------|------------|-------------|----------|---------------|-------------------|---------------|
| Op 14     | MobileNetV2 | 10×56×56   | 144      | 171.009      | ~171.0           | HEIGHT_SHARDED |
| Op 27     | MobileNetV2 | 10×14×14   | 576      | 137.788      | ~137.8           | BLOCK_SHARDED  |
| Op 40     | MobileNetV2 | 10×7×7     | 960      | 126.492      | ~126.5           | BLOCK_SHARDED  |
| Op 53     | MobileNetV2 | 10×7×7     | 112      | 18.451       | ~18.5            | HEIGHT_SHARDED |
| Op 66     | YOLOv10     | 1×80×80    | 320      | 232.140      | ~232.1           | HEIGHT_SHARDED |
| Op 79     | YOLOv10     | 1×40×40    | 640      | 181.828      | ~181.8           | BLOCK_SHARDED  |

**Performance Analysis:**
- **Worst performers**: YOLOv10 operations (232.1M, 181.8M cycles) due to large spatial + high channels
- **Best performer**: MobileNetV2 Op 53 (18.5M cycles) with smaller spatial and lower channels
- **Memory layout impact**: Both HEIGHT_SHARDED and BLOCK_SHARDED show similar performance patterns
- **Scaling pattern**: Performance correlates with spatial_size × channels complexity
- **Total baseline**: ~767ms (767M cycles) across all 6 operations

*Note: Assuming ~1 GHz device clock, 1ms ≈ 1M cycles for cycle estimation*

---

### Step 2: Investigate Pool Kernel Reuse Strategy
**Goal:** Determine how to reuse existing pool kernels for depthwise conv2d with specific conditions
**Achievement Method:**
- Study existing pool kernel code in `ttnn/cpp/ttnn/operations/pool/`
- Examine PoC approach from `skrstic/depthwise-opt` branch for ideas
- Identify conditions where pool kernels can be adapted for depthwise convolution
- Plan modifications needed to pool kernels for depthwise conv support

**Changes Required:**
- Analysis of pool kernel architecture
- Document modification strategy for pool kernels
- Plan conditional logic for depthwise conv detection

**Testing:**
- Theoretical validation of pool kernel reuse approach
- Design validation through code analysis

**Status After:** Clear strategy for reusing pool kernels, ready for implementation

**✅ COMPLETED - Pool Kernel Reuse Strategy:**
- **Key insight**: Use Pool2DType::AVG_POOL2D (SUM reduction) for depthwise conv spatial processing
- **PoC approach**: Simplified pool interface, direct tensor flow (Input → Pool → Output)
- **Reuse conditions**: When `groups == input_channels`, route to pool-based implementation
- **Required modifications**:
  1. Add depthwise detection in conv2d.cpp
  2. Adapt pool kernel to handle weight tensors
  3. Add per-channel weight multiplication step
  4. Integrate with existing conv2d API
- **Memory compatibility**: Pool supports same sharded layouts as current depthwise conv
- **Performance benefit**: Eliminates zero-multiplication overhead in current approach

---

## Phase 2: Core Implementation (Pool-Based Kernel)

### Step 3: Create Dedicated Depthwise Conv Factory (Revised Approach)
**Goal:** Create a dedicated depthwise conv2d operation factory that reuses pool kernel infrastructure
**Achievement Method:**
- **Revised Strategy**: Instead of calling avgpool directly, create a separate depthwise conv factory
- Reuse pool kernel compute infrastructure (same kernels, different factory)
- Add depthwise conv detection based on groups == input_channels
- Implement dedicated depthwise conv operation that internally uses pool-like processing

**Changes Required:**
- Create new `DepthwiseConv2DOp` factory class in conv2d operations
- Reuse existing pool compute kernels (`compute_pool_2d.cpp`) with depthwise-specific configuration
- Add detection logic in conv2d.cpp: when groups == input_channels, route to depthwise factory
- Configure pool kernels for depthwise convolution (SUM reduction + weight multiplication)

**Testing:**
- Use existing `test_groups_vs_pool2` for validation
- Test depthwise conv detection and routing
- Basic functionality tests with weights = 1 initially

**Status After:** Dedicated depthwise conv factory that internally reuses pool kernel infrastructure

**Key Architecture Decision:**
- **Factory Pattern**: Each operation (regular conv2d, depthwise conv2d, pool2d) has its own factory
- **Kernel Reuse**: Depthwise conv factory reuses pool compute kernels but with different configuration
- **Clean Separation**: Maintains API clarity while achieving performance through shared kernel infrastructure

---

### Step 4: Implement Output Stick Generation
**Goal:** Generate correct output format from pool-like processing
**Achievement Method:**
- Implement output tensor construction from processed windows
- Handle output stride and shape correctly
- Ensure compatibility with existing tensor layouts
- Optimize memory access patterns

**Changes Required:**
- Output generation logic in depthwise kernel
- Memory layout handling
- Output tensor construction utilities

**Testing:**
- Validate output shapes match PyTorch depthwise conv
- Test various input/output dimensions
- Memory layout correctness tests

**Status After:** Complete output generation working, ready for multiplication step

---

### Step 5: Add Multiplication Step for Weight Application
**Goal:** Implement the extra multiplication step for weight application
**Achievement Method:**
- Add weight multiplication to the kernel pipeline
- Handle per-channel weight application efficiently
- Optimize to minimize performance overhead
- Still use weights = 1 for initial validation

**Changes Required:**
- Weight multiplication logic in kernel
- Per-channel weight handling
- Performance optimization for multiplication step

**Testing:**
- Verify weights = 1 produces same results as before
- Test multiplication step performance impact
- Validate numerical accuracy

**Status After:** Complete pool-based depthwise kernel working with weight application

---

### Step 6: Integration with ttnn.conv2d Interface
**Goal:** Make pool-based approach accessible through standard ttnn.conv2d call
**Achievement Method:**
- Add detection logic for depthwise convolutions (groups == input_channels)
- Route depthwise calls to pool-based implementation
- Maintain backward compatibility with existing conv2d calls
- Ensure seamless user experience

**Changes Required:**
- Modify `ttnn/ttnn/operations/conv2d.py`
- Add routing logic in `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp`
- Update pybind interfaces if needed

**Testing:**
- Test `ttnn.conv2d` calls with groups=input_channels use new path
- Verify regular conv2d calls unchanged
- Integration tests with existing codebase

**Status After:** Pool-based depthwise conv accessible via standard interface

---

## Phase 3: Integration & Validation (Connect Everything)

### Step 7: Implement Real Weight Handling
**Goal:** Support actual weight tensors instead of assuming weights = 1
**Achievement Method:**
- Modify kernel to accept and process real weight tensors
- Handle weight tensor layout and memory access
- Optimize weight loading and application
- Validate against PyTorch depthwise conv

**Changes Required:**
- Weight tensor handling in kernel
- Memory access optimizations for weights
- Weight layout compatibility

**Testing:**
- Test with various weight tensor shapes and values
- Validate numerical accuracy against PyTorch
- Performance testing with real weights

**Status After:** Full weight support implemented and validated

---

### Step 8: Handle Edge Cases and Input Validation
**Goal:** Robust handling of various input configurations
**Achievement Method:**
- Add input validation for depthwise conv requirements
- Handle edge cases (1x1 kernels, different padding modes)
- Implement proper error handling and reporting
- Support various data types and layouts

**Changes Required:**
- Input validation logic
- Edge case handling in kernel
- Error reporting mechanisms

**Testing:**
- Comprehensive edge case testing
- Error condition validation
- Various input configuration tests

**Status After:** Robust implementation handling all valid input cases

---

### Step 9: Optimize Memory Access and Performance
**Goal:** Optimize the implementation for maximum performance
**Achievement Method:**
- Analyze and optimize memory access patterns
- Minimize data movement and copies
- Optimize kernel execution efficiency
- Profile and tune performance bottlenecks

**Changes Required:**
- Memory access optimizations
- Kernel execution tuning
- Data layout optimizations

**Testing:**
- Performance benchmarking
- Memory usage analysis
- Comparison with baseline implementation

**Status After:** Performance-optimized implementation ready for testing

---

## Phase 4: Performance & Testing (Validation & Benchmarking)

### Step 10: Comprehensive Testing Suite
**Goal:** Ensure correctness across all supported configurations
**Achievement Method:**
- Expand `test_groups_vs_pool2` with more test cases
- Add unit tests for kernel components
- Create integration tests with real models
- Add regression tests for edge cases

**Changes Required:**
- Enhanced test suite in `tests/ttnn/nightly/unit_tests/operations/conv/`
- Model integration tests
- Performance regression tests

**Testing:**
- All test cases pass with high accuracy (PCC > 0.99)
- Model-level validation with MobileNetV2 and YOLOv10
- Performance meets target improvements

**Status After:** Fully tested implementation with comprehensive validation

---

### Step 11: Performance Benchmarking & Optimization
**Goal:** Validate performance improvements and optimize further
**Achievement Method:**
- Benchmark against original conv2d implementation
- Measure improvements on MobileNetV2 and YOLOv10 models
- Profile and identify remaining optimization opportunities
- Document performance characteristics

**Changes Required:**
- Performance benchmarking scripts
- Profiling utilities
- Additional optimizations based on findings

**Testing:**
- Performance tests show 2-5x improvement target met
- Model-level performance improvements validated
- Stability under various workloads

**Status After:** Performance validated and optimized, ready for production use

---

### Step 12: Model Integration Validation
**Goal:** Validate end-to-end functionality with real models
**Achievement Method:**
- Test with complete MobileNetV2 model
- Validate YOLOv10 depthwise conv layers
- Ensure numerical accuracy and performance in full model context
- Test training and inference scenarios

**Changes Required:**
- Model integration test scripts
- End-to-end validation utilities
- Model-specific test configurations

**Testing:**
- Full model accuracy validation
- Performance improvement measurement in model context
- Training stability tests if applicable

**Status After:** Model integration validated, ready for production deployment

---

## Phase 5: Production Readiness (Finalization)

### Step 13: Documentation & Code Review Preparation
**Goal:** Prepare implementation for code review and deployment
**Achievement Method:**
- Add comprehensive code documentation
- Create user-facing documentation
- Prepare PR with clear description and test results
- Address any remaining code quality issues

**Changes Required:**
- Code comments and documentation
- User documentation updates
- PR preparation materials

**Testing:**
- Code review checklist validation
- Documentation accuracy verification
- Final integration tests

**Status After:** Implementation ready for code review and team validation

---

### Step 14: Final Validation & Deployment
**Goal:** Complete final validation and prepare for merge
**Achievement Method:**
- Conduct final comprehensive testing
- Address any code review feedback
- Validate performance targets achieved
- Prepare for merge to main branch

**Changes Required:**
- Code review feedback implementation
- Final testing and validation
- Merge preparation

**Testing:**
- Final regression test suite
- Performance validation
- Integration test suite

**Status After:** Feature complete and ready for production use

---

## Success Criteria

### Functional Requirements ✅
- [ ] `ttnn.conv2d` with groups=input_channels uses pool-based approach
- [ ] Numerical accuracy matches PyTorch (PCC > 0.99)
- [ ] All existing conv2d functionality preserved
- [ ] Support for various kernel sizes, strides, and padding
- [ ] Real weight tensor handling implemented

### Performance Requirements 📈
- [ ] 2-5x performance improvement over current depthwise conv
- [ ] MobileNetV2 model performance improved
- [ ] YOLOv10 model performance improved
- [ ] Memory usage optimized vs current implementation

### Quality Requirements 🔍
- [ ] Comprehensive test coverage
- [ ] All edge cases handled gracefully
- [ ] Code review approval
- [ ] Documentation complete
- [ ] No regression in existing functionality

---

## Risk Mitigation

### Technical Risks
1. **Pool kernel compatibility issues**
   *Mitigation:* Start with simple cases, gradually add complexity

2. **Performance overhead from multiplication step**
   *Mitigation:* Profile early and optimize continuously

3. **Memory layout incompatibilities**
   *Mitigation:* Thorough analysis of existing layouts, incremental testing

### Timeline Risks
1. **Complexity underestimation**
   *Mitigation:* Phased approach allows for re-estimation and adjustment

2. **Integration challenges**
   *Mitigation:* Early integration testing, frequent validation points

---

## Notes & Context

- **Branch:** Working in `depthwise_conv2d_optimized` branch
- **Key Files:** Focus on `ttnn/cpp/ttnn/operations/conv/conv2d/` and pool operations
- **Performance Baseline:** Established through `test_groups_vs_pool2`
- **Target Models:** MobileNetV2 and YOLOv10 are primary validation cases
- **Reference Implementation:** Use PyTorch depthwise conv as accuracy reference

This plan provides a systematic approach to implementing pool-based depthwise convolution optimization while maintaining code quality and ensuring thorough validation at each step.
