# MaxPool3D PASS 4 Implementation Plan

## Overview
This document outlines the plan to implement functional compute for MaxPool3D operation, moving from the dummy synchronization-only compute kernel (PASS 3) to actual max pooling computation.

## Current State (PASS 3)
- ✅ Reader kernel fetches appropriate sticks from DRAM (e.g., 8 sticks for 2x2x2 filter)
- ✅ Compute kernel handles basic synchronization without hanging (dummy - returns first stick)
- ✅ Writer kernel sends results back to DRAM output
- ✅ All kernels compile successfully

## PASS 4 Goals
Implement functional max pooling computation in the compute kernel while maintaining the working reader/writer infrastructure.

## Two-Phase Implementation Strategy

We will implement two separate solutions:
1. **Phase A: RISC-V based solution** - Simple CPU-based implementation
2. **Phase B: FPU/SFPU based solution** - Hardware-accelerated implementation (similar to pool2d)

Each phase will be broken into mini steps with testing at each step.

---

## Phase A: RISC-V Based Solution

### Overview
Implement max pooling using the RISC-V processor in the compute kernel for direct memory manipulation and simple max finding.

### Mini Steps with Testing

#### Step A1: Memory Access and Channel Data Extraction
**Goal**: Access input buffer and extract channel data from sticks

**Implementation**:
- Access input circular buffer using `get_read_ptr()`
- Create helper function to parse stick into channel values
- Read and extract channels from first stick
- Log/verify channel data is accessible and correct
- Still output dummy data (first stick)

**Testing**:
- Verify kernel compiles and doesn't hang
- Check that input data can be accessed without violations
- Unit test channel extraction with known input patterns
- Test with different channel counts (1, 2, 4, 8, 16, 32)
- Validate bfloat16 data type handling

**Success Criteria**: Kernel runs, extracts correct channel values, outputs dummy result

#### Step A2: Single Window Max Pooling Implementation
**Goal**: Implement complete max pooling for single window (all channels)

**Implementation**:
- Read all window_size sticks from input buffer
- Extract all channels from each stick in the window
- For each channel, find maximum value across the 3D window
- Construct output stick with all channel max values
- Replace dummy output with computed max pooling result

**Testing**:
- Test with 2x2x2 window and various channel counts
- Use known input patterns where max values are predictable
- Verify max finding logic for edge cases (all same values, negative values)
- Cross-validate against software reference implementation
- Test with different window sizes (2x2x2, 3x3x3)

**Success Criteria**: Correct max values computed for all channels in single window

#### Step A3: Multiple Windows and Production Ready
**Goal**: Handle multiple output positions and optimize for production use

**Implementation**:
- Process all num_windows correctly with proper synchronization
- Add bounds checking and error handling
- Optimize inner loops for performance
- Handle padding cases and edge conditions
- Add conditional debug logging

**Testing**:
- Test with multiple output positions and various tensor sizes
- Verify each window processes independently
- End-to-end testing with complete tensor operations
- Performance benchmarking vs dummy implementation
- Test edge cases (padding, boundary conditions)
- Stress testing with large inputs and memory usage validation

**Success Criteria**: Robust implementation handles all cases and multiple windows correctly

#### Step A4: Complete Max Functionality in RISC Approach
**Goal**: Implement proper output mechanism to write computed max values in RISC approach

**Implementation**:
- Fix output tile writing to actually write computed max values (not dummy data)
- Implement direct memory writing approach for computed channel maximums
- Create proper tile data construction for output
- Ensure computed max values reach the final output tensor

**Testing**:
- Test that output contains actual computed max values (8.0 from [1,2,3,4,5,6,7,8])
- Validate with different input patterns and known expected outputs
- Test single window scenarios with various channel counts
- Verify end-to-end correctness: input → computation → output

**Success Criteria**: RISC-V implementation outputs correct computed max values, not dummy data

---

## Phase B: FPU/SFPU Based Solution

### Overview
Implement max pooling using hardware-accelerated FPU/SFPU operations, similar to pool2d implementation.

### Mini Steps with Testing

#### Step B1: SFPU Initialization and Basic Operations
**Goal**: Set up SFPU for max operations and verify basic functionality

**Implementation**:
- Initialize SFPU with proper configuration
- Set up destination registers for computations
- Implement basic tile unpack/pack operations
- Process single tile with identity operation (no max yet)

**Testing**:
- Verify SFPU initialization doesn't hang
- Test tile unpack/pack operations work correctly
- Validate proper register management
- Compare output with input (should be identical)

**Success Criteria**: SFPU operations work, data flows correctly through tiles

#### Step B2: Single Tile Max Operation
**Goal**: Perform max operation on single tile using SFPU

**Implementation**:
- Use SFPU max operations on single input tile
- Implement proper tile register management
- Handle single window position only
- Output computed max tile

**Testing**:
- Test max operation on known tile patterns
- Verify SFPU max produces correct results
- Compare against software max operation
- Test with various data patterns

**Success Criteria**: Single tile max operation works correctly

#### Step B3: Multi-Tile Max Reduction
**Goal**: Perform max reduction across multiple input tiles (window)

**Implementation**:
- Load multiple tiles from input window
- Perform pairwise max operations across tiles
- Implement tree reduction for efficiency
- Output single max result tile

**Testing**:
- Test with 2-tile, 4-tile, 8-tile windows
- Verify reduction tree logic is correct
- Cross-validate against RISC-V implementation
- Test with different window sizes

**Success Criteria**: Multi-tile max reduction produces correct results

#### Step B4: Channel-wise Max Operations
**Goal**: Handle channel dimension properly in tile-based operations

**Implementation**:
- Ensure max operations preserve channel structure
- Handle different channel counts correctly
- Optimize for common channel sizes (32, 64, etc.)
- Maintain proper tile alignment

**Testing**:
- Test various channel counts with tile operations
- Verify channel-wise max is computed correctly
- Compare results with RISC-V implementation
- Test channel alignment edge cases

**Success Criteria**: Channel-wise operations work correctly with tiles

#### Step B5: Full Window Processing with SFPU
**Goal**: Complete 3D window max pooling with SFPU acceleration

**Implementation**:
- Integrate all components into full window processing
- Handle multiple windows efficiently
- Optimize register usage and tile management
- Maintain synchronization with reader/writer

**Testing**:
- End-to-end testing with complete operations
- Performance comparison with RISC-V implementation
- Correctness validation against software reference
- Multi-window processing verification

**Success Criteria**: Full SFPU-accelerated maxpool3d works correctly

#### Step B6: Performance Optimization and Validation
**Goal**: Optimize performance and validate production readiness

**Implementation**:
- Profile and optimize critical paths
- Implement efficient tile caching strategies
- Add comprehensive error handling
- Optimize for different tensor sizes

**Testing**:
- Performance benchmarking vs RISC-V version
- Large tensor stress testing
- Memory efficiency validation
- Production workload testing

**Success Criteria**: Optimized SFPU implementation outperforms RISC-V version

---

## Testing Framework for Each Step

### Micro-Tests (per mini step)
- **Unit tests**: Isolated functionality testing
- **Known pattern tests**: Predictable input/output validation
- **Edge case tests**: Boundary conditions and error cases
- **Performance tests**: Timing and resource usage

### Integration Tests (after each phase)
- **End-to-end tests**: Full operation pipeline testing
- **Cross-reference tests**: Compare with PyTorch MaxPool3d
- **Multi-configuration tests**: Various tensor sizes and parameters
- **Regression tests**: Ensure previous functionality still works

### Validation Criteria for Each Step
1. **Compilation**: Code compiles without errors
2. **Execution**: Kernel runs without hanging or crashes
3. **Correctness**: Output matches expected results
4. **Performance**: Reasonable execution time
5. **Memory Safety**: No buffer overruns or memory leaks

## Implementation Timeline

### Phase A: RISC-V Implementation
- Step A1: 2-3 hours (Memory access + Channel extraction)
- Step A2: 3-4 hours (Complete single window max pooling)
- Step A3: 2-3 hours (Multiple windows + Production optimization)
**Total Phase A**: 7-10 hours

### Phase B: SFPU Implementation
- Step B1: 2-3 hours (SFPU initialization)
- Step B2: 2-3 hours (Single tile max)
- Step B3: 3-4 hours (Multi-tile reduction)
- Step B4: 2-3 hours (Channel handling)
- Step B5: 3-4 hours (Full integration)
- Step B6: 2-3 hours (Optimization)
**Total Phase B**: 14-20 hours

**Overall Timeline**: 21-30 hours across both phases

## Next Steps for Implementation

### Getting Started
1. **Choose starting phase**: Begin with Phase A (RISC-V) for simpler debugging
2. **Set up test framework**: Create unit tests for each mini step
3. **Implement step-by-step**: Don't skip testing between steps
4. **Document progress**: Track what works and what doesn't for each step

### Success Criteria for PASS 4 Completion

**Phase A Success**:
- RISC-V implementation produces correct MaxPool3D results
- Performance is reasonable compared to dummy implementation
- All unit tests pass for each mini step
- Integration tests match PyTorch reference

**Phase B Success**:
- SFPU implementation outperforms RISC-V version
- Maintains correctness of RISC-V implementation
- Efficient use of hardware acceleration
- Production-ready performance characteristics

## References and Resources

- **TT-Metal Kernel APIs**: Compute kernel documentation and examples
- **Pool2D Implementation**: Reference for SFPU-based pooling operations
- **SFPU Documentation**: Special Function and Packing Unit APIs
- **Circular Buffer APIs**: Buffer management and synchronization
- **PyTorch MaxPool3D**: Mathematical reference for correctness validation

## Key Implementation Notes

### Data Format Considerations
- Input/output format: bfloat16 in row-major layout
- Stick format: All channels for a single spatial position
- Window layout: Organized as T×H×W sticks in circular buffer
- Channel extraction: Parse individual channels from stick data

### Performance Expectations
- **RISC-V Phase**: Functional but slower than hardware acceleration
- **SFPU Phase**: Significant performance improvement over RISC-V
- **Target**: Competitive with or better than CPU-based implementations

### Testing Philosophy
- **Test every step**: Never proceed without validating current step
- **Known patterns**: Use predictable inputs for easy validation
- **Cross-reference**: Always compare against software reference
- **Incremental complexity**: Start simple, add complexity gradually
