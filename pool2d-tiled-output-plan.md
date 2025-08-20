# Pool2D Tiled Output Implementation Plan

## Goal
Add optional tiled output support to maxpool operation while maintaining backward compatibility with existing BF16 row-major output.

## Overview
Currently maxpool supports only BF16 row-major output. We want to add two optional parameters:
- `output_data_format` (default: BF16)
- `output_layout` (default: ROW_MAJOR)

## Step-by-Step Implementation

### Step 1: Add Optional Parameters to API ✅ COMPLETED
**What needs to be changed:**
- ✅ Add `output_data_format` and `output_layout` parameters to maxpool function signature
- ✅ Set defaults to maintain backward compatibility (BF16, ROW_MAJOR)
- ✅ Update parameter validation

**Why:**
- Provides user control over output format while preserving existing behavior
- Clear separation between new and legacy functionality

**How to test:**
- ✅ Run existing maxpool tests to ensure no regression
- ✅ Test with explicit default parameters to verify same behavior
- ✅ Test parameter validation with invalid inputs

**Status:** COMPLETED - API parameters already exist in C++ and Python bindings with proper defaults.

**Test Coverage:** Added comprehensive parametrized test `test_max_pool2d_output_formats_and_layouts` that:
- ✅ Tests BF16 + ROW_MAJOR (passes - current implementation)
- ✅ Tests BF16 + TILE (passes - current implementation)
- ❌ Tests FLOAT32 + layouts (fails with clear validation error at `generic_pools.cpp:231`)
- ❌ Tests BFLOAT8_B + layouts (fails with clear validation error)

### Step 2: Update Output Shape Calculation
**What needs to be changed:**
- Modify output tensor shape calculation based on layout choice
- For tiled layout: adjust dimensions to tile boundaries
- For row-major layout: keep existing logic

**Why:**
- Tiled layout requires different memory layout and addressing
- Shape must match the chosen layout for proper memory allocation

**How to test:**
- Unit tests for shape calculation with both layouts
- Verify output dimensions match expected values
- Test edge cases (non-tile-aligned inputs)

### Step 3: Update Circular Buffer Allocation
**What needs to be changed:**
- Modify CB allocation logic to account for different output formats
- Adjust buffer sizes based on tile vs stick requirements
- Update the additional output buffer (CB) sizing

**Why:**
- Tiled output requires different memory patterns and buffer sizes
- CB allocation must match the compute kernel expectations

**How to test:**
- Memory allocation tests for both formats
- Verify CB sizes are correct for different input sizes
- Test with various batch sizes and spatial dimensions

### Step 4: Update Compute Kernel Logic
**What needs to be changed:**
- Make tilization conditional based on output format parameter
- Preserve existing stick-based output for row-major
- Use existing tilization logic for tiled output

**Why:**
- Kernel must produce output in the requested format
- Conditional logic avoids breaking existing functionality

**How to test:**
- Kernel unit tests with both output formats
- Verify output data matches expected layout
- Performance comparison between formats

### Step 5: Update Output Tensor Creation
**What needs to be changed:**
- Create output tensor with correct format and layout
- Update tensor metadata to reflect chosen parameters
- Ensure proper memory layout specification

**Why:**
- Output tensor must match the actual data layout produced
- Downstream operations need correct tensor metadata

**How to test:**
- Tensor creation tests with both formats
- Verify metadata correctness
- Test tensor compatibility with subsequent operations

### Step 6: Integration Testing
**What needs to be changed:**
- Comprehensive end-to-end tests
- Performance benchmarking
- Documentation updates

**Why:**
- Ensure complete functionality works correctly
- Verify performance characteristics
- Document new features for users

**How to test:**
- Full pipeline tests with realistic models
- Performance regression testing
- Accuracy verification against reference implementation

## Success Criteria
- All existing tests pass (no regression)
- New functionality works correctly with both output formats
- Performance is acceptable for both formats
- Clear documentation for new parameters

## Hang Recovery Mechanism
During development, system hangs may occur when testing new functionality. To handle this:

### Setup (First Time Only):
1. Create new terminal instance
2. `cd ~/tt-smi`
3. `pip install .`
4. Keep this terminal open throughout development

### Recovery Process (Every Time Hang Occurs):
1. In the tt-smi terminal: `tt-smi -r 1`
2. Wait for device reset
3. Continue development

### Timeout Configuration:
- Set hang detection timeout to 30 seconds
- Always keep tt-smi terminal instance active
- Use this mechanism before continuing to next development steps

**⚠️ REMINDER: Apply hang recovery mechanism every time we continue development work**
