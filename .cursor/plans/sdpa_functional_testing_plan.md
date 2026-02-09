# SDPA Functional Testing Plan

**Repository:** tt-metal
**Branch:** ncvetkovic/dnijemcevic_sdpa_benchmark
**Hardware Platform:** Blackhole
**Role:** LLK and Compute API developer creating functional tests for SDPA operations

## Executive Summary

Create a SINGLE, focused, functionally correct test that runs all SDPA (Scaled Dot-Product Attention) operations in sequence to verify Compute API changes. Build the test incrementally, adding operations one at a time: QK matmul, then sub_exp, then reduce_c, then QKV matmul. The test must produce correct results and be easily configurable via loop multipliers.

**CRITICAL:** All operations run in ONE test sequentially, not as separate independent tests.

## Motivation

The existing `sdpa_single_core` test only validates synchronization, not functional correctness. We need to ensure that Compute API layer changes and function call modifications produce correct SDPA operation results.

Making the existing `sdpa_single_core` test functionally correct is challenging due to its scalability design (many matmul shapes, iterations, heads, Q/K/V chunks). Computing golden references for all variations is hard.

**Solution:** Create a simpler, focused functional test that runs SDPA operations sequentially (QK matmul → sub_exp → reduce_c → QKV matmul) without all the surrounding loop complexity. All operations run in the same test, passing data through the pipeline.

## Build Instructions

### Environment Setup
All commands must be run from the repository root (`/localdev/ncvetkovic/djordje/tt-metal`).

**Required environment variables:**
```bash
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
```

### Building the Project
After making code changes, build with:
```bash
./build_metal.sh --build-programming-example --build-tests
```

**Build flags explained:**
- `--build-programming-example`: Builds programming examples (including sdpa_single_core reference)
- `--build-tests`: Builds test suite (including our new sequential SDPA test)

**Important notes:**
- Run build from repository root
- Both flags are needed to build tests and reference examples
- Build time varies; check terminal output for completion
- Kernel changes are picked up automatically; rebuild only when changing test `.cpp` files

## Reference Files

### Template Files (Use as Pattern)
- **Test template:** `tests/tt_metal/tt_metal/test_sdpa_reduce_c.cpp`
- **Kernel template:** `tests/tt_metal/tt_metal/test_kernels/misc/sdpa/reduce_c/compute.cpp`

### Source Reference Files
- **Source kernel:** `tt_metal/programming_examples/sdpa_single_core/kernels/compute/sdpa.cpp`
- **Source test:** `tt_metal/programming_examples/sdpa_single_core/sdpa_single_core.cpp`

## Development Sequence

**IMPORTANT:** Build ONE test incrementally by adding operations sequentially. Each phase adds a new operation to the existing test.

### Phase 1: QK Matmul Operation ⭐ **START HERE**

**Add operation:** Q@K^T matrix multiplication
**Output:** Single 2x4 output block
**Block size:** 2x4 subblocks

**Verification criteria:**
- Test compiles and runs without hangs
- QK matmul results match golden values

**Key considerations:**
- This is the first operation in the test/SDPA pipeline
- Establishes the test structure and circular buffer flow
- Validates basic matmul operation with transpose
- Output becomes input for sub_exp in Phase 2

### Phase 2: Add Sub-Exp Operation

**Add operation:** `sub_exp_block_bcast_cols_inplace_2x4`
**Block size:** 2x4 subblocks
**Input:** Output from Phase 1 (QK matmul result)

**Verification criteria:**
- Correct exponential computation on QK results
- Correct subtraction with broadcast
- Results match golden values

**Key considerations:**
- Operates on QK matmul output from Phase 1
- Tests in-place operation behavior
- Validates column broadcast functionality
- Critical for attention score normalization
- Output becomes input for reduce_c in Phase 3

### Phase 3: Add Reduce-C Operation

**Add operation:** `reduce_c_row_pair`
**Input:** Output from Phase 2 (sub_exp result)

**Reference:** Can use patterns from existing standalone test
- Reference test: `tests/tt_metal/tt_metal/test_sdpa_reduce_c.cpp`
- Reference kernel: `tests/tt_metal/tt_metal/test_kernels/misc/sdpa/reduce_c/compute.cpp`

**Verification criteria:**
- Correct reduction across columns on sub_exp output
- Row-pair reduction behavior verified
- Results match golden values

**Key considerations:**
- Operates on sub_exp output from Phase 2
- Output used for normalization in Phase 4

### Phase 4: Add QKV Matmul Operation

**Add operation:** (QK^T)@V matrix multiplication
**Block size:** 2x4 subblocks
**Input:** Normalized results using reduce_c output from Phase 3

**Verification criteria:**
- Correct final matmul results
- Results match golden values
- Complete SDPA pipeline produces correct attention output

**Key considerations:**
- Final operation in SDPA pipeline
- Uses normalized attention scores from previous phases
- Produces final attention-weighted values
- Completes the full sequential SDPA operation test

## Test Requirements

### Critical Requirements

1. **Functional Correctness (CRITICAL)**
   - Results MUST match expected golden values
   - This is the primary success criterion
   - Any discrepancy indicates API or implementation issue

2. **Configurable Loop Multipliers**
   - Add loop multipliers around each operation call
   - Example: `number_of_sub_exp_iterations`
   - Single parameter change propagates to:
     - Golden value computation
     - Reader kernel
     - Writer kernel
     - Compute kernel

3. **Easy Adjustability**
   - Block shapes can be modified easily
   - Function arguments can be changed
   - Iteration counts adjustable
   - Changes maintain functional correctness

4. **Automatic Propagation**
   - Multiplier changes automatically update all components
   - No manual synchronization needed across kernels
   - Reduces error-prone manual updates

## Hardware Constraints

### Thread Model
- **UNPACK thread (ID 0):** Unpacks data from circular buffers
- **MATH thread (ID 1):** Performs mathematical operations
- **PACK thread (ID 2):** Packs results back to circular buffers

**CRITICAL:** All three threads run IN PARALLEL, not sequentially. Thread names describe their function, not execution order.

### Block Sizes
- Standard subblock: 2x4 for matmul operations
- Focus on individual operations, not full `sdpa_inner_loop` complexity

### Architecture Support
- **Primary target:** Blackhole (next-gen)
- **Secondary target:** Wormhole B0 (current production)
- Both architectures require synchronized changes when modifying LLK layer

## Test Structure Pattern

### File Organization
```
tests/tt_metal/tt_metal/
├── test_sdpa_sequential.cpp            # Main test file (single test for all ops)
└── test_kernels/misc/sdpa/sequential/
    ├── reader.cpp                       # Data reader kernel
    ├── compute.cpp                      # Compute kernel (all ops in sequence)
    └── writer.cpp                       # Output writer kernel
```

### Incremental Build Workflow
1. Create initial test with first operation (QK matmul with 2x4 output block)
2. Build the project using the build instructions above
3. Verify test compiles, runs without hanging, produces correct results
4. Add next operation to the same test (sub_exp operating on QK output)
5. Rebuild and verify updated test produces correct results for both operations
6. Add next operation (reduce_c operating on sub_exp output)
7. Continue until all operations are in the single sequential test

**Build reminder:** After each code change, rebuild from repo root with:
```bash
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
./build_metal.sh --build-programming-example --build-tests
```

### Loop Multiplier Benefits
- Easy to change operation shapes/blocks
- Easy to modify function arguments
- Easy to adjust iteration counts
- Maintains functional correctness with minimal changes

## Implementation Guidelines

### Golden Value Generation
- Compute expected results using reference implementation
- Use same loop multipliers as kernel execution
- Validate against known correct values
- Consider numerical precision limits

### Circular Buffer Management
- Proper buffer allocation for input/output
- Correct sizing based on block dimensions and multipliers
- Thread synchronization points

### Kernel Configuration
- Match compile-time parameters to test requirements
- Use appropriate data formats (bfloat16, fp32 accumulation)
- Configure destination accumulation mode correctly

## Testing & Validation

### Smoke Testing
1. Does test compile without errors?
2. Does test run without hanging?
3. Does board remain responsive?

### Functional Testing
1. Do results match golden values within tolerance?
2. Are edge cases handled correctly?
3. Do different loop multipliers produce correct scaled results?

### Regression Testing
1. Do changes break existing tests?
2. Are both architectures still synchronized?
3. Do pre-commit hooks pass?

## Recovery Procedures

### Test Hangs
- **Primary method:** `tt-smi -r` (reset board)
- Used for SDPA single core test hangs
- Restores board to clean state

### Build Issues
- **When to rebuild:** Only when changing test `.cpp` files, NOT kernel files
- Kernel changes are picked up automatically during next build
- Check pre-commit hook failures for formatting issues
- If build fails, check last 50 lines of terminal output for errors
- Make sure environment variables are set: `TT_METAL_HOME` and `PYTHONPATH`

## Key Principles

1. **Functional correctness is critical** - Results must match expected golden values
2. **Simplicity over completeness** - Don't need to test full `sdpa_inner_loop` complexity
3. **Isolated testing** - Focus on individual operations without surrounding complexity
4. **Incremental development** - Build one test at a time, verify, then move to next
5. **Configurability** - Use loop multipliers for easy test adjustment

## Success Criteria

### Per-Phase Success
- Test compiles without errors after adding new operation
- Test runs without hanging with all operations so far
- Results match golden values within numerical tolerance for all operations
- Test remains configurable via loop multipliers
- Data flows correctly from one operation to the next

### Overall Success
- Single test runs all four SDPA operations sequentially
- Test produces functionally correct results matching golden pipeline output
- Test can be easily adjusted for different scenarios via loop multipliers
- Test serves as validation for Compute API changes
- Test provides clear pass/fail indication for the full SDPA sequence

## Next Steps

### Getting Started

1. **Set up environment** (from repo root):
   ```bash
   cd /localdev/ncvetkovic/djordje/tt-metal
   export TT_METAL_HOME=$(pwd)
   export PYTHONPATH=$(pwd)
   ```

2. **Start with Phase 1** (Add QK Matmul operation to new test):
   - Use `test_sdpa_reduce_c.cpp` as template for test structure
   - Implement golden value generation for Q@K^T
   - Create compute kernel with QK matmul operation

3. **Build the test**:
   ```bash
   ./build_metal.sh --build-programming-example --build-tests
   ```

4. **Run and verify**:
   - Test compiles without errors
   - Test runs without hanging
   - QK matmul results match golden values

5. **Proceed to Phase 2** (add sub_exp to the same test)

6. **Continue iteratively**: Add operations one at a time, rebuilding and verifying after each addition

## Notes

- **Single test, multiple operations:** All SDPA operations run sequentially in ONE test
- Focus on getting each operation functionally correct before adding the next
- Use existing `reduce_c` test as reference for structure, but adapt for sequential execution
- This is an integration test of the SDPA pipeline, not separate unit tests
- Data flows through operations: QK matmul → sub_exp → reduce_c → QKV matmul
- Board reset is normal for development - don't worry about occasional hangs
- Pre-commit hooks will format code automatically
- Test name suggestion: `test_sdpa_sequential.cpp` or `test_sdpa_pipeline.cpp`
