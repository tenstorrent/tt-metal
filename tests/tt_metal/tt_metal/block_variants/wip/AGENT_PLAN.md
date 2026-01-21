# AI Agent Implementation Plan: Block Variants for Compute API

## Meta Information

**Task**: Add missing `*_block` and `pack_*_block` variants to tt-metal Compute API
**Reference**: [GitHub Issue #35739](https://github.com/tenstorrent/tt-metal/issues/35739)
**Branch**: `ncvetkovic/35739_add_missing_functions`
**Repository**: `/localdev/ncvetkovic/reconfig/tt-metal`
**Prerequisites**: Read `CLAUDE.md`, `TASK.md`, `API_Abstraction_Layers.md`, `Low Level Contract and API Split.txt`

## Critical Constraints

1. **Block variants are FOR-LOOPS ONLY** - Do NOT create new init functions or LLK calls
2. **Use existing single-tile functions** - Call `*_tiles()` or `*_tile()` in a loop
3. **Template parameters for block size** - `Ht` (height) and `Wt` (width)
4. **DEST capacity limit** - Always `static_assert(Ht * Wt <= 16)`
5. **WIP marking** - All functions marked "WORK IN PROGRESS - Use with caution"
6. **Architecture agnostic** - Code works on both Blackhole and Wormhole B0
7. **No new LLK primitives** - Only use existing llk_* functions

## Phase 1: Discovery and Inventory (Agent 1)

### Step 1.1: Identify Existing Tile Operations
**Input**: Compute API header files
**Output**: `inventory.json`

```bash
# Scan all compute API headers
grep -r "ALWI void.*_tiles\|ALWI void.*_tile" \
  tt_metal/include/compute_kernel_api/*.h \
  --include="*.h" -n
```

**Expected Operations**:
- Eltwise binary: `add_tiles`, `sub_tiles`, `mul_tiles`, `div_tiles`
- Eltwise unary: `copy_tile`, `relu_tiles`, etc.
- Reduce: `reduce_tile`
- Matmul: `matmul_tiles`
- Tilize/Untilize operations
- Transpose operations

**Action**: Create JSON inventory:
```json
{
  "eltwise_binary": {
    "file": "eltwise_binary.h",
    "operations": [
      {"name": "add_tiles", "params": ["icb0", "icb1", "itile0", "itile1", "idst"]},
      {"name": "sub_tiles", "params": ["icb0", "icb1", "itile0", "itile1", "idst"]},
      {"name": "mul_tiles", "params": ["icb0", "icb1", "itile0", "itile1", "idst"]}
    ]
  },
  "reduce": {
    "file": "reduce.h",
    "operations": [
      {"name": "reduce_tile", "params": ["icb", "icb_scaler", "itile", "itile_scaler", "idst"],
       "templates": ["PoolType", "ReduceDim"]}
    ]
  }
}
```

### Step 1.2: Check for Existing Block Variants
**Input**: Compute API headers
**Output**: `existing_blocks.txt`

```bash
# Find existing *_block functions
grep -r "_block\(" tt_metal/include/compute_kernel_api/*.h \
  --include="*.h" -A 5
```

**Action**: Document what already exists to avoid duplication

### Step 1.3: Determine Missing Block Variants
**Input**: `inventory.json`, `existing_blocks.txt`
**Output**: `missing_blocks.json`

**Logic**:
```
for each operation in inventory:
  if operation NOT in existing_blocks:
    add to missing_blocks
```

## Phase 2: Template Generation (Agent 2)

### Step 2.1: Generate Block Variant Templates
**Input**: `missing_blocks.json`
**Output**: `templates/*.cpp` (one per operation category)

**Template Structure for Eltwise Binary**:
```cpp
// File: templates/eltwise_binary_block.cpp
// Location in real code: tt_metal/include/compute_kernel_api/eltwise_binary.h

// clang-format off
/**
 * WORK IN PROGRESS - Use with caution
 *
 * L1 → DEST: Block-level element-wise addition.
 * This is a for-loop wrapper around add_tiles().
 * Result stays in DEST. Conforms to Compute API Contract.
 *
 * | Argument   | Description                           | Type     | Valid Range     | Required |
 * |------------|---------------------------------------|----------|-----------------|----------|
 * | Ht         | Block height in tiles                 | uint32_t | 1 to 16         | True     |
 * | Wt         | Block width in tiles                  | uint32_t | 1 to 16         | True     |
 * | icb0       | CB containing operand A               | uint32_t | 0 to 31         | True     |
 * | icb1       | CB containing operand B               | uint32_t | 0 to 31         | True     |
 * | itile0_start| Starting tile index for A            | uint32_t | < CB size       | True     |
 * | itile1_start| Starting tile index for B            | uint32_t | < CB size       | True     |
 * | idst_start | Starting DEST index                   | uint32_t | < 16            | True     |
 */
// clang-format on
template <uint32_t Ht, uint32_t Wt>
ALWI void add_block(
    uint32_t icb0, uint32_t icb1,
    uint32_t itile0_start, uint32_t itile1_start,
    uint32_t idst_start) {

    static_assert(Ht * Wt <= 16, "Block size Ht * Wt exceeds DEST capacity (max 16 tiles)");

    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t w = 0; w < Wt; w++) {
            uint32_t offset = h * Wt + w;
            add_tiles(icb0, icb1,
                     itile0_start + offset,
                     itile1_start + offset,
                     idst_start + offset);
        }
    }
}

// Repeat for sub_block, mul_block, etc.
```

**Template Structure for Reduce**:
```cpp
// File: templates/reduce_block.cpp
// Location in real code: tt_metal/include/compute_kernel_api/reduce.h

template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, uint32_t Ht, uint32_t Wt>
ALWI void reduce_block(
    uint32_t icb, uint32_t icb_scaler,
    uint32_t itile_start, uint32_t itile_scaler,
    uint32_t idst_start) {

    static_assert(Ht * Wt <= 16, "Block size Ht * Wt exceeds DEST capacity (max 16 tiles)");

    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t w = 0; w < Wt; w++) {
            uint32_t offset = h * Wt + w;
            reduce_tile<reduce_type, reduce_dim>(
                icb, icb_scaler,
                itile_start + offset,
                itile_scaler,  // Note: scaler typically doesn't change per tile
                idst_start + offset);
        }
    }
}
```

**Template Structure for Pack Block**:
```cpp
// File: templates/pack_block.cpp
// Location in real code: tt_metal/include/compute_kernel_api/pack.h

template <uint32_t Ht, uint32_t Wt>
ALWI void pack_block(uint32_t idst_start, uint32_t ocb) {
    static_assert(Ht * Wt <= 16, "Block size Ht * Wt exceeds DEST capacity (max 16 tiles)");

    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t w = 0; w < Wt; w++) {
            uint32_t offset = h * Wt + w;
            pack_tile(idst_start + offset, ocb);
        }
    }
}
```

### Step 2.2: Validation Rules
**For Each Template**:
1. ✅ Has `static_assert(Ht * Wt <= 16, "...")`
2. ✅ Has "WORK IN PROGRESS" in doc comment
3. ✅ Template parameters `<uint32_t Ht, uint32_t Wt>` OR operation-specific templates first
4. ✅ Calls existing single-tile function inside for-loop
5. ✅ Parameters match single-tile function + start offsets
6. ✅ Uses `ALWI` macro (ALWAYS_INLINE)
7. ✅ Doc comment table with all parameters
8. ✅ No new init functions
9. ✅ No direct LLK calls (only through existing tile functions)

## Phase 3: Code Integration (Agent 3)

### Step 3.1: Locate Insertion Points
**Input**: `templates/*.cpp`, Compute API headers
**Output**: `insertion_plan.json`

**Rules for Insertion**:
- Insert AFTER existing tile functions
- Insert BEFORE closing namespace `}  // namespace ckernel`
- Group by operation type (all add_* together)
- Maintain alphabetical order within groups

**Example insertion_plan.json**:
```json
{
  "eltwise_binary.h": {
    "insertions": [
      {
        "after_line": "sub_tiles function",
        "template": "add_block",
        "order": 1
      },
      {
        "after_line": "add_block",
        "template": "sub_block",
        "order": 2
      }
    ]
  }
}
```

### Step 3.2: Insert Code
**Input**: `insertion_plan.json`, `templates/*.cpp`
**Action**: Use search_replace tool to insert templates at correct locations

**Verification After Each Insertion**:
```bash
# Check syntax
clang-format --dry-run --Werror tt_metal/include/compute_kernel_api/eltwise_binary.h

# Check for linter errors
# (use read_lints tool)
```

### Step 3.3: Update Pack Header
**Input**: `templates/pack_block.cpp`
**Action**: Add generic `pack_block` and operation-specific variants to `pack.h`

**Variants to Add**:
1. Generic `pack_block<Ht, Wt>()` - for standard operations
2. `pack_reduce_block<reduce_dim, Ht, Wt>()` - for reduce (if special handling needed)
3. Others as identified in Phase 1

## Phase 4: Testing Infrastructure (Agent 4)

### Step 4.1: Create Test Plan
**Input**: `missing_blocks.json`
**Output**: `test_plan.json`

**Test Plan Structure**:
```json
{
  "test_categories": [
    {
      "operation": "add_block",
      "file": "tests/tt_metal/tt_metal/test_add_block.cpp",
      "test_cases": [
        {"Ht": 1, "Wt": 1, "description": "Minimum block size"},
        {"Ht": 2, "Wt": 2, "description": "Small square block"},
        {"Ht": 2, "Wt": 4, "description": "Rectangular block"},
        {"Ht": 4, "Wt": 4, "description": "Maximum square block"}
      ]
    }
  ]
}
```

### Step 4.2: Generate Test Skeletons
**Output**: `tests/compute_api_block_variants/`

**Test Structure**:
```cpp
// tests/tt_metal/tt_metal/test_add_block.cpp
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

TEST(ComputeAPI, AddBlock_2x2) {
    constexpr uint32_t Ht = 2;
    constexpr uint32_t Wt = 2;

    // Test that add_block<2,2>() produces same result as
    // calling add_tiles() 4 times

    // Setup: Create input tensors
    // Execute: Call add_block in compute kernel
    // Verify: Compare against reference (4x add_tiles calls)
}
```

### Step 4.3: Create Testing Documentation
**Output**: `TESTING_GUIDE.md`

**Contents**:
- How to run tests: `pytest tests/compute_api_block_variants/ -v`
- Expected PCC thresholds: `> 0.9999`
- How to add new tests
- Architecture-specific considerations

## Phase 5: Documentation (Agent 5)

### Step 5.1: Generate Function Documentation
**Input**: All added functions
**Output**: `BLOCK_VARIANTS_API.md`

**Structure**:
```markdown
# Block Variants API Reference

## Eltwise Binary Operations

### add_block<Ht, Wt>
**Signature**:
```cpp
template <uint32_t Ht, uint32_t Wt>
ALWI void add_block(uint32_t icb0, uint32_t icb1,
                    uint32_t itile0_start, uint32_t itile1_start,
                    uint32_t idst_start)
```

**Description**: Block-level element-wise addition (L1 → DEST)

**Parameters**:
- `Ht`: Block height in tiles (compile-time, max total 16)
- `Wt`: Block width in tiles (compile-time, max total 16)
- `icb0`: Circular buffer ID for operand A (0-31)
- `icb1`: Circular buffer ID for operand B (0-31)
- `itile0_start`: Starting tile index in CB for A
- `itile1_start`: Starting tile index in CB for B
- `idst_start`: Starting DEST register index (0-15)

**Example Usage**:
```cpp
// Initialize (use existing init)
add_tiles_init(cb_a, cb_b);

// Process 2x4 block
acquire_dst();
add_block<2, 4>(cb_a, cb_b, 0, 0, 0);
release_dst();

// Pack result
pack_block<2, 4>(0, cb_out);
```

**Conforms To**: Compute API Contract `*_block` pattern
```

### Step 5.2: Update TASK.md
**Action**: Add "Completed Functions" section listing all implemented variants

### Step 5.3: Create Migration Guide
**Output**: `MIGRATION_GUIDE.md`

**Contents**:
- How to convert tile-by-tile loops to block variants
- Performance considerations
- When to use block vs. tile variants
- Example migrations

## Phase 6: Build and Verification (Agent 6)

### Step 6.1: Syntax Check
```bash
cd /localdev/ncvetkovic/reconfig/tt-metal

# Check C++ syntax
find tt_metal/include/compute_kernel_api/ -name "*.h" -exec \
  clang-format --dry-run --Werror {} \;
```

### Step 6.2: Full Build
```bash
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
./build_metal.sh
```

**Expected Output**: Build completes without errors

### Step 6.3: Linter Check
**Action**: Use `read_lints` tool on all modified files

**Expected**: Zero linter errors

### Step 6.4: Verification Checklist
- [ ] All files compile without errors
- [ ] No linter warnings
- [ ] clang-format passes
- [ ] All `static_assert` statements present
- [ ] All functions have "WORK IN PROGRESS" marking
- [ ] No new init functions were added
- [ ] All block functions call existing tile functions

## Phase 7: Final Review (Agent 7)

### Step 7.1: Code Review Checklist
**For Each Block Variant**:
- [ ] Template parameters correct (`<uint32_t Ht, uint32_t Wt>`)
- [ ] `static_assert(Ht * Wt <= 16)` present
- [ ] Calls existing `*_tiles()` or `*_tile()` function
- [ ] Parameters match single-tile function + start indices
- [ ] Doc comment complete with parameter table
- [ ] "WORK IN PROGRESS" warning present
- [ ] No new LLK calls
- [ ] No new init functions
- [ ] For-loop iterates correctly (row-major: `h * Wt + w`)

### Step 7.2: Contract Conformance Check
**Verify Each Function Conforms To**:
```
*_block pattern:
- Threads: UNPACK + MATH ✓
- Capacity: Up to 16 tiles ✓
- Result stays in DEST ✓
- DEST Sync: Manual ✓
- No packing ✓

pack_*_block pattern:
- Threads: PACK only ✓
- Capacity: Up to 16 tiles ✓
- Pattern: DEST → L1 ✓
```

### Step 7.3: Generate Summary
**Output**: `IMPLEMENTATION_SUMMARY.md`

**Contents**:
- Total functions added
- Files modified
- Lines of code added
- Architecture support verified
- Test coverage
- Known limitations
- Next steps

## Execution Order

1. **Sequential Phases**: Must complete Phase N before starting Phase N+1
2. **Parallel Within Phase**: Steps within a phase can run in parallel if independent
3. **Rollback on Failure**: If any phase fails, rollback to last known good state
4. **Incremental Commits**: Commit after each successful phase

## Success Criteria

- [ ] All missing block variants identified and implemented
- [ ] Zero compilation errors
- [ ] Zero linter errors
- [ ] All functions conform to Compute API Contract
- [ ] No new init functions added
- [ ] All functions use for-loops over existing tile operations
- [ ] Documentation complete
- [ ] Test infrastructure in place
- [ ] Build succeeds on both architectures

## Failure Recovery

If any phase fails:
1. Document the failure in `failure_log.txt`
2. Rollback changes from failed phase
3. Analyze root cause
4. Adjust plan if needed
5. Retry phase with fixes

## Agent Coordination

- **Agent 1 → Agent 2**: `inventory.json`, `missing_blocks.json`
- **Agent 2 → Agent 3**: `templates/*.cpp`
- **Agent 3 → Agent 4**: List of implemented functions
- **Agent 4 → Agent 5**: Test results and coverage data
- **Agent 5 → Agent 6**: Documentation files
- **Agent 6 → Agent 7**: Build logs and verification results
- **Agent 7**: Final approval and summary
