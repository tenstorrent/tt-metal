# AI Agent Implementation Plan: Block Variants (Condensed)

**Task**: Add `*_block` and `pack_*_block` variants to tt-metal Compute API
**Reference**: [Issue #35739](https://github.com/tenstorrent/tt-metal/issues/35739)
**Branch**: `ncvetkovic/35739_add_missing_functions`
**Repository**: `/localdev/ncvetkovic/reconfig/tt-metal`

## Critical Rules

1. **Block = For-Loop**: Block variants are ONLY for-loops over existing `*_tile()` functions
2. **No New Inits**: Use existing init functions (e.g., `add_tiles_init()`)
3. **No New LLK Calls**: Only call existing single-tile functions
4. **Template Size**: Always `template<uint32_t Ht, uint32_t Wt>`
5. **Capacity Check**: Always `static_assert(Ht * Wt <= 16)`
6. **WIP Mark**: All functions marked "WORK IN PROGRESS - Use with caution"

## Phase 1: Inventory (Agent 1)

### Input
Compute API headers: `tt_metal/include/compute_kernel_api/*.h`

### Actions
1. **Find tile operations**:
   ```bash
   grep "ALWI void.*_tile\(" tt_metal/include/compute_kernel_api/*.h
   ```
2. **Check existing blocks**:
   ```bash
   grep "_block\(" tt_metal/include/compute_kernel_api/*.h
   ```
3. **Create inventory**: List operations needing block variants

### Output
`inventory.json`:
```json
{
  "eltwise_binary.h": ["add_tiles", "sub_tiles", "mul_tiles"],
  "reduce.h": ["reduce_tile"],
  "pack.h": ["pack_tile"]
}
```

## Phase 2: Template Generation (Agent 2)

### Template Pattern

**For Binary Operations** (e.g., `add_block`):
```cpp
template <uint32_t Ht, uint32_t Wt>
ALWI void add_block(uint32_t cb0, uint32_t cb1, uint32_t tile0_start, uint32_t tile1_start, uint32_t dst_start) {
    static_assert(Ht * Wt <= 16, "Block exceeds DEST capacity");
    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t w = 0; w < Wt; w++) {
            uint32_t offset = h * Wt + w;
            add_tiles(cb0, cb1, tile0_start + offset, tile1_start + offset, dst_start + offset);
        }
    }
}
```

**For Reduce Operations**:
```cpp
template <PoolType type = REDUCE_OP, ReduceDim dim = REDUCE_DIM, uint32_t Ht, uint32_t Wt>
ALWI void reduce_block(uint32_t cb, uint32_t cb_scaler, uint32_t tile_start, uint32_t tile_scaler, uint32_t dst_start) {
    static_assert(Ht * Wt <= 16, "Block exceeds DEST capacity");
    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t w = 0; w < Wt; w++) {
            uint32_t offset = h * Wt + w;
            reduce_tile<type, dim>(cb, cb_scaler, tile_start + offset, tile_scaler, dst_start + offset);
        }
    }
}
```

**For Pack Operations**:
```cpp
template <uint32_t Ht, uint32_t Wt>
ALWI void pack_block(uint32_t dst_start, uint32_t cb_out) {
    static_assert(Ht * Wt <= 16, "Block exceeds DEST capacity");
    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t w = 0; w < Wt; w++) {
            pack_tile(dst_start + h * Wt + w, cb_out);
        }
    }
}
```

### Doc Comment Template
```cpp
// clang-format off
/**
 * WORK IN PROGRESS - Use with caution
 *
 * L1 → DEST: Block-level [operation]. For-loop wrapper around [single_tile_func]().
 * Result stays in DEST. Conforms to Compute API Contract.
 *
 * | Argument   | Description              | Type     | Valid Range | Required |
 * |------------|--------------------------|----------|-------------|----------|
 * | Ht         | Block height in tiles    | uint32_t | 1 to 16     | True     |
 * | Wt         | Block width in tiles     | uint32_t | 1 to 16     | True     |
 * | ...        | [other params]           | ...      | ...         | ...      |
 */
// clang-format on
```

### Validation Checklist
- ✅ `static_assert(Ht * Wt <= 16)` present
- ✅ Template parameters `<uint32_t Ht, uint32_t Wt>` (or with operation params first)
- ✅ Calls existing single-tile function
- ✅ Uses `ALWI` macro
- ✅ Has complete doc comment
- ✅ "WORK IN PROGRESS" in doc

### Output
Generate templates for: `add_block`, `sub_block`, `mul_block`, `reduce_block`, `pack_block`

## Phase 3: Code Integration (Agent 3)

### Insertion Rules
- Insert AFTER corresponding tile function
- Insert BEFORE `}  // namespace ckernel`
- Group related functions together
- Maintain file structure

### Actions
1. **For each template**:
   - Locate insertion point in target file
   - Use `search_replace` to insert code
   - Verify with `read_lints`

2. **Files to modify**:
   - `eltwise_binary.h`: Add `add_block`, `sub_block`, `mul_block`
   - `reduce.h` or `reduce_custom.h`: Add `reduce_block`
   - `pack.h`: Add `pack_block`, `pack_reduce_block`

### Output
Modified header files with block variants added

## Phase 4: Documentation (Agent 4)

### Create
1. **BLOCK_VARIANTS_API.md**: API reference with usage examples
2. **Update TASK.md**: Add "Completed Functions" section

### Output
Complete documentation set

## Phase 5: Testing (Agent 5)

### Test Plan
Test that `*_block<H,W>()` produces same result as H×W calls to `*_tile()`

### Test Sizes
1x1, 2x2, 2x4, 4x4 for each operation

### Output
Test suite (optional for Phase 1)

## Phase 6: Build & Verify (Agent 6)

### Build Process
```bash
cd /localdev/ncvetkovic/reconfig/tt-metal
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
./build_metal.sh
```

### Verification Steps
1. **Syntax**: `clang-format --dry-run --Werror <files>`
2. **Lints**: Use `read_lints` on modified files
3. **Compile**: Full build completes without errors
4. **Tests**: Run test suite (if implemented)

### Success Criteria
- [ ] Zero compilation errors
- [ ] Zero linter errors
- [ ] clang-format passes
- [ ] All `static_assert` present
- [ ] All "WORK IN PROGRESS" markings present
- [ ] No new init functions added

### Output
Build success confirmation + verification report

## Phase 7: Final Review (Agent 7)

### Review Checklist

**Per Function**:
- [ ] Correct template signature
- [ ] `static_assert(Ht * Wt <= 16)` present
- [ ] Calls existing tile function (name matches pattern)
- [ ] Parameters = tile function params + start indices
- [ ] Complete doc comment with table
- [ ] "WORK IN PROGRESS" warning
- [ ] No LLK calls (only tile function calls)
- [ ] For-loop: `offset = h * Wt + w`

**Per File**:
- [ ] All insertions after corresponding tile functions
- [ ] Proper namespace closure maintained
- [ ] Consistent formatting
- [ ] No new includes needed

**Compute API Contract Conformance**:
```
✓ *_block: UNPACK+MATH, up to 16 tiles, result in DEST, manual sync
✓ pack_*_block: PACK only, up to 16 tiles, DEST→L1
```

### Generate Summary
Create `IMPLEMENTATION_SUMMARY.md` with:
- Functions added count
- Files modified list
- Lines of code added
- Build status
- Test coverage (if applicable)
- Next steps

### Output
Final approval + comprehensive summary

## Execution Flow

```
Phase 1 (Inventory)
    ↓ inventory.json
Phase 2 (Templates)
    ↓ templates/
Phase 3 (Integration)
    ↓ modified headers
Phase 4 (Documentation)
    ↓ docs/
Phase 5 (Testing)
    ↓ tests/
Phase 6 (Build & Verify)
    ↓ build confirmation
Phase 7 (Final Review)
    ↓ APPROVED / REVISIONS NEEDED
```

## Rollback Strategy

If any phase fails:
1. Git stash or revert changes from failed phase
2. Log failure in `failure_report.txt`
3. Analyze root cause
4. Fix issue
5. Restart from failed phase

## Expected Deliverables

1. **Code**: 10-20 new block variant functions
2. **Documentation**: API reference + usage guide
3. **Tests**: Test suite (optional but recommended)
4. **Summary**: Complete implementation report
5. **Build**: Successful compilation on both Blackhole and Wormhole B0

## Estimated Effort

- Phase 1: 15 minutes
- Phase 2: 30 minutes
- Phase 3: 45 minutes
- Phase 4: 30 minutes
- Phase 5: 60 minutes (if full tests)
- Phase 6: 20 minutes (build time)
- Phase 7: 15 minutes

**Total**: ~3.5 hours for complete implementation
