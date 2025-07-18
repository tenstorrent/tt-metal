# Extract Program Factories from transpose_program_factory.cpp

## Overview
This file contains 5 program factory functions that need to be extracted into separate files as part of the ProgramDescriptor refactoring effort.

**File:** `ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_program_factory.cpp`

## Program Factories to Extract

### 1. transpose_hc_multi_core
- **Function:** `transpose_hc_multi_core`
- **Line:** 624
- **Status:** TODO

### 2. transpose_hc_multi_core_sharded
- **Function:** `transpose_hc_multi_core_sharded`
- **Line:** 1148
- **Status:** TODO

### 3. transpose_wh_multi_core
- **Function:** `transpose_wh_multi_core`
- **Line:** 1511
- **Status:** TODO

### 4. transpose_wh_multi_core_sharded
- **Function:** `transpose_wh_multi_core_sharded`
- **Line:** 1755
- **Status:** TODO

### 5. transpose_wh_multi_core_sharded_rm
- **Function:** `transpose_wh_multi_core_sharded_rm`
- **Line:** 1964
- **Status:** TODO

## Tasks

- [ ] Analyze dependencies and shared code between the 5 program factories
- [ ] Create separate files for each program factory
- [ ] Extract each factory function to its own file
- [ ] Update includes and dependencies
- [ ] Ensure no functionality is lost during extraction
- [ ] Update any references to these functions
- [ ] Test that all extracted factories work correctly

## Notes
- This file contains multiple program factories that should be separated for better maintainability
- Each factory should be moved to its own file following the new ProgramDescriptor pattern
- Consider shared utilities or common code that might need to be extracted as well

## Labels
- `ai-split-program-factory`
- `refactoring`
- `program-descriptor`

---
*Generated on 2025-07-10 22:35:58*
