# Extract Program Factories from untilize_with_unpadding_program_factory.cpp

## Overview
This file contains 5 program factory functions that need to be extracted into separate files as part of the ProgramDescriptor refactoring effort.

**File:** `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_program_factory.cpp`

## Program Factories to Extract

### 1. untilize_with_unpadding_single_core
- **Function:** `untilize_with_unpadding_single_core`
- **Line:** 24
- **Status:** TODO

### 2. untilize_with_unpadding_multi_core_block_interleaved
- **Function:** `untilize_with_unpadding_multi_core_block_interleaved`
- **Line:** 217
- **Status:** TODO

### 3. untilize_with_unpadding_multi_core_col_interleaved
- **Function:** `untilize_with_unpadding_multi_core_col_interleaved`
- **Line:** 505
- **Status:** TODO

### 4. untilize_with_unpadding_multi_core_interleaved
- **Function:** `untilize_with_unpadding_multi_core_interleaved`
- **Line:** 662
- **Status:** TODO

### 5. untilize_with_unpadding_multi_core_sharded
- **Function:** `untilize_with_unpadding_multi_core_sharded`
- **Line:** 882
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
