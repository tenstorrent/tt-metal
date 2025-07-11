# Extract Program Factories from untilize_program_factory.cpp

## Overview
This file contains 6 program factory functions that need to be extracted into separate files as part of the ProgramDescriptor refactoring effort.

**File:** `ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_program_factory.cpp`

## Program Factories to Extract

### 1. untilize_multi_core_sub_core_grids
- **Function:** `untilize_multi_core_sub_core_grids`
- **Line:** 35
- **Status:** TODO

### 2. untilize_multi_core_parallelize_column
- **Function:** `untilize_multi_core_parallelize_column`
- **Line:** 220
- **Status:** TODO

### 3. untilize_multi_core_block
- **Function:** `untilize_multi_core_block`
- **Line:** 453
- **Status:** TODO

### 4. untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical
- **Function:** `untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical`
- **Line:** 740
- **Status:** TODO

### 5. untilize_multi_core
- **Function:** `untilize_multi_core`
- **Line:** 867
- **Status:** TODO

### 6. untilize_single_core
- **Function:** `untilize_single_core`
- **Line:** 1313
- **Status:** TODO

## Tasks

- [ ] Analyze dependencies and shared code between the 6 program factories
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
