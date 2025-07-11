# Extract Program Factories from concat_program_factory.cpp

## Overview
This file contains 5 program factory functions that need to be extracted into separate files as part of the ProgramDescriptor refactoring effort.

**File:** `ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_program_factory.cpp`

## Program Factories to Extract

### 1. s2s_tiled_concat_two_tensors_height_multi_core
- **Function:** `s2s_tiled_concat_two_tensors_height_multi_core`
- **Line:** 35
- **Status:** TODO

### 2. s2s_rm_concat_two_tensors_height_multi_core
- **Function:** `s2s_rm_concat_two_tensors_height_multi_core`
- **Line:** 248
- **Status:** TODO

### 3. s2s_concat_multi_core
- **Function:** `s2s_concat_multi_core`
- **Line:** 445
- **Status:** TODO

### 4. s2i_rm_concat_multi_core
- **Function:** `s2i_rm_concat_multi_core`
- **Line:** 566
- **Status:** TODO

### 5. concat_multi_core
- **Function:** `concat_multi_core`
- **Line:** 726
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
