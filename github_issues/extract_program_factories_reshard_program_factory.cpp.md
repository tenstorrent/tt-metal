# Extract Program Factories from reshard_program_factory.cpp

## Overview
This file contains 3 program factory functions that need to be extracted into separate files as part of the ProgramDescriptor refactoring effort.

**File:** `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory.cpp`

## Program Factories to Extract

### 1. reshard_multi_core_same_width
- **Function:** `reshard_multi_core_same_width`
- **Line:** 296
- **Status:** TODO

### 2. reshard_multi_core_generic
- **Function:** `reshard_multi_core_generic`
- **Line:** 455
- **Status:** TODO

### 3. reshard_multi_core_same_height
- **Function:** `reshard_multi_core_same_height`
- **Line:** 562
- **Status:** TODO

## Tasks

- [ ] Analyze dependencies and shared code between the 3 program factories
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
