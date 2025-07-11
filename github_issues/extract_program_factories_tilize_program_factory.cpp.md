# Extract Program Factories from tilize_program_factory.cpp

## Overview
This file contains 4 program factory functions that need to be extracted into separate files as part of the ProgramDescriptor refactoring effort.

**File:** `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_program_factory.cpp`

## Program Factories to Extract

### 1. tilize_single_core
- **Function:** `tilize_single_core`
- **Line:** 21
- **Status:** TODO

### 2. tilize_multi_core_block
- **Function:** `tilize_multi_core_block`
- **Line:** 162
- **Status:** TODO

### 3. tilize_multi_core_interleaved
- **Function:** `tilize_multi_core_interleaved`
- **Line:** 437
- **Status:** TODO

### 4. tilize_multi_core_sharded
- **Function:** `tilize_multi_core_sharded`
- **Line:** 628
- **Status:** TODO

## Tasks

- [ ] Analyze dependencies and shared code between the 4 program factories
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
