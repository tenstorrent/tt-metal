# Extract Program Factories from pad_program_factory.cpp

## Overview
This file contains 6 program factory functions that need to be extracted into separate files as part of the ProgramDescriptor refactoring effort.

**File:** `ttnn/cpp/ttnn/operations/data_movement/pad/device/pad_program_factory.cpp`

## Program Factories to Extract

### 1. pad_rm_reader_writer
- **Function:** `pad_rm_reader_writer`
- **Line:** 23
- **Status:** TODO

### 2. pad_tile
- **Function:** `pad_tile`
- **Line:** 179
- **Status:** TODO

### 3. pad_rm_reader_writer_multi_core
- **Function:** `pad_rm_reader_writer_multi_core`
- **Line:** 449
- **Status:** TODO

### 4. pad_rm_reader_writer_multi_core_v2
- **Function:** `pad_rm_reader_writer_multi_core_v2`
- **Line:** 797
- **Status:** TODO

### 5. pad_rm_sharded_height_only
- **Function:** `pad_rm_sharded_height_only`
- **Line:** 1183
- **Status:** TODO

### 6. pad_rm_sharded_width_only
- **Function:** `pad_rm_sharded_width_only`
- **Line:** 1361
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
