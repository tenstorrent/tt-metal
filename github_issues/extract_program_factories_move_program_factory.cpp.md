# Extract Program Factories from move_program_factory.cpp

## Overview
This file contains 2 program factory functions that need to be extracted into separate files as part of the ProgramDescriptor refactoring effort.

**File:** `ttnn/cpp/ttnn/operations/data_movement/move/device/move_program_factory.cpp`

## Program Factories to Extract

### 1. move_multi_core_with_overlap
- **Function:** `move_multi_core_with_overlap`
- **Line:** 67
- **Status:** DONE

### 2. move_multi_core_sharded
- **Function:** `move_multi_core_sharded`
- **Line:** 210
- **Status:** DONE

## Tasks

- [x] Analyze dependencies and shared code between the 2 program factories
- [x] Create separate files for each program factory
- [x] Extract each factory function to its own file
- [x] Update includes and dependencies
- [x] Ensure no functionality is lost during extraction
- [x] Update any references to these functions
- [x] Test that all extracted factories work correctly

## Status: COMPLETED ✅

**Branch:** `ai-split-move-program-factory`
**Commit:** 6017889681
**Build Status:** ✅ Successfully builds with `./build_metal.sh --debug`

### Changes Made:
- Created `move_multi_core_with_overlap_program_factory.hpp` and `.cpp`
- Created `move_multi_core_sharded_program_factory.hpp` and `.cpp`
- Updated original file to include the new extracted files
- Updated CMakeLists.txt to compile new source files
- Removed extracted functions from the original file

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
