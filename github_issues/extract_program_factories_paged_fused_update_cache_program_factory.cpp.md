# Extract Program Factories from paged_fused_update_cache_program_factory.cpp

## Overview
This file contains 2 program factory functions that need to be extracted into separate files as part of the ProgramDescriptor refactoring effort.

**File:** `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/paged_fused_update_cache_program_factory.cpp`

## Program Factories to Extract

### 1. paged_tiled_fused_update_cache_multi_core
- **Function:** `paged_tiled_fused_update_cache_multi_core`
- **Line:** 69
- **Status:** TODO

### 2. paged_row_major_fused_update_cache_multi_core
- **Function:** `paged_row_major_fused_update_cache_multi_core`
- **Line:** 533
- **Status:** TODO

## Tasks

- [x] Analyze dependencies and shared code between the 2 program factories
- [x] Create separate files for each program factory
- [x] Extract each factory function to its own file
- [x] Update includes and dependencies
- [x] Ensure no functionality is lost during extraction
- [x] Update any references to these functions
- [x] Test that all extracted factories work correctly

## Status: COMPLETED ✅

**Branch:** `ai-split-paged-fused-update-cache-program-factory`
**Commit:** 516bbef841
**Build Status:** ✅ Successfully builds with `./build_metal.sh --debug`

### Changes Made:
- Created `paged_tiled_fused_update_cache_multi_core_program_factory.hpp` and `.cpp`
- Created `paged_row_major_fused_update_cache_multi_core_program_factory.hpp` and `.cpp`
- Updated original file to include the new extracted files
- Updated CMakeLists.txt to compile new source files
- Fixed duplicate function definition issues using extern declarations

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
