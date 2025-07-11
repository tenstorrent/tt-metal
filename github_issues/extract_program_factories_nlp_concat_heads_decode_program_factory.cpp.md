# Extract Program Factories from nlp_concat_heads_decode_program_factory.cpp

## Overview
This file contains 2 program factory functions that need to be extracted into separate files as part of the ProgramDescriptor refactoring effort.

**File:** `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_program_factory.cpp`

## Program Factories to Extract

### 1. multi_core_nlp_concat_heads_decode
- **Function:** `multi_core_nlp_concat_heads_decode`
- **Line:** 16
- **Status:** DONE

### 2. multi_core_nlp_concat_heads_decode_subcoregrids
- **Function:** `multi_core_nlp_concat_heads_decode_subcoregrids`
- **Line:** 153
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

**Branch:** `ai-split-nlp-concat-heads-decode-program-factory`
**Commit:** `1c7bca060a`
**Build Status:** ✅ PASSED (`./build_metal.sh --debug`)
**PR Link:** https://github.com/tenstorrent/tt-metal/pull/new/ai-split-nlp-concat-heads-decode-program-factory

**Files Created:**
- `multi_core_nlp_concat_heads_decode_program_factory.hpp`
- `multi_core_nlp_concat_heads_decode_program_factory.cpp`
- `multi_core_nlp_concat_heads_decode_subcoregrids_program_factory.hpp`
- `multi_core_nlp_concat_heads_decode_subcoregrids_program_factory.cpp`

**Files Modified:**
- `nlp_concat_heads_decode_program_factory.cpp` (functions extracted)
- `CMakeLists.txt` (added new source files)

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
