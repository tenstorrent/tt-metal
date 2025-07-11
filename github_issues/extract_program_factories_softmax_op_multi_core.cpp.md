# Extract Program Factories from softmax_op_multi_core.cpp

## Overview
This file contains 2 program factory functions that need to be extracted into separate files as part of the ProgramDescriptor refactoring effort.

**File:** `ttnn/cpp/ttnn/operations/normalization/softmax/device/multi_core/softmax_op_multi_core.cpp`

## Program Factories to Extract

### 1. scale_mask_softmax_multi_core
- **Function:** `scale_mask_softmax_multi_core`
- **Line:** 34
- **Status:** TODO

### 2. scale_mask_softmax_sharded_multi_core
- **Function:** `scale_mask_softmax_sharded_multi_core`
- **Line:** 591
- **Status:** TODO

## Tasks

- [ ] Analyze dependencies and shared code between the 2 program factories
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
