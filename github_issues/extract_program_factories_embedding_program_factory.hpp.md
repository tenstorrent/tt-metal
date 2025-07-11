# Extract Program Factories from embedding_program_factory.hpp

## Overview
This file contains 3 program factory functions that need to be extracted into separate files as part of the ProgramDescriptor refactoring effort.

**File:** `ttnn/cpp/ttnn/operations/embedding/device/embedding_program_factory.hpp`

## Program Factories to Extract

### 1. embeddings_fused
- **Function:** `embeddings_fused`
- **Line:** 77
- **Status:** TODO

### 2. embeddings_rm
- **Function:** `embeddings_rm`
- **Line:** 375
- **Status:** TODO

### 3. embeddings_tilized_indices
- **Function:** `embeddings_tilized_indices`
- **Line:** 635
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
