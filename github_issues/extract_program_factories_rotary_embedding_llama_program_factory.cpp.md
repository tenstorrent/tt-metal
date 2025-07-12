# Extract Program Factories from rotary_embedding_llama_program_factory.cpp

## Overview
This file contains 2 program factory functions that need to be extracted into separate files as part of the ProgramDescriptor refactoring effort.

**File:** `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/rotary_embedding_llama_program_factory.cpp`

## Program Factories to Extract

### 1. rotary_embedding_llama_multi_core
- **Function:** `rotary_embedding_llama_multi_core`
- **Line:** 17
- **Status:** TODO

### 2. rotary_embedding_llama_multi_core_sharded
- **Function:** `rotary_embedding_llama_multi_core_sharded`
- **Line:** 335
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
