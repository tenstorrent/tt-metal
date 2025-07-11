# Extract Program Factories from split_query_key_value_and_split_heads_program_factory.hpp

## Overview
This file contains 2 program factory functions that need to be extracted into separate files as part of the ProgramDescriptor refactoring effort.

**File:** `ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/split_query_key_value_and_split_heads_program_factory.hpp`

## Program Factories to Extract

### 1. multi_core_split_query_key_value_and_split_heads
- **Function:** `multi_core_split_query_key_value_and_split_heads`
- **Line:** 16
- **Status:** TODO

### 2. multi_core_split_query_key_value_and_split_heads_sharded
- **Function:** `multi_core_split_query_key_value_and_split_heads_sharded`
- **Line:** 211
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
