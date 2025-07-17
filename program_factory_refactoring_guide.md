# Program Factory Refactoring Guide

## Overview
This guide documents the systematic approach to refactor program factory files that contain multiple Program-creating functions into separate files with one function each.

## Step-by-Step Process

### 1. Analysis Phase
- Identify files with multiple Program-creating functions
- Read the original file to understand the structure
- Note function names, parameters, and any helper functions
- Check associated .hpp files for declarations

### 2. File Creation Phase
For each Program-creating function, create:
- `<function_name>_program_factory.hpp` - Header with function declaration
- `<function_name>_program_factory.cpp` - Implementation file

**File naming pattern:**
- Extract function name (e.g., `topk_single_core_interleaved`)
- Convert to file name (e.g., `topk_single_core_program_factory`)

### 3. Code Extraction
- Copy function implementation to new .cpp file
- Include necessary headers and dependencies
- Copy any helper functions used exclusively by that function
- Ensure proper namespace usage

### 4. Header Updates
- Create function declaration in new .hpp file
- Update main program_factory.hpp to include new headers
- Remove original function declarations from main header

### 5. Main File Updates
- Remove extracted functions from main .cpp file
- Keep shared helper functions if used by multiple functions
- Update main .cpp to just include the main header

### 6. Build System Updates
- Add new .cpp files to CMakeLists.txt in the target_sources section
- Follow existing patterns in the file

### 7. Validation
- Compile with `./build_metal.sh -c --debug`
- Ensure no compilation errors
- Verify all pre-commit hooks pass

### 8. Version Control Workflow
```bash
# Create branch
git checkout -b refactor/<operation>-program-factory-split

# Add files
git add <all_modified_and_new_files>

# Commit
git commit -m "refactor: split <operation> program factory into separate files"

# Push
git push -u origin refactor/<operation>-program-factory-split

# Create PR
gh pr create --title "refactor: split <operation> program factory into separate files" --body "..."

# Run workflow
gh workflow run "All post-commit tests" --ref refactor/<operation>-program-factory-split

# Add workflow link to PR
gh run list --workflow="All post-commit tests" --limit=1  # Get run ID
gh pr comment <PR_NUMBER> --body "🚀 All post-commit tests workflow: <workflow_url>"
```

## File Structure Template

### New Header File (.hpp)
```cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"

namespace ttnn::operations::<namespace>::detail {

tt::tt_metal::operation::ProgramWithCallbacks <function_name>(
    // function parameters
);

}  // namespace ttnn::operations::<namespace>::detail
```

### New Implementation File (.cpp)
```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include "ttnn/operation.hpp"
// other necessary includes

using namespace tt::tt_metal;
using namespace std;
namespace ttnn::operations::<namespace>::detail {

// Helper functions (if any)

// Main function implementation
tt::tt_metal::operation::ProgramWithCallbacks <function_name>(
    // parameters
) {
    // implementation
}

}  // namespace ttnn::operations::<namespace>::detail
```

### Updated Main Header
```cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "<function1>_program_factory.hpp"
#include "<function2>_program_factory.hpp"
// etc.
```

## Key Considerations

1. **Backward Compatibility**: Maintain all existing API through header includes
2. **Helper Functions**: Only move helper functions if they're used by single function
3. **Namespace Consistency**: Maintain original namespace structure
4. **Build Dependencies**: Ensure all necessary dependencies are included
5. **Function Declarations**: Avoid duplication - declare only once per function

## Common Patterns

### File Locations
- Usually in: `ttnn/cpp/ttnn/operations/<domain>/<operation>/device/`
- CMakeLists.txt location: `ttnn/cpp/ttnn/operations/<domain>/CMakeLists.txt`

### Branch Naming
- Pattern: `refactor/<operation>-program-factory-split`
- Examples: `refactor/topk-program-factory-split`, `refactor/argmax-program-factory-split`

### PR Template
```markdown
## Summary
- Split `<operation>_program_factory.cpp` into separate files to improve code organization
- Extract `<function1>` into `<function1>_program_factory.cpp/hpp`
- Extract `<function2>` into `<function2>_program_factory.cpp/hpp`
- Update CMakeLists.txt to include new source files
- Maintain backward compatibility through updated header includes

## Motivation
This refactoring reduces the number of Program-creating functions per file from X to 1, making the codebase more modular and easier to maintain while preserving all existing functionality and API compatibility.

## Test plan
- [x] Compilation test with `./build_metal.sh -c --debug` passed
- [x] All pre-commit hooks passed
- [x] No functional changes - maintains full backward compatibility
```

## Completed Refactoring (✅)

### **Round 1 - Initial 8 Files**:
1. **`topk_program_factory.cpp`** - 2 functions → PR #25231
2. **`argmax_program_factory.cpp`** - 2 functions → PR #25232
3. **`padded_slice_program_factory.cpp`** - 2 functions → PR #25234
4. **`reshard_program_factory.cpp`** - 3 functions → PR #25235
5. **`sort_program_factory.cpp`** - 3 functions → PR #25236
6. **`tilize_with_val_padding_program_factory.cpp`** - 4 functions → PR #25237
7. **`update_cache_op_multi_core.cpp`** - 2 functions → PR #25238
8. **`fill_cache_multi_core.cpp`** - 2 functions → PR #25238

**Total Completed**: 8 files, 20+ functions

## Common Issues & Solutions

### 1. Pre-commit Hook Failures
- **Issue**: Missing newlines at end of files
- **Solution**: Let pre-commit hooks fix automatically, then commit again

### 2. Compilation Errors
- **Issue**: Missing includes or namespace issues
- **Solution**: Add proper includes and use full namespace qualifiers

### 3. Function Redefinition
- **Issue**: Functions declared in multiple headers
- **Solution**: Remove declarations from main header, keep only in separate headers

### 4. Complex Helper Functions
- **Issue**: Helper functions used by multiple Program-creating functions
- **Solution**: Keep shared helpers in main file, move function-specific helpers to separate files

### 5. CMakeLists.txt Updates
- **Issue**: Forgetting to add new source files
- **Solution**: Always add new .cpp files to target_sources in CMakeLists.txt

## Best Practices

1. **Always compile test** after each refactoring
2. **Follow existing naming conventions** in the codebase
3. **Maintain backward compatibility** through header includes
4. **Use specific file staging** (`git add <specific_files>`) instead of `git add -A`
5. **Test PRs with full workflow** before merging
6. **Document function counts** in PR descriptions for clarity

## Next Steps

See `program_factory_refactoring_remaining_tasks.md` for the complete list of remaining files to refactor.

---

**Last Updated**: December 2024
**Guide Version**: 2.0
**Status**: Active refactoring in progress
