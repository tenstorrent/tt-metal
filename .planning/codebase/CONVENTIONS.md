# Coding Conventions

**Analysis Date:** 2026-03-16

## Naming Patterns

**Files:**
- C++: `.cpp` and `.hpp` extensions
- Python: `.py` extension
- Test files: `*_test.cpp`, `test_*.py` suffix
- Headers: `#pragma once` (enforced by modernize-use-pragma-once check)
- Kernel files: end with `_cb_test.cpp`, `_test.cpp` for dataflow kernels

**Functions:**
- C++: snake_case (e.g., `read_and_push_to_cb`, `get_core_coord_from_relative`, `merge_num_sticks_to_read`)
- Python: snake_case (e.g., `get_test_metadata`, `metadata_field_to_column_name`, `load_test_information`)
- Inline functions with `__attribute__((always_inline))` for performance-critical dataflow code

**Variables:**
- C++: snake_case for local and static variables (e.g., `dram_buffer_src_addr`, `num_cores_x`, `start_core`)
- C++: UPPERCASE for macro constants and compile-time arguments (e.g., `MAX_HOPS`, `get_compile_time_arg_val`)
- Python: snake_case for local variables and module-level constants
- Static singletons use lowercase with underscore suffix: `is_enabled_`, `exec_mutex`

**Types:**
- C++: PascalCase for structs and classes (e.g., `RelativeCoreCoord`, `CoreRange`, `CoreCoord`, `PacketInfo`)
- C++: Specialized templates use lowercase with underscores (e.g., `to_json_t`, `from_json_t`)
- Python: PascalCase for classes (e.g., `TestMetadataLoader`, `BinaryMulTest`, `OpTestBase`)
- Python: snake_case for data classes and named tuples in lower-level utilities

**Template Parameters:**
- Single letter or descriptive: `F`, `Args...` for generic templates
- Policy parameters in all caps: `Language`, `Regex`, `Priority`

**Macros:**
- All caps with underscores: `TT_FATAL`, `TT_ASSERT`
- Project-specific macros use `TT_` prefix
- Unsafe macros documented (e.g., unsafe variable shadowing in callback closures)

## Code Style

**Formatting:**
- Tool: clang-format with `.clang-format` configuration
- Line length: 120 characters (ColumnLimit)
- Indentation: 4 spaces, no tabs (TabWidth: 4, UseTab: Never)
- Brace style: Attach braces (BreakBeforeBraces: Attach), no brace wrapping for classes/functions
- Namespace indentation: None (namespaces not indented)
- Pointer alignment: Left (PointerAlignment: Left)

**Linting:**
- Tool: clang-tidy with `.clang-tidy` configuration
- Enabled check categories: bugprone-*, cppcoreguidelines-*, modernize-*, readability-*, misc-*
- Intentionally disabled: sizeof-expression, unchecked-optional-access, narrow-cast warnings
- Cognitive complexity threshold: 312 (readability-function-cognitive-complexity)
- AllowedTypes for copy initialization: MemoryPin (exempt from copy checks)

**Key Formatting Rules:**
- Always break after open bracket: AlwaysBreak (function parameters, template arguments)
- Break before multiline strings: true
- Binary operators: no break before (BreakBeforeBinaryOperators: None)
- Template declarations: always break (AlwaysBreakTemplateDeclarations: Yes)
- Spacing in containers: true (SpacesInContainerLiterals)
- No spacing in template brackets: SpacesInAngles: false

**Files requiring exemption from formatting:**
- Extensive CCL operations and data structures (see `.clang-format-ignore`)
- Device operations with complex type hierarchies
- Generated code from codegen tools
- Files marked "Suspicious Formatting Could Cause Issues" in ignore list

## Import Organization

**Order (C++):**
1. System headers with brackets (`<ext/...h>`, `<algorithm>`, `<string>`)
2. Project headers in quotes (`"common/executor.hpp"`)
3. Other includes
4. Standard library at end

**Include Style:**
- `#include <tt_stl/assert.hpp>` for assertions
- `#include <tt-metalium/...>` for device/memory APIs
- `#include "tracy/Tracy.hpp"` for profiling
- SortIncludes: false (manual ordering preferred)

**Path Aliases (Python):**
- Direct imports from package root: `import ttnn`, `from ttnn import ...`
- Relative imports in test suites: `from tests.scripts.common import ...`
- Device fixture imports: `from tests.didt.op_test_base import OpTestBase, OpParameter`

## Error Handling

**C++ Patterns:**
- Macro-based assertions: `TT_FATAL(condition, "message", format_args...)`
- Macro-based validation: `TT_ASSERT(condition, "message")`
- Exceptions for type/format errors: `throw std::runtime_error`, `throw std::invalid_argument`
- Task execution wrapping with exception propagation via `packaged_task<return_type()>`
- Worker thread exception capture and rethrow in thread pools (see `thread_pool.cpp`)

**Python Patterns:**
- Try-catch for subprocess operations with custom error logging
- pytest markers for marking test failures (skip_slow, skip_unmarked, requires_fast_runtime_mode_off)
- Logger usage for non-fatal errors (loguru.logger)
- Assertions for preconditions in test classes (e.g., `assert model_path`, `assert isinstance(timeout_in_s, int)`)

**Error Messages:**
- Use fmt-style formatting: `"Start core must be within grid size"` with variadic arguments
- Include context: show coordinates, dimensions, indices in error messages
- For fatal errors, include actual vs. expected values

## Logging

**Framework:**
- C++: No standard logging framework; uses `std::cout` and macros in select locations
- Python: loguru logger (imported as `from loguru import logger`)

**Patterns:**
- Python: `logger.info()`, `logger.warning()`, `logger.error()`, `logger.debug()`
- Profiling logs use Tracy: `#include "tracy/Tracy.hpp"` with PROFILER_LOGS_DIR
- Device-side logging via profiler (TT_METAL_DEVICE_PROFILER=1 environment variable)

**When to Log:**
- Test setup/teardown: logger.info() for test parameters and device configuration
- Performance metrics: logger.info() for bandwidth, latency measurements
- Configuration issues: logger.warning() for deprecated or fallback behaviors
- Errors and exceptions: logger.error() with context

## Comments

**When to Comment:**
- Complex algorithmic logic: explain the approach before the code block
- Temporary workarounds: mark with `TODO:` or `FIXME:` comment with explanation
- Non-obvious design choices: "This avoids iterator invalidation when modifying map nodes"
- Device-specific behaviors: note hardware requirements or limitations

**JSDoc/TSDoc:**
- C++: No JSDoc standard; struct members have no doc comments
- Python: Classes include docstrings describing behavior (e.g., TestMetadataLoader)
- Method docstrings: describe parameters (Args:), return value (Returns:), and raised exceptions (Raises:)

**Comment Style:**
- C++ single-line: `//` for brief notes
- C++ multi-line: `/* */` rarely used; prefer multiple `//` lines
- SPDX headers: `// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.` and `// SPDX-License-Identifier: Apache-2.0` at top of every file

## Function Design

**Size:**
- Prefer short functions (no enforced limit; cognitive complexity threshold of 312 is high)
- Larger functions acceptable for kernel code and performance-critical paths
- Extract inline functions with `__attribute__((always_inline))` for dataflow kernels

**Parameters:**
- Pass by const reference: `const CoreCoord&`
- Output parameters by non-const reference: `CoreRangeSet&` result
- Move semantics for heavy objects
- No default arguments preferred (google-default-arguments disabled in tidy)

**Return Values:**
- Return by value for small types (uint32_t, bool)
- Return by reference for complex types (`CoreRange&`)
- Use `std::optional<T>` for optional returns (e.g., `std::optional<CoreRange> intersection()`)
- Tuple returns for multiple values: `std::tuple<uint32_t, uint32_t>` for (num_cores, per_core_tiles)

## Module Design

**Exports:**
- Header-only utilities: define inline functions and templates in `.hpp`
- Implementation separation: `.cpp` files for non-inline definitions
- Namespace wrapping: `namespace tt::tt_metal { ... }`
- Inline functions in headers: GetExecutor(), GetExecutorMutex() for singletons

**Barrel Files:**
- Not consistently used
- Headers typically include what they export without re-exporting from other headers

**Initialization:**
- Static initialization in functions: `static Executor* exec = [](){ ... }();`
- Thread-safe singleton pattern with `std::atexit` cleanup
- Late binding via `GetExecutor()` rather than static global instantiation

---

*Convention analysis: 2026-03-16*
