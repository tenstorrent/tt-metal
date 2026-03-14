# Coding Conventions

**Analysis Date:** 2025-03-12

## Naming Patterns

**Files:**
- Headers: `.hpp` for C++ headers, `.h` for C headers
- Source: `.cpp` for implementations
- Test files: `test_*.cpp` or `*_test.cpp` for C++, `test_*.py` for Python
- Internal implementation: Typically named after the feature (e.g., `fabric_erisc_router.cpp`, `core_coord.cpp`)

**Functions:**
- snake_case for all functions: `get_core_coord_from_relative()`, `create_kernel()`, `add_circular_buffer_()`
- Const correctness with `const` suffix for read-only operations
- Private implementation functions use trailing underscore: `add_circular_buffer_()`, `setup_program_state_()`

**Variables:**
- snake_case for local and member variables: `start_coord`, `end_coord`, `max_cbs_`, `program_configs_`
- Trailing underscore for private member variables: `kernels_`, `cb_mask_width_`, `program_configs_`, `grid_extent_`
- UPPERCASE_SNAKE_CASE for compile-time constants: `RX_CH_TRID_STARTS`, `pingpong_trid_a`
- Loop counters: `i`, `j`, `x`, `y` are acceptable for coordinate systems

**Types:**
- PascalCase for classes and structs: `RelativeCoreCoord`, `CoreRange`, `CoreCoord`, `Program`, `Buffer`, `Kernel`
- Template parameters: PascalCase: `Config`, `Processor`, `NOC`
- Type aliases: PascalCase with context: `using DistributedContext = tt::tt_metal::distributed::multihost::DistributedContext;`

**Macros and Compile-time Constants:**
- UPPERCASE_SNAKE_CASE: `FORCE_INLINE`, `ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA`, `FABRIC_TELEMETRY_BANDWIDTH`
- Conditionals: `if constexpr` for compile-time selection based on template parameters

## Code Style

**Formatting:**
- Tool: clang-format with Google-based style (`.clang-format` at repo root)
- Indentation: 4 spaces (never tabs, `UseTab: Never`)
- Column limit: 120 characters
- Brace style: Attach (opening brace on same line as statement)

**Key Settings:**
- `AlwaysBreakTemplateDeclarations: Yes` - template declarations on new line
- `BreakBeforeBinaryOperators: None` - operators stay at end of line
- `ConstructorInitializerAllOnOneLineOrOnePerLine: true` - initializer lists strictly formatted
- `AlignAfterOpenBracket: AlwaysBreak` - parameters always break after opening bracket
- `BinPackArguments: false`, `BinPackParameters: false` - one argument per line (readable for complex calls)

**Linting:**
- No explicit ESLint/linter config detected for C++ beyond clang-format
- Manual code review standards enforced via Pull Request process
- Google C++ style guide conventions observed for naming and structure

## Import Organization

**Order:**
1. Standard library headers: `<vector>`, `<string>`, `<memory>`, `<algorithm>`, etc.
2. Third-party headers: `<nlohmann/json.hpp>`, `<tt_stl/...>`, `<fmt/...>`
3. Project includes with quotes: `"core_coord.hpp"`, `"impl/buffers/circular_buffer.hpp"`
4. Implementation-specific headers: Device-specific includes at the end

**Example from `program.cpp`:**
```cpp
#include <allocator.hpp>                      // public API
#include <circular_buffer.hpp>
#include <vector>
#include <algorithm>                          // standard library
#include <tt_stl/assert.hpp>                  // tt_stl framework
#include "buffer.hpp"                         // project includes
#include "impl/buffers/circular_buffer.hpp"
#include "core_coord.hpp"
#include "common/stable_hash.hpp"
```

**Path Aliases:**
- Quoted includes use relative paths resolved by CMake include directories
- Angle brackets for public API and standard library: `#include <tt-metalium/...>`
- Project headers typically relative to source: `#include "fabric/hw/inc/edm_fabric/..."`

## Error Handling

**Patterns:**
- Use `TT_FATAL()` macro for fatal errors that should terminate:
  ```cpp
  TT_FATAL(
      end_coord.x >= start_coord.x and end_coord.y >= start_coord.y,
      "Invalid core range for start_coord: {}, end_coord: {}",
      start_coord.str(),
      end_coord.str());
  ```
- Use `TT_ASSERT()` for assertions that should fail in debug builds:
  ```cpp
  TT_ASSERT(
      cb_mask_width_ >= max_cbs_,
      "CB mask width ({}) is insufficient for architecture's {} CBs",
      cb_mask_width_,
      max_cbs_);
  ```
- Format strings support fmt-style placeholders: `{}`, `{:04x}`, etc.
- No exceptions thrown; assertion macros terminate on failure

## Logging

**Framework:** Custom tt_metal logging system via `log_*` macros

**Patterns:**
- `log_info(tt::LogTest, "message")` for informational logs
- `log_warning(tt::LogTest, "message")` for warnings
- `log_error(tt::LogTest, "message")` for errors
- Category tags: `tt::LogTest`, context-specific categories available

**Python Tests:**
- Use `loguru.logger` for Python tests: `logger.info()`, `logger.warning()`
- Logs typically printed to stdout during test execution

## Comments

**When to Comment:**
- Complex algorithms explaining intent: See `CoreRangeSet::compress()` for multi-step optimization
- Non-obvious workarounds or constraints: "By overallocating by one x entry, we can avoid boundary checks..."
- Public API methods have brief purpose descriptions
- Hardware-specific behaviors or constraints are documented inline

**JSDoc/TSDoc:**
- Minimal use; primary documentation in public headers (`.hpp` files)
- Method signatures self-document most behavior
- Complex structs documented with member comments in header files

**Example from fabric_erisc_router.cpp:**
```cpp
// Merge lined-up (in x or y dimension) intersecting/adjacent rectangles
std::optional<CoreRange> CoreRange::merge(const CoreRange& cr) const {
    if (this->intersects(cr) || this->adjacent(cr)) {
        // Aligned on x-axis - merge y coordinates
        if (this->start_coord.x == cr.start_coord.x && this->end_coord.x == cr.end_coord.x) {
            return CoreRange({this->start_coord.x, std::min(this->start_coord.y, cr.start_coord.y)}, ...);
        }
    }
}
```

## Function Design

**Size:**
- Prefer small functions (50-150 lines typical)
- Larger functions in device kernels due to hardware constraints
- Data movement kernels may contain loops and state machines in single function

**Parameters:**
- Pass complex objects by `const&` when not modified
- Pass mutable state by non-const reference: `LineSenderState& ss`
- Use struct parameters for related grouped data rather than individual parameters
- Avoid default parameters; explicit arguments clarified intent

**Return Values:**
- Return `bool` for success/failure without side effects
- Return `std::optional<T>` for nullable results: `std::optional<CoreRange>`
- Return `std::variant<>` for multiple possible return types
- Return `void` when only side effects (state mutation) occur
- Prefer multiple return values via `std::tuple` or output parameters

**Example from program.cpp:**
```cpp
// Multi-type return using std::variant
auto config = std::visit(
    tt::stl::overloaded{
        [&](const ReaderConfigDescriptor&) -> std::variant<DataMovementConfig, ComputeConfig> {
            return ReaderDataMovementConfig{...};
        },
        [&](const WriterConfigDescriptor&) -> std::variant<DataMovementConfig, ComputeConfig> {
            return WriterDataMovementConfig{...};
        },
    },
    kernel_descriptor.config);
```

## Module Design

**Exports:**
- Public API in header files (`.hpp`)
- Forward declarations in headers
- Implementation in corresponding `.cpp` files
- Private implementation details in `detail::` namespace or nested `namespace { }`

**Barrel Files:**
- Not extensively used
- Include guards via `#pragma once` standard

**Example from core_coord.hpp:**
```cpp
namespace tt::tt_metal {
    // Public struct
    struct RelativeCoreCoord {
        long x = 0;
        long y = 0;
        std::string str() const;
    };

    // Public function
    CoreCoord get_core_coord_from_relative(const RelativeCoreCoord& in, const CoreCoord& grid_size);
}

// Specializations outside main namespace
template <>
struct std::hash<tt::tt_metal::RelativeCoreCoord> { ... };
```

## Namespace Organization

**Pattern:** `tt::tt_metal::*` for core framework
- `tt::tt_metal::distributed::multihost::*` for distributed/multihost features
- `tt::tt_fabric::*` for fabric-specific code
- Implementation details in `detail::` namespace or unnamed `namespace { }`
- Nested namespaces rare; typically 2-3 levels deep

**Using Declarations:**
```cpp
using Rank = tt::tt_metal::distributed::multihost::Rank;
using DistributedContext = tt::tt_metal::distributed::multihost::DistributedContext;
```

## Inline and Optimization Hints

**FORCE_INLINE macro:**
- Used in hot paths (e.g., `FORCE_INLINE void line_speedy_send_one_packet()`)
- Applied to small utility functions called frequently
- Ensures compiler inlines despite complexity heuristics
- See `fabric_erisc_router_speedy_path.hpp` for examples

**constexpr:**
- Used for compile-time constants: `static constexpr uint8_t pingpong_trid_a = ...`
- Used in `if constexpr` for compile-time feature selection
- Example: `if constexpr (FABRIC_TELEMETRY_BANDWIDTH) { ... }`

---

*Convention analysis: 2025-03-12*
