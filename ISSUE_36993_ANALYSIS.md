# Fix #36993: Restore compiler selection logic for tt-train standalone builds

## Problem Summary

Commit `a7618b9282` ("TT-Train: bump clang version from 17 to 20 #36568") **deleted** all compiler selection logic from tt-train:
- Deleted `tt-train/cmake/compilers.cmake` (63 lines: `FIND_AND_SET_CLANG20()`, `CHECK_COMPILERS()`, `ADJUST_COMPILER_WARNINGS()`)
- Removed 15 lines from `tt-train/CMakeLists.txt` (env var override, clang-20 default, compiler validation)
- Replaced with: CI workflow `env: CC/CXX` variables + README documentation

**Result:** `pip install -e tt-train/` (standalone via scikit-build-core) uses system default GCC 11, which fails on `reflect` library's `constexpr` lambda function pointers (GCC 12+ required).

**When tt-train is built as a tt-metal subproject** (via `add_subdirectory(tt-train)` in tt-metal's `CMakeLists.txt:309`), it inherits the parent's toolchain (clang-20) and works fine.

## Why clang-20 is mandatory (not just preferred)

tt-train links directly against tt-metal's compiled C++ shared libraries:
- `ttml` → `TT::Metalium` (libtt_metal.so) + `TTNN::TTNN` (_ttnncpp.so) — `tt-train/sources/ttml/CMakeLists.txt:447-455`
- `_ttml` (Python bindings) → same libraries — `tt-train/sources/ttml/CMakeLists.txt:576-584`

These libraries are compiled with **clang-20 + libstdc++** (via `cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake`). Building tt-train with a different compiler risks:

1. **ABI incompatibilities** — different name mangling, vtable layout, exception handling
2. **C++20 feature gaps** — GCC 11 doesn't support the same C++20 features as clang-20
3. **Subtle runtime bugs** — even if compilation succeeded, linking clang-20-compiled `.so` files into GCC-compiled binaries can produce hard-to-debug crashes
4. **The `reflect` library failure** — the immediate symptom (GCC 11 constexpr lambda bug), but just the tip of the iceberg

**Conclusion:** The fix should require clang-20 or fail clearly. GCC support is not meaningful when the linked libraries are clang-20-compiled.

## Current State

| Component | Compiler selection | Default | Works? |
|-----------|-------------------|---------|--------|
| tt-metal top-level | `build_metal.sh` → toolchain file `cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake` | clang-20 | Yes |
| tt-train as subproject | Inherits parent toolchain | clang-20 | Yes |
| tt-train standalone (`pip install -e`) | **None** — CMake default detection | System GCC (11) | **No** |

## Root Cause Analysis

### The Destructive Commit

**Commit:** `a7618b9282` — "TT-Train: bump clang version from 17 to 20 (#36568)"

**What was removed:**

1. **`tt-train/cmake/compilers.cmake`** (entire file deleted, 63 lines):
   - `FIND_AND_SET_CLANG20()` — searches PATH for clang-20, sets as default compiler
   - `CHECK_COMPILERS()` — validates Clang 17+ recommended, GCC >= 12 required, rejects unsupported compilers
   - `ADJUST_COMPILER_WARNINGS()` — 20 compiler-specific warning suppression flags (10 Clang, 10 GCC)

2. **`tt-train/CMakeLists.txt`** (15 lines removed):
   ```cmake
   include(cmake/compilers.cmake)                          # ← DELETED
   if(DEFINED ENV{CMAKE_C_COMPILER} AND DEFINED ENV{CMAKE_CXX_COMPILER})
       set(CMAKE_C_COMPILER $ENV{CMAKE_C_COMPILER})
       set(CMAKE_CXX_COMPILER $ENV{CMAKE_CXX_COMPILER})
   endif()
   if(CMAKE_CXX_COMPILER AND CMAKE_C_COMPILER)
       ...
   else()
       find_and_set_clang20()                              # ← DELETED
   endif()
   CHECK_COMPILERS()                                       # ← DELETED
   ```

**What was added instead:**
- CI workflow YAML files with `env: CC=clang-20 CXX=clang++-20`
- README documentation telling users to set CC/CXX env vars manually

**Verdict:** This was an **intentional simplification** that shifted compiler selection from CMake logic to environment variables. The author assumed users would always set `CC`/`CXX`. But `pip install -e tt-train/` via scikit-build-core creates an isolated CMake invocation with no env vars set.

### Timeline

- `5eac9853ea` (Jan 25) — Properly refactored clang-17 → clang-20, keeping all infrastructure
- `a7618b9282` (Jan 28) — Deleted all infrastructure, 3 days later (appears to be a duplicate effort that chose deletion)

## Key Files

- `tt-train/CMakeLists.txt` — missing compiler logic (lines 1-8)
- `tt-train/pyproject.toml` — scikit-build-core config, no cmake args (lines 10-18)
- `tt-train/cmake/compilers.cmake` — **deleted**, needs restoration
- `cmake/compilers.cmake` — tt-metal's version (simpler, 22 lines, only `CHECK_COMPILERS()`)
- `cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake` — tt-metal's toolchain file

## Implementation Alternatives

### Alternative A: Restore clang-20-only `compilers.cmake` (recommended)

Restore `tt-train/cmake/compilers.cmake` with a **simplified, clang-20-only** version (no GCC fallback), and re-add the compiler selection logic to `tt-train/CMakeLists.txt`.

Since tt-train links against clang-20-compiled tt-metal libraries, GCC support is not meaningful. The restored file is simpler than the original:

**Changes:**
1. Recreate `tt-train/cmake/compilers.cmake` with:
   - `FIND_AND_SET_CLANG20()` — find clang-20 or fail with a clear error
   - `CHECK_COMPILERS()` — validate clang-20 is actually being used; reject GCC with a clear message explaining ABI incompatibility
   - `ADJUST_COMPILER_WARNINGS()` — Clang-specific warning flags only

2. Add back to `tt-train/CMakeLists.txt` (before `project()`):
   ```cmake
   include(cmake/compilers.cmake)

   # Allow CI/user override via environment variables
   if(DEFINED ENV{CMAKE_C_COMPILER} AND DEFINED ENV{CMAKE_CXX_COMPILER})
       set(CMAKE_C_COMPILER $ENV{CMAKE_C_COMPILER})
       set(CMAKE_CXX_COMPILER $ENV{CMAKE_CXX_COMPILER})
   elseif(NOT CMAKE_CXX_COMPILER)
       find_and_set_clang20()
   endif()

   project(ml-framework-cpp LANGUAGES C CXX)
   CHECK_COMPILERS()
   ```

**Pros:**
- Restores proven auto-detection behavior
- Standalone builds find clang-20 automatically
- Clear error message if clang-20 is not installed
- Rejects GCC with an explanation (ABI mismatch with tt-metal)
- Simpler than the original (no GCC warning flags, no GCC version checks)
- CI `env: CC/CXX` override still works

**Cons:**
- tt-train maintains its own `compilers.cmake` separate from tt-metal's `cmake/compilers.cmake`
- Blocks users who intentionally want to build everything with GCC-12 (edge case — they'd need to rebuild tt-metal with GCC-12 too)

### Alternative B: Use tt-metal's toolchain file from tt-train

Make tt-train reference tt-metal's existing `cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake` via relative path.

**Changes:**
1. Add to `tt-train/CMakeLists.txt` (before `project()`):
   ```cmake
   if(NOT CMAKE_TOOLCHAIN_FILE AND NOT CMAKE_CXX_COMPILER)
       set(_TT_METAL_TOOLCHAIN "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake")
       if(EXISTS "${_TT_METAL_TOOLCHAIN}")
           set(CMAKE_TOOLCHAIN_FILE "${_TT_METAL_TOOLCHAIN}" CACHE FILEPATH "")
       else()
           message(FATAL_ERROR "tt-metal toolchain not found at ${_TT_METAL_TOOLCHAIN}. "
                   "tt-train must be built within the tt-metal source tree.")
       endif()
   endif()
   ```

**Pros:**
- No file duplication — reuses tt-metal's toolchain directly
- Stays in sync automatically (if tt-metal bumps to clang-21, tt-train follows)
- Also gets the mold/lld linker setup from the toolchain file

**Cons:**
- Assumes tt-train lives inside tt-metal's source tree (`../cmake/` path)
- Breaks if tt-train is ever extracted to a standalone repo
- `CMAKE_TOOLCHAIN_FILE` must be set before `project()` — may not work reliably with scikit-build-core
- No `ADJUST_COMPILER_WARNINGS()` (tt-metal's toolchain doesn't have it)
- No clear error message when non-clang compiler is used

### Alternative C: Add scikit-build-core cmake.args to pyproject.toml

Configure scikit-build-core to pass the toolchain file to CMake automatically.

**Changes:**
1. In `tt-train/pyproject.toml`, add:
   ```toml
   [tool.scikit-build.cmake.define]
   CMAKE_TOOLCHAIN_FILE = {env = "CMAKE_TOOLCHAIN_FILE", default = "../cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake"}
   ```

**Pros:**
- Minimal change (1-2 lines in pyproject.toml)
- Directly addresses the `pip install -e` path
- Allows override via `CMAKE_TOOLCHAIN_FILE` env var

**Cons:**
- Only fixes the `pip install -e` path, not standalone `cmake -B build` path
- Relative path `../cmake/` assumes tt-metal source tree layout
- No compiler validation or clear error messages
- scikit-build-core `{env=..., default=...}` syntax may not be supported in all versions

## Recommendation

**Alternative A** (clang-20-only `compilers.cmake`) is the recommended approach:
- Self-contained — works whether tt-train is built standalone or as subproject
- Clear error messages — tells the user exactly what's wrong and how to fix it
- Simpler than the original — no GCC fallback paths needed
- Compatible with CI's `env: CC/CXX` approach

## Verification Plan

1. **Container build:** Use `~/tt/tt-container/cached/` with `verify/issue-36993-gcc-constexpr-lambda` branch
   - Build container from the fix branch
   - Verify `pip install -e tt-train/` succeeds with clang-20 auto-detection
   - Verify `uv pip install -e tt-train` also works (uses venv uv from #37007 fix)
2. **Local test:** On the fix branch in tt-metal, run standalone tt-train build:
   ```bash
   cd tt-train && cmake -B build -GNinja && cmake --build build
   ```
3. **Verify GCC rejection:** Test that GCC is properly rejected:
   ```bash
   CC=gcc CXX=g++ cmake -B build_gcc -GNinja  # Should fail with clear ABI mismatch message
   ```

## Related Issues and PRs

- **#36993:** This issue — standalone build fails with GCC
- **#37007:** uv not found after venv activation (prerequisite, fixed in PR #37160)
- **PR #37160:** The uv symlink fix (merged or pending)
- **#36568:** The original PR that caused the regression
