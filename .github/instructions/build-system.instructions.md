---
description: 'PR review for CMake build system — dependency management, sources.cmake correctness, cross-target compatibility, and include export hygiene'
applyTo: '**/CMakeLists.txt,cmake/**,**/sources.cmake,build_metal.sh,CMakePresets.json'
excludeAgent: "cloud-agent"
---

# Build System Review

## 🔴 CRITICAL

- **New external dependency**: adding a dependency via `find_package()`, `CPMAddPackage()`, `FetchContent_Declare`, or a new git submodule in `third_party/` requires explicit infra team review. Flag unconditionally — each mechanism has different packaging, caching, and reproducibility implications.
- **`sources.cmake` changes** (`tt_metal/sources.cmake`, `tt_metal/impl/sources.cmake`): these are high-risk. A duplicate symbol or missing file silently produces linker errors or ODR violations. Verify every added file is not already listed elsewhere.
- **`INTERFACE_LINK_LIBRARIES` leakage**: a library target that publicly links a heavyweight dependency forces that dependency on every downstream consumer. Prefer `PRIVATE` linking unless the dependency appears in public headers.
- **Toolchain breakage**: any change must not break the Blackhole, Wormhole, or Quasar cross-compilation targets. Flag changes that use host-specific paths or flags without appropriate guards.

## 🟡 IMPORTANT

- **`CMakePresets.json` vs. cache variables**: prefer setting options via `CMakePresets.json` over direct cache variable flags in CI scripts.
- **Transitive include pollution**: check that `target_include_directories` uses `PRIVATE` or `INTERFACE` correctly. Public include directories from implementation targets leak into all consumers.
- **CMake policy changes**: if a change sets `cmake_policy(SET ...)`, note the policy number and whether it affects downstream consumers.

## 🟢 SUGGESTION

- Use `target_sources()` with `PRIVATE` rather than appending to a list variable when adding files to a target.
- **Prefer generator expressions** (`$<...>`) over `if(CMAKE_BUILD_TYPE ...)` for config-specific flags, include paths, or link libraries. `if()` checks only evaluate at configure time and are incorrect with multi-config generators (Xcode, Ninja Multi-Config). Generator expressions evaluate at build time and are correct in all configurations. Flag `if(CMAKE_BUILD_TYPE STREQUAL ...)` guarding `target_compile_options` or `target_link_libraries` in targets that are expected to support multi-config generators — these should use `$<$<CONFIG:Release>:...>` instead. Single-config builds (plain Ninja, Makefiles) can use `if(CMAKE_BUILD_TYPE ...)` safely.
- **Avoid globally disabling compiler warnings** (e.g., `add_compile_options(-Wno-...)` or `set(CMAKE_CXX_FLAGS ...)`). Prefer `target_compile_options(target PRIVATE -Wno-...)` scoped to the specific target that needs it. Global suppressions hide issues in unrelated code.

## Target & Linkage Conventions

- **New library targets must define a namespaced alias** immediately after creation. The alias is what consumers use in `target_link_libraries` — it works identically whether the project is consumed via `add_subdirectory()`, `FetchContent`, or `find_package()`. Without the alias, each consumption path resolves the target differently.

```cmake
add_library(ttnn_op_reduction ${LIB_TYPE})
add_library(TTNN::Ops::Reduction ALIAS ttnn_op_reduction)
```

- **Always use namespaced targets** (`Foo::Bar`) when linking dependencies — raw library names like `libfoo.so` bypass CMake's dependency tracking and can silently link the wrong artifact.
- **Propagate includes via target linkage** — never use `${PROJECT_SOURCE_DIR}` or hardcoded paths in `target_include_directories`. Link the library target and let its `INTERFACE_INCLUDE_DIRECTORIES` propagate correctly.
- **Guard optional targets**: wrap references to optional or conditionally-built targets with `if(TARGET Foo::Bar)` so the build degrades gracefully in configurations that omit them.
- **Use `PROJECT_BINARY_DIR` not `CMAKE_BINARY_DIR`** — `CMAKE_BINARY_DIR` is the top-level build directory and breaks when tt-metal is consumed via `add_subdirectory()`. Always use `PROJECT_BINARY_DIR` or generator expressions for generated file paths.

## Source List vs. Build Infrastructure

File additions/removals are separated from build architecture:
- **`sources.cmake`** — adding or removing source files goes here.
- **`CMakeLists.txt`** — contains target definitions, link dependencies, compile options, and install rules.

Flag any PR that mixes both concerns. A developer adding a new `.cpp` file should only need to touch `sources.cmake`, not the `CMakeLists.txt` build structure.

## Firmware Build Boundary

`tt_metal/hw/CMakeLists.txt` and its subdirectories control **firmware** compilation for device cores (BRISC, NCRISC, ERISC, etc.). These targets use a bare-metal cross-compiler and cannot use host-side libraries, flags, or include paths. Flag any change that:
- Adds host-only flags (e.g., `-fsanitize=`, `-fPIC`) to firmware targets
- Links firmware targets against host libraries
- Introduces `#include` of host-only headers into firmware build paths

## Precompiled Headers (PCH)

New executable or library targets under `tt_metal/` or `tests/` should reuse the shared PCH when available:

```cmake
if(TARGET TT::CommonPCH)
    target_precompile_headers(my_target REUSE_FROM TT::CommonPCH)
endif()
```

Flag targets that compile heavy headers (fmt, nlohmann/json, spdlog, STL containers) without PCH reuse — each instance adds seconds to clean builds.

## Unity Builds

New library targets should call `TT_ENABLE_UNITY_BUILD(target)` to opt in to unity builds (controlled by the `TT_UNITY_BUILDS` option). Unity builds combine multiple `.cpp` files into a single translation unit for faster compilation, but introduce constraints:

- **Anonymous namespaces and `static` symbols collide** when combined. If a source file uses `static` or anonymous-namespace symbols that conflict with another file in the same target, mark it with `SKIP_UNITY_BUILD_INCLUSION`:

```cmake
set_source_files_properties(
    problematic_file.cpp
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION TRUE
)
```

- **Generated files** (e.g., protobuf) often have conflicting static symbols and should be excluded from unity builds.
- **Extremely expensive files** that bottleneck the unity batch should also be excluded — a single slow file in a unity TU blocks the entire batch from completing, leaving cores idle.

Flag new targets under `ttnn/` or `tt_metal/` that omit `TT_ENABLE_UNITY_BUILD`.

## CPM Package Naming

CPM package names must match the upstream project's `find_package()` name exactly (e.g., `GTest` not `googletest`). Mismatches break `CPM_LOCAL_PACKAGES_ONLY` mode and prevent system-installed packages from satisfying the dependency.

## Kernel Source Packaging

Device kernel sources (compiled at runtime via JIT) must be packaged into the `.deb` so they ship with the runtime. The pattern is:

1. Glob kernel files into a variable
2. Declare them as a `FILE_SET kernels` (type `HEADERS`) on the target
3. Install the file set to `${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/...` with component `ttnn-runtime`

```cmake
file(GLOB_RECURSE kernels device/kernels/*)

target_sources(my_target
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
)

install(TARGETS my_target
    FILE_SET kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/path/to/op
        COMPONENT ttnn-runtime
)
```

Flag any new op target that adds kernel source files without a `FILE_SET kernels` + `install()` rule — those kernels will be missing from packages and fail at runtime on customer machines.

## `project_options.cmake` and `build_metal.sh` Synchronization

`cmake/project_options.cmake` is the single source of truth for all build-time `option()` declarations and their defaults. `build_metal.sh` is the primary developer and CI entry point that translates CLI flags into those CMake cache variables (e.g., `--disable-unity-builds` → `-DTT_UNITY_BUILDS=OFF`). These three surfaces — CMake options, script flags, and script defaults — must stay in sync:

- **New CMake option**: if you add a new `option()` in `project_options.cmake` that developers should control, it needs a corresponding `--flag` in `build_metal.sh` with argument parsing, help text, and the appropriate `-D` passthrough.
- **Renamed or removed CMake variable**: if a variable in `project_options.cmake` is renamed or removed, update `build_metal.sh` to match — stale `-D` flags silently become no-ops and confuse users.
- **Default value drift**: the defaults in `build_metal.sh` (e.g., `unity_builds="ON"`, `pch="ON"`) must match the `option()` defaults in `project_options.cmake`. Flag any PR that changes one without updating the other.
- **Conditional overrides**: `project_options.cmake` contains conditional logic that disables options in certain contexts (e.g., unity builds disabled when `CMAKE_EXPORT_COMPILE_COMMANDS=ON` or during clang-tidy). New options with similar interactions should follow this pattern.

## Review Checklist

- [ ] No new dependencies (`find_package`, `CPMAddPackage`, `FetchContent`, submodule) without infra review
- [ ] `sources.cmake` additions checked for duplicates
- [ ] New target links use `PRIVATE` unless header exposure requires `PUBLIC`
- [ ] Cross-toolchain compatibility preserved (WH, BH, QSR)
- [ ] Namespaced targets used for all `target_link_libraries` calls
- [ ] No `CMAKE_BINARY_DIR` — use `PROJECT_BINARY_DIR` instead
- [ ] No host-only flags or libraries in firmware targets (`tt_metal/hw/`)
- [ ] New targets reuse PCH where applicable (`TT::CommonPCH`)
- [ ] New library targets call `TT_ENABLE_UNITY_BUILD(target)`
- [ ] New op targets with kernel sources define `FILE_SET kernels` + `install()` rule
- [ ] CPM package names match upstream `find_package()` names
- [ ] New `option()` in `project_options.cmake` has a matching flag in `build_metal.sh`
