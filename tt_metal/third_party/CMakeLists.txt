set(CMAKE_C_CLANG_TIDY "")
set(CMAKE_CXX_CLANG_TIDY "")
set(CMAKE_VERIFY_INTERFACE_HEADER_SETS FALSE)
set(TT_UMD_ENABLE_CLANG_TIDY OFF CACHE BOOL "Disable clang-tidy checks for UMD" FORCE)

if(TT_METAL_USE_EMULE)
    set(TT_UMD_BUILD_EMULE ON CACHE BOOL "Enable SWEmuleChip (forwarded from TT_METAL_USE_EMULE)" FORCE)
endif()

add_subdirectory(umd)

# Propagate the top-level ENABLE_TRACY option into Tracy's own TRACY_ENABLE option *before*
# add_subdirectory(tracy). Tracy's CMakeLists.txt reads TRACY_ENABLE in tracy_set_option()
# and, when ON, attaches `target_compile_definitions(TracyClient PUBLIC TRACY_ENABLE)`. That
# PUBLIC define is what gates the runtime check in tt_metal/llrt/rtoptions.cpp
# (`#if !defined(TRACY_ENABLE) TT_FATAL(...)` for TT_METAL_DEVICE_PROFILER) and every other
# Tracy-instrumented call site in tt-metal. Without this propagation a build configured with
# `-DENABLE_TRACY=ON` still produces binaries that abort the moment device profiling is
# requested.
# Seed as CACHE (FORCE) — Tracy's CMakeLists.txt uses `cmake_minimum_required(VERSION 3.10)`,
# which keeps CMP0077 at OLD inside that scope. Under OLD, `option(TRACY_ENABLE ... OFF)`
# wipes a normal variable of the same name and re-creates the cache entry at its default,
# silently overriding `set(TRACY_ENABLE ON)`. Setting the cache value here is honored
# by `option()` under both OLD and NEW policy.
if(ENABLE_TRACY)
    set(TRACY_ENABLE ON CACHE BOOL "Enable Tracy profiling instrumentation" FORCE)
else()
    set(TRACY_ENABLE OFF CACHE BOOL "Enable Tracy profiling instrumentation" FORCE)
endif()

# Tracy will always be built with fallback timing enabled — avoids hard failures on systems
# without invariant TSC (some VMs, older hardware) by falling back to OS clocks.
set(TRACY_TIMER_FALLBACK ON CACHE BOOL "Tracy: use lower-resolution timer fallback" FORCE)

add_subdirectory(tracy)

# When Tracy is enabled, TracyClient consumers need frame pointers preserved (so backtrace
# walking works) and `-rdynamic` on the final link line (so Tracy can resolve symbols from
# the dynamic symbol table). Both must be PUBLIC so they propagate to anything linking
# TracyClient.
if(ENABLE_TRACY AND TARGET TracyClient)
    target_compile_options(TracyClient PUBLIC -fno-omit-frame-pointer)
    target_link_options(TracyClient PUBLIC -rdynamic)

    # Rename the shared library to libtracy.so to follow the project's library naming convention
    # and install it alongside the other runtime libraries. Without an explicit install rule in a
    # component CPack already packages, dpkg-shlibdeps cannot locate libTracyClient.so when
    # resolving dependencies of binaries that link against it, which fails the .deb packaging step.
    set_target_properties(
        TracyClient
        PROPERTIES
            OUTPUT_NAME
                "tracy"
    )
    install(
        TARGETS
            TracyClient
        LIBRARY
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT metalium-runtime
        ARCHIVE
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT metalium-runtime
    )
endif()

set(NO_FILESELECTOR ON)

# Repair CMAKE_(C|CXX)_COMPILER_(AR|RANLIB) before building the Tracy CLI tools.
#
# Tracy's csvexport/capture pull zstd via CPM, which uses add_library(... STATIC ...).
# CMake static-library rules dereference the language-specific archiver
# (CMAKE_<LANG>_COMPILER_AR), and Clang on Linux often leaves it as
# `CMAKE_<LANG>_COMPILER_AR-NOTFOUND` because llvm-ar-<N> isn't installed beside
# clang-<N>. That literal then leaks into the Ninja link command and ccache fails
# trying to invoke it. Tracy's own config sets RULE_LAUNCH_LINK to ccache (see
# tt_metal/third_party/tracy/cmake/config.cmake), which makes the failure surface
# even on the archive step. Fall back to the generic CMAKE_AR/CMAKE_RANLIB (which
# CMake does discover via /usr/bin/ar), so Tracy's vendored zstd builds without
# requiring the LTO-aware llvm-ar-<N> on PATH.
if(NOT WIN32)
    if(NOT CMAKE_AR OR "${CMAKE_AR}" MATCHES "NOTFOUND")
        find_program(
            _tt_resolved_ar
            NAMES
                ar
                llvm-ar
                gcc-ar
                NO_CACHE
        )
        if(_tt_resolved_ar)
            set(CMAKE_AR "${_tt_resolved_ar}" CACHE FILEPATH "Path to archiver for static libraries" FORCE)
        endif()
        unset(_tt_resolved_ar)
    endif()
    if(NOT CMAKE_RANLIB OR "${CMAKE_RANLIB}" MATCHES "NOTFOUND")
        find_program(
            _tt_resolved_ranlib
            NAMES
                ranlib
                llvm-ranlib
                gcc-ranlib
                NO_CACHE
        )
        if(_tt_resolved_ranlib)
            set(CMAKE_RANLIB "${_tt_resolved_ranlib}" CACHE FILEPATH "Path to ranlib for static libraries" FORCE)
        endif()
        unset(_tt_resolved_ranlib)
    endif()
    foreach(_tt_lang IN ITEMS C CXX ASM)
        if(
            CMAKE_AR
            AND NOT "${CMAKE_AR}"
                MATCHES
                "NOTFOUND"
            AND (
                NOT CMAKE_${_tt_lang}_COMPILER_AR
                OR "${CMAKE_${_tt_lang}_COMPILER_AR}"
                    MATCHES
                    "NOTFOUND"
            )
        )
            set(CMAKE_${_tt_lang}_COMPILER_AR "${CMAKE_AR}" CACHE FILEPATH "Archiver for ${_tt_lang}" FORCE)
            set(CMAKE_${_tt_lang}_COMPILER_AR "${CMAKE_AR}")
        endif()
        if(
            CMAKE_RANLIB
            AND NOT "${CMAKE_RANLIB}"
                MATCHES
                "NOTFOUND"
            AND (
                NOT CMAKE_${_tt_lang}_COMPILER_RANLIB
                OR "${CMAKE_${_tt_lang}_COMPILER_RANLIB}"
                    MATCHES
                    "NOTFOUND"
            )
        )
            set(CMAKE_${_tt_lang}_COMPILER_RANLIB "${CMAKE_RANLIB}" CACHE FILEPATH "Ranlib for ${_tt_lang}" FORCE)
            set(CMAKE_${_tt_lang}_COMPILER_RANLIB "${CMAKE_RANLIB}")
        endif()
    endforeach()
    unset(_tt_lang)

    # CMake's Compiler/Clang.cmake (line ~89-107 in cmake-4.2) interpolates the value of
    # CMAKE_<LANG>_COMPILER_AR / _RANLIB *at compiler detection time* into the IPO archive
    # templates (CMAKE_<LANG>_ARCHIVE_CREATE_IPO, _APPEND_IPO, _FINISH_IPO). Those templates
    # are what static-lib link rules end up referencing for Clang on Linux, so updating the
    # cache variables alone leaves the frozen NOTFOUND strings in rules.ninja. Re-stamp the
    # templates here with the resolved tools so the rules embed valid paths in this configure
    # pass (no two-pass dance needed).
    foreach(_tt_lang IN ITEMS C CXX)
        if(CMAKE_AR AND NOT "${CMAKE_AR}" MATCHES "NOTFOUND")
            set(CMAKE_${_tt_lang}_ARCHIVE_CREATE_IPO "\"${CMAKE_AR}\" qc <TARGET> <LINK_FLAGS> <OBJECTS>")
            set(CMAKE_${_tt_lang}_ARCHIVE_APPEND_IPO "\"${CMAKE_AR}\" q <TARGET> <LINK_FLAGS> <OBJECTS>")
        endif()
        if(CMAKE_RANLIB AND NOT "${CMAKE_RANLIB}" MATCHES "NOTFOUND")
            set(CMAKE_${_tt_lang}_ARCHIVE_FINISH_IPO "\"${CMAKE_RANLIB}\" <TARGET>")
        endif()
    endforeach()
    unset(_tt_lang)

    # project() writes CMakeFiles/<ver>/CMake{C,CXX}Compiler.cmake at compiler detection time
    # and re-includes them on every subsequent configure, overriding the CACHE FORCE above.
    # Patch the persisted files so future configures (where the IPO template re-stamping is
    # not even reached if some other code calls project() again) don't reintroduce NOTFOUND.
    foreach(_tt_lang_pair IN ITEMS "C;CMakeCCompiler.cmake" "CXX;CMakeCXXCompiler.cmake")
        list(GET _tt_lang_pair 0 _tt_lang)
        list(GET _tt_lang_pair 1 _tt_filename)
        file(GLOB _tt_compiler_info_files "${CMAKE_BINARY_DIR}/CMakeFiles/*/${_tt_filename}")
        foreach(_tt_compiler_info_file IN LISTS _tt_compiler_info_files)
            file(READ "${_tt_compiler_info_file}" _tt_txt)
            if(CMAKE_AR AND NOT "${CMAKE_AR}" MATCHES "NOTFOUND")
                string(
                    REPLACE
                    "set(CMAKE_${_tt_lang}_COMPILER_AR \"CMAKE_${_tt_lang}_COMPILER_AR-NOTFOUND\")"
                    "set(CMAKE_${_tt_lang}_COMPILER_AR \"${CMAKE_AR}\")"
                    _tt_txt
                    "${_tt_txt}"
                )
            endif()
            if(CMAKE_RANLIB AND NOT "${CMAKE_RANLIB}" MATCHES "NOTFOUND")
                string(
                    REPLACE
                    "set(CMAKE_${_tt_lang}_COMPILER_RANLIB \"CMAKE_${_tt_lang}_COMPILER_RANLIB-NOTFOUND\")"
                    "set(CMAKE_${_tt_lang}_COMPILER_RANLIB \"${CMAKE_RANLIB}\")"
                    _tt_txt
                    "${_tt_txt}"
                )
            endif()
            file(WRITE "${_tt_compiler_info_file}" "${_tt_txt}")
        endforeach()
    endforeach()
    unset(_tt_lang)
    unset(_tt_filename)
    unset(_tt_lang_pair)
    unset(_tt_compiler_info_file)
    unset(_tt_compiler_info_files)
    unset(_tt_txt)
endif()

add_subdirectory(tracy/csvexport)
add_subdirectory(tracy/capture)

# Land the host-side Tracy CLI tools at ${CMAKE_BINARY_DIR}/tools/profiler/bin so they line
# up with PROFILER_BIN_DIR in tools/tracy/common.py and the path baked into the artifact
# packaging step (.github/workflows/build-artifact.yaml: TRACY_CLI_DIR=build/tools/profiler/bin)
# and the CLI smoke test (tests/scripts/tracy/tracy_cli_smoke_test.sh: TRACY_BIN_DIR). Vendored
# Tracy CMake puts them in their own subdirs by default; override here so we don't touch the
# vendored CMakeLists.
set(TRACY_CLI_BIN_DIR "${CMAKE_BINARY_DIR}/tools/profiler/bin")
set_target_properties(
    tracy-csvexport
    tracy-capture
    tracy-capture-daemon
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            "${TRACY_CLI_BIN_DIR}"
)

# Aggregate target referenced by the artifact workflow's failure message
# ("Build tracy_profiler_cli_tools before packaging the profiler artifact.") so a single
# `cmake --build ... --target tracy_profiler_cli_tools` produces both binaries.
add_custom_target(
    tracy_profiler_cli_tools
    DEPENDS
        tracy-csvexport
        tracy-capture
)

# Tracy profiler WASM viewer (Emscripten).
#
# emsdk is fetched via CPM (just the installer/manager scripts ~250 KB). The
# actual Emscripten toolchain (emcc, Emscripten.cmake, node, ...) is materialized
# alongside the scripts on first configure via emsdk's own installer
# (`./emsdk install/activate`), so neither the source tree nor the shared CPM
# cache is polluted with the install. The CPM GIT_TAG pin determines what
# `latest` resolves to.
#
# CPM_SOURCE_CACHE is shadowed for this single call so CPM falls back to its
# default ${CMAKE_BINARY_DIR}/_deps/<name>-src/ destination. emsdk.py uses
# os.path.realpath(__file__) to locate its install root, so the package must
# live somewhere we can write hundreds of MB into; the build dir is the right
# place, the shared CPM cache is not.
set(_cpm_source_cache_save "${CPM_SOURCE_CACHE}")
set(CPM_SOURCE_CACHE "")
CPMAddPackage(
    NAME emsdk
    GITHUB_REPOSITORY emscripten-core/emsdk
    GIT_TAG
        0dfff153dc3c596948ac088b8853006c570f2609 # 5.0.7-4-g0dfff15
    DOWNLOAD_ONLY YES
)
set(CPM_SOURCE_CACHE "${_cpm_source_cache_save}")
unset(_cpm_source_cache_save)

set(EMSDK_DIR "${emsdk_SOURCE_DIR}")
set(EMSCRIPTEN_TOOLCHAIN_FILE "${EMSDK_DIR}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake")

if(NOT EXISTS "${EMSCRIPTEN_TOOLCHAIN_FILE}")
    message(
        STATUS
        "Installing Emscripten via emsdk into ${EMSDK_DIR} (first configure only; downloads ~hundreds of MB)..."
    )
    execute_process(
        COMMAND
            ./emsdk install latest
        WORKING_DIRECTORY "${EMSDK_DIR}"
        RESULT_VARIABLE _emsdk_install_rc
    )
    if(NOT _emsdk_install_rc EQUAL 0)
        message(FATAL_ERROR "emsdk install failed (exit ${_emsdk_install_rc})")
    endif()
    execute_process(
        COMMAND
            ./emsdk activate latest
        WORKING_DIRECTORY "${EMSDK_DIR}"
        RESULT_VARIABLE _emsdk_activate_rc
    )
    if(NOT _emsdk_activate_rc EQUAL 0)
        message(FATAL_ERROR "emsdk activate failed (exit ${_emsdk_activate_rc})")
    endif()
endif()

if(NOT EXISTS "${EMSCRIPTEN_TOOLCHAIN_FILE}")
    message(FATAL_ERROR "Emscripten toolchain file not found at ${EMSCRIPTEN_TOOLCHAIN_FILE} after install/activate.")
endif()

# Output path is consumed by tools/tracy/common.py (PROFILER_WASM_DIR), profiler CI tests,
# and the .github/workflows/build-artifact.yaml packaging step. Keep them in sync.
ExternalProject_Add(
    tracy_profiler_wasm
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tracy/profiler
    BINARY_DIR ${CMAKE_BINARY_DIR}/profiler/build_wasm
    CMAKE_ARGS
        -DCMAKE_TOOLCHAIN_FILE=${EMSCRIPTEN_TOOLCHAIN_FILE} -DCMAKE_BUILD_TYPE=MinSizeRel
    BUILD_ALWAYS FALSE
    INSTALL_COMMAND
        ""
)
