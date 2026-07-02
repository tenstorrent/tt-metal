set(CMAKE_C_CLANG_TIDY "")
set(CMAKE_CXX_CLANG_TIDY "")
set(CMAKE_VERIFY_INTERFACE_HEADER_SETS FALSE)
set(TT_UMD_ENABLE_CLANG_TIDY OFF CACHE BOOL "Disable clang-tidy checks for UMD" FORCE)

if(TT_METAL_USE_EMULE)
    set(TT_UMD_BUILD_EMULE ON CACHE BOOL "Enable SWEmuleChip (forwarded from TT_METAL_USE_EMULE)" FORCE)
endif()

add_subdirectory(umd)

# Propagate ENABLE_TRACY into Tracy's option before add_subdirectory(tracy).
# Tracy's add_subdirectory must always run: it creates Tracy::TracyClient, which many
# targets link against unconditionally. With TRACY_ENABLE=OFF it becomes a no-op stub.
if(ENABLE_TRACY)
    set(TRACY_ENABLE ON)
else()
    set(TRACY_ENABLE OFF)
endif()

# Avoid hard failures on hosts without invariant TSC.
set(TRACY_TIMER_FALLBACK ON)

add_subdirectory(tracy)

# TracyClient is linked unconditionally by libtt_metal and several installed tools. With
# TRACY_ENABLE=OFF it is still built (as a no-op stub) and still baked into those binaries as a
# DT_NEEDED (libtracy.so.<ver>). It must therefore be renamed and installed in BOTH
# configurations -- otherwise CPack/dpkg-shlibdeps cannot resolve that runtime dependency when
# packaging a profiler-disabled (ENABLE_TRACY=OFF) build, and `cmake --build --target package`
# fails with "cannot find library libtracy.so.<ver>". Keep OUTPUT_NAME + install ABOVE the
# ENABLE_TRACY early-return so the dependency is satisfied regardless of ENABLE_TRACY.
set_target_properties(
    TracyClient
    PROPERTIES
        EXCLUDE_FROM_ALL
            TRUE
        LIBRARY_OUTPUT_DIRECTORY
            "${PROJECT_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY
            "${PROJECT_BINARY_DIR}/lib"
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

if(NOT ENABLE_TRACY)
    return()
endif()

# Tracy-enabled-only tuning (only meaningful when profiling is actually compiled in):
# Their code preserves frame pointers (-fno-omit-frame-pointer) → backtraces work.
target_compile_options(TracyClient PUBLIC -fno-omit-frame-pointer)
# Tracy sources emit this warning under our global -Werror; keep suppression local to TracyClient.
target_compile_options(TracyClient PRIVATE -Wno-unused-variable)
# Their executable exports symbols (-rdynamic) → Tracy can map addresses to names.
target_link_options(TracyClient PUBLIC -rdynamic)

# TracyClient is a PUBLIC dependency of TT::Metalium, so its interface exports
# $<INSTALL_INTERFACE:include/tracy>. Tracy's own header install rules are
# COMPONENT-less (they land in "Unspecified", which cmake/packaging.cmake drops),
# so the packaged SDK advertised that include dir without shipping it -> every
# downstream find_package(tt-metalium) consumer failed at configure with
# "Imported target TT::Metalium includes non-existent path .../include/tracy".
# Ship the public headers under the dev component so the SDK package is consistent.
# Layout mirrors Tracy's own install: tracy/Tracy.hpp plus its ../client and
# ../common siblings, all under <includedir>/tracy.
install(
    DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/tracy/public/tracy
        ${CMAKE_CURRENT_SOURCE_DIR}/tracy/public/client
        ${CMAKE_CURRENT_SOURCE_DIR}/tracy/public/common
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tracy
    COMPONENT metalium-dev
    FILES_MATCHING
    PATTERN
    "*.h"
    PATTERN
    "*.hpp"
)

set(NO_FILESELECTOR ON)

# Build Tracy CLI deps statically so the binaries run from packaged artifacts without build/_deps.
set(BUILD_SHARED_LIBS OFF)

# Disable Tracy's -march=native for portable CLI binaries in CI.
set(NO_ISA_EXTENSIONS ON)

# Tracy's csvexport/capture each include cmake/config.cmake, which sets
# RULE_LAUNCH_COMPILE ccache as a GLOBAL property.  tt-metal already wires
# ccache via CMAKE_CXX_COMPILER_LAUNCHER (cmake/ccache.cmake), so having both
# active causes double-wrapping: the outer ccache sees the entire
# "cmake -E env ... ccache clang++ -c file.cpp" as its argument list and
# rejects every tt_metal translation unit as "Multiple source files".
# Suppress Tracy's own ccache logic; ours handles it correctly.
set(NO_CCACHE ON)
add_subdirectory(tracy/csvexport)
add_subdirectory(tracy/capture)

# Keep CLI output in the path used by Python tooling and CI packaging.
set(TRACY_CLI_BIN_DIR "${CMAKE_BINARY_DIR}/tools/profiler/bin")
set_target_properties(
    tracy-csvexport
    tracy-capture
    tracy-capture-daemon
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            "${TRACY_CLI_BIN_DIR}"
)

# Force a portable ISA baseline for Tracy server/CLI and key deps on x86_64.
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(_tt_tracy_portable_march -march=x86-64-v2)
    foreach(
        _tt_tracy_portable_target
        IN
        ITEMS
            TracyServer
            tracy-capture
            tracy-csvexport
            tracy-capture-daemon
            capstone
            libzstd
            libzstd_static
            PPQSort
            TracyGetOpt
    )
        if(NOT TARGET ${_tt_tracy_portable_target})
            continue()
        endif()
        get_target_property(_tt_tracy_portable_target_type ${_tt_tracy_portable_target} TYPE)
        if(_tt_tracy_portable_target_type STREQUAL "INTERFACE_LIBRARY")
            continue()
        endif()
        target_compile_options(${_tt_tracy_portable_target} PRIVATE ${_tt_tracy_portable_march})
    endforeach()
    unset(_tt_tracy_portable_march)
    unset(_tt_tracy_portable_target)
    unset(_tt_tracy_portable_target_type)
endif()

# Aggregate target used by CI/artifact workflows.
add_custom_target(
    tracy_profiler_cli_tools
    DEPENDS
        tracy-csvexport
        tracy-capture
)

# Tracy profiler WASM viewer (Emscripten).
CPMAddPackage(
    NAME emsdk
    GITHUB_REPOSITORY emscripten-core/emsdk
    GIT_TAG
        0dfff153dc3c596948ac088b8853006c570f2609 # 5.0.7-4-g0dfff15
    DOWNLOAD_ONLY YES
)

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

# Keep output path aligned with Python tooling and CI packaging/tests.
# No EXCLUDE_FROM_ALL: Ninja's install target depends on all, so cmake --build --target install
# builds this automatically when Tracy is enabled.
ExternalProject_Add(
    tracy_profiler_wasm
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tracy/profiler
    BINARY_DIR ${CMAKE_BINARY_DIR}/profiler/build_wasm
    CMAKE_ARGS
        -DCMAKE_TOOLCHAIN_FILE=${EMSCRIPTEN_TOOLCHAIN_FILE} -DCMAKE_BUILD_TYPE=MinSizeRel
    INSTALL_COMMAND
        "" # Suppress default cmake --install: artifacts stay in BINARY_DIR for CI packaging to pick up.
)
