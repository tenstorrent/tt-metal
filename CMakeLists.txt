cmake_minimum_required(VERSION 3.24...3.30)

# Sanity check, forgetting to clone submodules is a common omission and results in a poor error message
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/third_party/umd/CMakeLists.txt")
    message(FATAL_ERROR "Missing submodules.  Run: git submodule update --init --recursive")
endif()

# Clear any compile options that may have been set by consuming projects that have leaked into us.
set_directory_properties(
    PROPERTIES
        COMPILE_OPTIONS
            ""
)

############################################
# Project setup
############################################

# For single-config generators, default to RelWithDebInfo if unspecified
get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT isMultiConfig)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type")
endif()

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(version)
ParseGitDescribe()
project(
    Metalium
    VERSION ${VERSION_NUMERIC}
    DESCRIPTION "Tenstorrent Metalium"
    HOMEPAGE_URL "https://github.com/tenstorrent/tt-metal"
    LANGUAGES
        C # Some of the jit-build files are plain C
        CXX
)
message(STATUS "Metalium version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -ggnu-pubnames -DDEBUG -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -ggnu-pubnames -DDEBUG -fno-omit-frame-pointer")

# Defining build types is the purview of the top-level project
if(PROJECT_IS_TOP_LEVEL)
    include(sanitizers)
endif()

if(${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    message(
        FATAL_ERROR
        "CMake generation is not allowed within source directory!! Please set a build folder with '-B'!!"
    )
endif()

if(DEFINED CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE)
    message(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
else()
    message(STATUS "CMAKE_TOOLCHAIN_FILE is not set.")
endif()

include(project_options)
include(unity)
include(clang-tidy)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# CMake 3.19: "smartly dropping dependencies of static and object libraries"
set(CMAKE_OPTIMIZE_DEPENDENCIES TRUE)

if(NOT ENABLE_LIBCXX)
    # required when linking with libstdc++ with clang and gcc
    add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fsized-deallocation>)
endif()

include(CTest)

get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

# Global settings if we're the top-level project
if(PROJECT_IS_TOP_LEVEL)
    set_property(
        GLOBAL
        PROPERTY
            GLOBAL_DEPENDS_NO_CYCLES
                TRUE
    )

    if(ENABLE_CCACHE)
        include(ccache)
    endif()
endif()

include(compilers)
CHECK_COMPILERS()

include(linking)

# We're not currently using C++20 modules, so don't bother scanning for them
set(CMAKE_CXX_SCAN_FOR_MODULES FALSE)

# Promote all IMPORTED targets discovered by find_package() to a GLOBAL scope
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL TRUE)

############################################################################################################################
# Project Options
#   The following options and their defaults impact what artifacts get built
############################################################################################################################
message(STATUS "ENABLE_LIBCXX: ${ENABLE_LIBCXX}")
message(STATUS "Build shared libs: ${BUILD_SHARED_LIBS}")
message(STATUS "Build Python bindings: ${WITH_PYTHON_BINDINGS}")
message(STATUS "Build Programming Examples: ${BUILD_PROGRAMMING_EXAMPLES}")
message(STATUS "Build TT METAL Tests: ${TT_METAL_BUILD_TESTS}")
message(STATUS "Build TTNN Tests: ${TTNN_BUILD_TESTS}")
message(STATUS "Build TT Telemetry Server: ${BUILD_TELEMETRY}")
message(STATUS "Build with Unity builds: ${TT_UNITY_BUILDS}")
message(STATUS "Build with LTO: ${TT_LTO_ENABLED}")
message(STATUS "Build with Shared TTNN Sublibraries: ${ENABLE_TTNN_SHARED_SUBLIBS}")
message(STATUS "Build with LightMetal Trace Enabled: ${TT_ENABLE_LIGHT_METAL_TRACE}")
message(STATUS "Build with Multihost Distributed Compute: ${ENABLE_DISTRIBUTED}")

############################################################################################################################

if(ENABLE_BUILD_TIME_TRACE)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        message(STATUS "Adding compile option: -ftime-trace")
        add_compile_options("-ftime-trace")
    else()
        message(FATAL_ERROR "ENABLE_BUILD_TIME_TRACE is only supported with Clang")
    endif()
endif()

# Top-level target to build all generated files
# Useful for eg: ensuring all files exist before linting
add_custom_target(all_generated_files)

include(CPM)
if(CMAKE_VERSION VERSION_LESS 3.25)
    # FIXME(14681): `SYSTEM` was introduced in v3.25; remove this when we can require v3.25
    add_subdirectory(third_party)
    add_subdirectory(tt_metal/third_party)
else()
    add_subdirectory(third_party SYSTEM)
    add_subdirectory(tt_metal/third_party SYSTEM)
endif()

############################################################################################################################
# Constructing interface libs for common compiler flags, header directories, and libraries
#   These interface libs are linked with PUBLIC scope at lowest common target (tt_metal/common) and at tt_metal_libs level
#   in order to propagate to the rest of tt_metal, tt_eager, etc.
############################################################################################################################
add_library(metal_common_libs INTERFACE)
target_link_libraries(
    metal_common_libs
    INTERFACE
        dl
        pthread
        hwloc
        numa
)

add_compile_options(
    "$<$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86_64>:-march=x86-64-v3>"
    -fPIC
    -pipe # use pipes instead of tmp files
    -fvisibility-inlines-hidden
    -Wall
    -Werror
    "$<$<CXX_COMPILER_ID:Clang>:-Wconditional-uninitialized>"
    # The following list is not indicative of what we _desire_ only the status quo to build.
    -Wno-deprecated-declarations
    "$<$<CXX_COMPILER_ID:Clang>:SHELL:-Xclang -fno-pch-timestamp>" # ccache-friendly PCH flag
    "$<$<CXX_COMPILER_ID:GNU>:-fpch-preprocess>" # ccache-friendly PCH flag
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-array-bounds>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-int-in-bool-context>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-non-template-friend>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-restrict>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-strict-aliasing>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-stringop-overflow>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-stringop-overread>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-unused-local-typedefs>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-pessimizing-move>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-dangling-reference>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-overloaded-virtual>"
)

# Planned to be temporary, remove later.
if(TT_ENABLE_LIGHT_METAL_TRACE)
    add_compile_definitions(TT_ENABLE_LIGHT_METAL_TRACE=1)
else()
    add_compile_definitions(TT_ENABLE_LIGHT_METAL_TRACE=0)
endif()

# Control compile time logging for TT_METAL - needs to be set before PCH compilation
if(NOT DEFINED TT_METAL_ENABLE_LOGGING)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(_default_enable_logging OFF)
    else()
        set(_default_enable_logging ON)
    endif()
endif()
option(
    TT_METAL_ENABLE_LOGGING
    "Enables inclusion of log_trace and log_debug into the binary."
    ${_default_enable_logging}
)
if(TT_METAL_ENABLE_LOGGING)
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)
endif()

include(GNUInstallDirs)
# GNUInstallDirs takes PROJECT_DIR verbatim, but directories should always be lowercase
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)
string(REPLACE ${PROJECT_NAME} ${PROJECT_NAME_LOWER} CMAKE_INSTALL_DOCDIR ${CMAKE_INSTALL_DOCDIR})

if(ENABLE_CODE_TIMERS)
    add_compile_definitions(TT_ENABLE_CODE_TIMERS)
endif()
include(tracy)

if(ENABLE_TRACY AND TARGET TracyClient)
    message(STATUS "Adding -Wno-unused-variable to TracyClient")
    target_compile_options(TracyClient PRIVATE -Wno-unused-variable)
endif()

add_library(metal_common_pch STATIC)
add_library(TT::CommonPCH ALIAS metal_common_pch)
file(GENERATE OUTPUT metal_common_pch.cpp CONTENT "/*dummy pch file*/\n")
target_sources(metal_common_pch PRIVATE metal_common_pch.cpp)

target_precompile_headers(
    metal_common_pch
    PUBLIC
        <fmt/core.h>
        <fmt/format.h>
        <nlohmann/json.hpp>
        <algorithm>
        <cstddef>
        <cstdint>
        <functional>
        <map>
        <memory>
        <tuple>
        <unordered_map>
        <variant>
        <vector>
)

target_link_libraries(
    metal_common_pch
    PRIVATE
        nlohmann_json::nlohmann_json
        fmt::fmt-header-only
)

############################################################################################################################
# Build subdirectories
############################################################################################################################

add_subdirectory(tt_stl)
add_subdirectory(tools)
add_subdirectory(tt_metal)
add_subdirectory(ttnn)

if(TT_METAL_BUILD_TESTS OR TTNN_BUILD_TESTS)
    add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
endif()

if(BUILD_TELEMETRY)
    add_subdirectory(${PROJECT_SOURCE_DIR}/tt_telemetry)
endif()

############################################################################################################################
# Install targets for build artifacts and nanobinds
#   If built with Tracy, cannot install 'all' since it will pick up install targets from Tracy
# For top level install: cmake --build build --target install  or  make/ninja install -C build
############################################################################################################################
# Install for build artifacts that will upload build/lib
install(
    TARGETS
        tt_metal
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT tar
)

if(NOT TT_USE_SYSTEM_SFPI)
    # FIXME(17578): figure out what bits we actually need to ship and omit the rest
    install(
        DIRECTORY
            runtime/sfpi
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/runtime
        USE_SOURCE_PERMISSIONS
        COMPONENT jit-build
    )
endif()

# Custom clean target for `built` folder for when new kernel changes are pulled
add_custom_target(
    clean-built
    COMMAND
        ${CMAKE_COMMAND} -E remove_directory ${PROJECT_SOURCE_DIR}/built
    COMMENT "Cleaning `built` directory"
)

if(BUILD_TT_TRAIN)
    add_subdirectory(tt-train)
endif()

# If we've been asked to build static libs, packaging is left up to them to sort out.
if(TT_INSTALL)
    include(packaging)
endif()
