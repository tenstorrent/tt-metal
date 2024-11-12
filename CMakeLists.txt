# Copyright
#
# © 2023 Tenstorrent Inc.
# © 2024 Tactical Computing Labs, LLC
#

cmake_minimum_required(VERSION 3.16..3.30)

# Sanity check, forgetting to clone submodules is a common omission and results in a poor error message
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/third_party/umd/CMakeLists.txt")
    message(FATAL_ERROR "Missing submodules.  Run: git submodule update --init --recursive")
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set (X86 TRUE)
else ()
    set (X86 FALSE)
endif ()

############################################
# Project setup
############################################
include(cmake/compilers.cmake)

if(DEFINED ENV{CMAKE_C_COMPILER} AND DEFINED ENV{CMAKE_CXX_COMPILER})
    message(STATUS "Setting C and C++ compiler from environment variables")
    set(CMAKE_C_COMPILER $ENV{CMAKE_C_COMPILER})
    set(CMAKE_CXX_COMPILER $ENV{CMAKE_CXX_COMPILER})
endif()

if(CMAKE_CXX_COMPILER AND CMAKE_C_COMPILER)
    message(STATUS "Using specifed C++ compiler: ${CMAKE_CXX_COMPILER}")
    message(STATUS "Using specifed C compiler: ${CMAKE_C_COMPILER}")
else()
    message(STATUS "No C or C++ compiler specified, defaulting to Clang-17")
    FIND_AND_SET_CLANG17()
endif()

project(
    Metalium
    VERSION 0.50.0
    DESCRIPTION "Tenstorrent Metalium"
    HOMEPAGE_URL "https://github.com/tenstorrent/tt-metal"
    LANGUAGES
        CXX
)

if(${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    message(
        FATAL_ERROR
        "CMake generation is not allowed within source directory!! Please set a build folder with '-B'!!"
    )
endif()

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(project_options)
include(unity)
include(clang-tidy)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(ENABLE_LIBCXX)
    add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,Clang>:-stdlib=libc++>)
    #add_link_options(
    #    $<$<LINK_LANG_AND_ID:CXX,Clang>:-lc++>
    #    $<$<LINK_LANG_AND_ID:CXX,Clang>:-lc++abi>
    #)
else()
    # required when linking with libstdc++ with clang and gcc
    add_compile_options(-fsized-deallocation)
endif()

# Using below until we can move to CMake >= 3.18 for LINK_LANG_AND_ID
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND ENABLE_LIBCXX)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++ -lc++abi")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lc++ -lc++abi")
endif()

include(CTest)

get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

# Global settings if we're the top-level project
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CMAKE_CONFIGURATION_TYPES
        Release
        RelWithDebInfo
        Debug
        CI
    )
    if(NOT CMAKE_BUILD_TYPE AND NOT isMultiConfig)
        message(STATUS "Setting build type to 'Release' as none was specified.")
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release build is the default" FORCE)
    endif()

    set_property(
        GLOBAL
        PROPERTY
            GLOBAL_DEPENDS_NO_CYCLES
                FALSE #FIXME(14541): We've got work to do :(
    )

    if(ENABLE_CCACHE)
        include(cmake/ccache.cmake)
    endif()
endif()

CHECK_COMPILERS()

if(NOT isMultiConfig AND NOT CMAKE_BUILD_TYPE IN_LIST CMAKE_CONFIGURATION_TYPES)
    message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}. Valid options are: ${CMAKE_CONFIGURATION_TYPES}")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_CI "-O3 -DDEBUG")

# Set default values for variables/options
set(UMD_HOME "${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd")

############################################################################################################################
# Project Options
#   The following options and their defaults impact what artifacts get built
############################################################################################################################
message(STATUS "Build shared libs: ${BUILD_SHARED_LIBS}")
message(STATUS "Build with ASAN: ${ENABLE_ASAN}")
message(STATUS "Build with MSAN: ${ENABLE_MSAN}")
message(STATUS "Build with TSAN: ${ENABLE_TSAN}")
message(STATUS "Build with UBSAN: ${ENABLE_UBSAN}")
message(STATUS "Build Python bindings: ${WITH_PYTHON_BINDINGS}")
message(STATUS "Build Programming Examples: ${BUILD_PROGRAMMING_EXAMPLES}")
message(STATUS "Build TT METAL Tests: ${TT_METAL_BUILD_TESTS}")
message(STATUS "Build TTNN Tests: ${TTNN_BUILD_TESTS}")
message(STATUS "Build with Unity builds: ${TT_UNITY_BUILDS}")
############################################################################################################################

if(ENABLE_BUILD_TIME_TRACE)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        message(STATUS "Adding compile option: -ftime-trace")
        add_compile_options("-ftime-trace")
    else()
        message(FATAL_ERROR "ENABLE_BUILD_TIME_TRACE is only supported with Clang")
    endif()
endif()

set(SANITIZER_ENABLED ${ENABLE_ASAN})
if(SANITIZER_ENABLED AND ENABLE_MSAN)
    message(FATAL_ERROR "Multiple sanitizers are not supported")
elseif(ENABLE_MSAN)
    set(SANITIZER_ENABLED ${ENABLE_MSAN})
endif()

if(SANITIZER_ENABLED AND ENABLE_TSAN)
    message(FATAL_ERROR "Multiple sanitizers are not supported")
elseif(ENABLE_TSAN)
    set(SANITIZER_ENABLED ${ENABLE_TSAN})
endif()

if(SANITIZER_ENABLED AND ENABLE_UBSAN)
    message(FATAL_ERROR "Multiple sanitizers are not supported")
endif()

unset(SANITIZER_ENABLED)

############################################################################################################################
# Find all required libraries to build
############################################################################################################################
set(ENV{CPM_SOURCE_CACHE} "${PROJECT_SOURCE_DIR}/.cpmcache")
include(CPM)
if(CMAKE_VERSION VERSION_LESS 3.25)
    # FIXME(14681): `SYSTEM` was introduced in v3.25; remove this when we can require v3.25
    add_subdirectory(dependencies EXCLUDE_FROM_ALL)
else()
    add_subdirectory(dependencies EXCLUDE_FROM_ALL SYSTEM)
endif()

if(WITH_PYTHON_BINDINGS)
    set(Python3_FIND_STRATEGY LOCATION)
    find_package(
        Python3
        COMPONENTS
            Interpreter
            Development
    )
    message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
endif()

find_library(NUMA_LIBRARY NAMES numa)
if(NOT NUMA_LIBRARY)
    message(FATAL_ERROR "NUMA library not found")
endif()

find_package(umd)
find_package(nng)
find_package(flatbuffers)
find_package(libuv)

# Bring in UMD and all it's dependencies
add_subdirectory(tt_metal/third_party/umd)

############################################################################################################################
# Constructing interface libs for common compiler flags, header directories, and libraries
#   These interface libs are linked with PUBLIC scope at lowest common target (tt_metal/common) and at tt_metal_libs level
#   in order to propogate to the rest of tt_metal, tt_eager, etc.
############################################################################################################################
add_library(metal_common_libs INTERFACE)
target_link_libraries(
    metal_common_libs
    INTERFACE
        dl
        z
        pthread
        atomic
        hwloc
        numa
)

string(TOUPPER "$ENV{ARCH_NAME}" ARCH_NAME_DEF)
add_compile_definitions(ARCH_${ARCH_NAME_DEF})

if(X86)

add_compile_options(
    -Werror
    -Wdelete-non-virtual-dtor
    -Wreturn-type
    -Wswitch
    -Wuninitialized
    -Wno-unused-parameter
    -mavx2
    -fPIC
    -fvisibility-inlines-hidden
    -fno-lto # FIXME: This seems to be here for ttnn; it should go to TTNN, then.
    "$<$<CXX_COMPILER_ID:Clang>:-Wsometimes-uninitialized>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-narrowing>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-error=local-type-template-args>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-delete-non-abstract-non-virtual-dtor>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-c99-designator>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-shift-op-parentheses>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-non-c-typedef-for-linkage>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-this-capture>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-volatile>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-builtins>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-declarations>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated-declarations>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-attributes>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-stringop-overread>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-stringop-overflow>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-missing-requires>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-non-template-friend>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-error=non-template-friend>"
    "$<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>"
    "$<$<BOOL:${ENABLE_MSAN}>:-fsanitize=memory>"
    "$<$<BOOL:${ENABLE_TSAN}>:-fsanitize=thread>"
    "$<$<BOOL:${ENABLE_UBSAN}>:-fsanitize=undefined>"
)

else()

add_compile_options(
    -Werror
    -Wdelete-non-virtual-dtor
    -Wreturn-type
    -Wswitch
    -Wuninitialized
    -Wno-unused-parameter
    -fPIC
    -fvisibility-inlines-hidden
    -fno-lto # FIXME: This seems to be here for ttnn; it should go to TTNN, then.
    "$<$<CXX_COMPILER_ID:Clang>:-Wsometimes-uninitialized>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-narrowing>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-error=local-type-template-args>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-delete-non-abstract-non-virtual-dtor>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-c99-designator>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-shift-op-parentheses>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-non-c-typedef-for-linkage>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-this-capture>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-volatile>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-builtins>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-declarations>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated-declarations>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-attributes>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-stringop-overread>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-stringop-overflow>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-missing-requires>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-non-template-friend>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-error=non-template-friend>"
    "$<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>"
    "$<$<BOOL:${ENABLE_MSAN}>:-fsanitize=memory>"
    "$<$<BOOL:${ENABLE_TSAN}>:-fsanitize=thread>"
    "$<$<BOOL:${ENABLE_UBSAN}>:-fsanitize=undefined>"
)

endif()

add_link_options(
    "$<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>"
    "$<$<BOOL:${ENABLE_MSAN}>:-fsanitize=memory>"
    "$<$<BOOL:${ENABLE_TSAN}>:-fsanitize=thread>"
    "$<$<BOOL:${ENABLE_UBSAN}>:-fsanitize=undefined>"
)

if(ENABLE_CODE_TIMERS)
    add_compile_definitions(TT_ENABLE_CODE_TIMERS)
endif()
if(ENABLE_TRACY)
    add_compile_definitions(TRACY_ENABLE)
    add_compile_options(-fno-omit-frame-pointer)
    add_link_options(-rdynamic)
endif()

if(WITH_PYTHON_BINDINGS)
    # Can't use the `REUSE_FROM` option bc tt_lib and ttnn have different build flags :(
    add_library(pch_pybinds INTERFACE)
    target_precompile_headers(
        pch_pybinds
        INTERFACE
            ${PROJECT_SOURCE_DIR}/tt_metal/third_party/pybind11/include/pybind11/operators.h
            ${PROJECT_SOURCE_DIR}/tt_metal/third_party/pybind11/include/pybind11/pybind11.h
            ${PROJECT_SOURCE_DIR}/tt_metal/third_party/pybind11/include/pybind11/stl.h
    )
endif()

############################################################################################################################
# Build subdirectories
############################################################################################################################
if(ENABLE_TRACY)
    include(tracy)
endif()

add_subdirectory(tt_metal)
add_subdirectory(ttnn)

if(TT_METAL_BUILD_TESTS OR TTNN_BUILD_TESTS)
    add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
endif()

############################################################################################################################
# Install targets for build artifacts and pybinds
#   If built with Tracy, cannot install 'all' since it will pick up install targets from Tracy
# For top level install: cmake --build build --target install  or  make/ninja install -C build
############################################################################################################################
# Install for build artifacts that will upload build/lib
include(GNUInstallDirs)

install(
    TARGETS
        tt_metal
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT dev
)
install(
    TARGETS
        ttnn
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT dev
)
if(WITH_PYTHON_BINDINGS)
    # Install .so into src files for pybinds implementation
    install(
        TARGETS
            ttnn
            DESTINATION
            ${PROJECT_SOURCE_DIR}/ttnn/ttnn
            COMPONENT
            tt_pybinds
    )
endif()

# Custom clean target for `built` folder for when new kernel changes are pulled
add_custom_target(
    clean-built
    COMMAND
        ${CMAKE_COMMAND} -E remove_directory ${PROJECT_SOURCE_DIR}/built
    COMMENT "Cleaning `built` directory"
)

include(packaging)
