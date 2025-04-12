cmake_minimum_required(VERSION 3.24...3.30)

# Sanity check, forgetting to clone submodules is a common omission and results in a poor error message
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/third_party/umd/CMakeLists.txt")
    message(FATAL_ERROR "Missing submodules.  Run: git submodule update --init --recursive")
endif()

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
message(STATUS "Building Unified Library for all architectures, thanks to blozano-tt")

# Defining build types is the pervue of the top-level project
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

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_CI "-O3 -DDEBUG")

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
message(STATUS "Build with ASAN: ${ENABLE_ASAN}")
message(STATUS "Build with MSAN: ${ENABLE_MSAN}")
message(STATUS "Build with TSAN: ${ENABLE_TSAN}")
message(STATUS "Build with UBSAN: ${ENABLE_UBSAN}")
message(STATUS "Build Python bindings: ${WITH_PYTHON_BINDINGS}")
message(STATUS "Build Programming Examples: ${BUILD_PROGRAMMING_EXAMPLES}")
message(STATUS "Build TT METAL Tests: ${TT_METAL_BUILD_TESTS}")
message(STATUS "Build TTNN Tests: ${TTNN_BUILD_TESTS}")
message(STATUS "Build with Unity builds: ${TT_UNITY_BUILDS}")
message(STATUS "Build with Shared TTNN Sublibraries: ${ENABLE_TTNN_SHARED_SUBLIBS}")
message(STATUS "Build with LightMetal Trace Enabled: ${TT_ENABLE_LIGHT_METAL_TRACE}")

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
        pthread
        atomic
        hwloc
        numa
)

add_compile_options(
    -Werror
    -Wno-deprecated-declarations
    -Wdelete-non-virtual-dtor
    -Wreturn-type
    -Wswitch
    -Wuninitialized
    -fPIC
    -fvisibility-inlines-hidden
    -fno-lto # FIXME: This seems to be here for ttnn; it should go to TTNN, then.
    "$<$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86_64>:-mavx2>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-narrowing>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-error=local-type-template-args>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-delete-non-abstract-non-virtual-dtor>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-c99-designator>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-shift-op-parentheses>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-non-c-typedef-for-linkage>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-this-capture>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-volatile>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-builtins>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated>"
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
add_link_options(
    "$<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>"
    "$<$<BOOL:${ENABLE_MSAN}>:-fsanitize=memory>"
    "$<$<BOOL:${ENABLE_TSAN}>:-fsanitize=thread>"
    "$<$<BOOL:${ENABLE_UBSAN}>:-fsanitize=undefined>"
)

# Planned to be temporary, remove later.
if(TT_ENABLE_LIGHT_METAL_TRACE)
    add_compile_definitions(TT_ENABLE_LIGHT_METAL_TRACE=1)
else()
    add_compile_definitions(TT_ENABLE_LIGHT_METAL_TRACE=0)
endif()

include(GNUInstallDirs)
# GNUInstallDirs takes PROJECT_DIR verbatim, but directories should always be lowercase
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)
string(REPLACE ${PROJECT_NAME} ${PROJECT_NAME_LOWER} CMAKE_INSTALL_DOCDIR ${CMAKE_INSTALL_DOCDIR})

if(ENABLE_CODE_TIMERS)
    add_compile_definitions(TT_ENABLE_CODE_TIMERS)
endif()
include(tracy)

############################################################################################################################
# Build subdirectories
############################################################################################################################

add_subdirectory(tt_stl)
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
install(
    TARGETS
        tt_metal
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT tar
)
if(WITH_PYTHON_BINDINGS)
    # Install .so into src files for pybinds implementation
    install(
        TARGETS
            ttnn
        ARCHIVE
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT tar
    )

    install(
        TARGETS
            ttnn
            DESTINATION
            ${PROJECT_SOURCE_DIR}/ttnn/ttnn
            COMPONENT
            tt_pybinds
    )
else()
    #when we don't build python bindings, we generate a dynamic library ttnncpp
    install(
        TARGETS
            ttnncpp
        ARCHIVE
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT tar
    )
endif()

# FIXME(17578): figure out what bits we actually need to ship and omit the rest
install(
    DIRECTORY
        runtime
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium
    USE_SOURCE_PERMISSIONS
    COMPONENT jit-build
)

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

include(packaging)
