cmake_minimum_required(VERSION 3.16)
cmake_policy(VERSION 3.16)

############################################
# Project setup
############################################
include(cmake/compilers.cmake)

if (DEFINED ENV{CMAKE_C_COMPILER} AND DEFINED ENV{CMAKE_CXX_COMPILER})
    message(STATUS "Setting C and C++ compiler from environment variables")
    set(CMAKE_C_COMPILER $ENV{CMAKE_C_COMPILER})
    set(CMAKE_CXX_COMPILER $ENV{CMAKE_CXX_COMPILER})
endif()

if (CMAKE_CXX_COMPILER AND CMAKE_C_COMPILER)
    message(STATUS "Using specifed C++ compiler: ${CMAKE_CXX_COMPILER}")
    message(STATUS "Using specifed C compiler: ${CMAKE_C_COMPILER}")
else()
    message(STATUS "No C or C++ compiler specified, defaulting to Clang-17")
    FIND_AND_SET_CLANG17()
endif()

if(${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    message(FATAL_ERROR "CMake generation is not allowed within source directory!! Please set a build folder with '-B'!!")
endif()

project(tt-metal
        VERSION 1.0
        DESCRIPTION "Tenstorrent Metallium"
        HOMEPAGE_URL "https://github.com/tenstorrent/tt-metal"
        LANGUAGES CXX
)

CHECK_COMPILERS()

############################################################################################################################
# Setting build type flags
#   Will default to Release build unless manually set with -DCMAKE_BUILD_TYPE
############################################################################################################################
# Define valid build types
set(VALID_BUILD_TYPES Debug Release RelWithDebInfo CI)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release build is the default" FORCE)
endif()

# Check if the specified build type is valid
list(FIND VALID_BUILD_TYPES ${CMAKE_BUILD_TYPE} _build_type_index)

if(_build_type_index EQUAL -1)
    message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}. Valid options are: ${VALID_BUILD_TYPES}")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG=DEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DDEBUG=DEBUG")
set(CMAKE_CXX_FLAGS_CI "-O3 -DDEBUG=DEBUG")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default values for variables/options
set(UMD_HOME "${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd")

option(WITH_PYTHON_BINDINGS "Enables build of python bindings" ON)
message(STATUS "Build Python bindings: ${WITH_PYTHON_BINDINGS}")

option(ENABLE_CODE_TIMERS "Enable code timers" OFF)

option(ENABLE_TRACY "Enable Tracy Profiling" OFF)

option(ENABLE_LIBCXX "Enable using libc++" ON)


option(ENABLE_BUILD_TIME_TRACE "Enable build time trace (Clang only -ftime-trace)" OFF)
if(ENABLE_BUILD_TIME_TRACE)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        message(STATUS "Adding compile option: -ftime-trace")
        add_compile_options("-ftime-trace")
    else()
        message(FATAL "ENABLE_BUILD_TIME_TRACE is only supported with Clang")
    endif()
endif()

# Default to building everything as a shared lib
if($ENV{TT_METAL_CREATE_STATIC_LIB})
    option(BUILD_SHARED_LIBS "Create shared library" OFF)
else()
    option(BUILD_SHARED_LIBS "Create shared library" ON)
endif()
message(STATUS "Build shared libs: ${BUILD_SHARED_LIBS}")

option(ENABLE_ASAN "Enable build with AddressSanitizer" OFF)
message(STATUS "Build with ASAN: ${ENABLE_ASAN}")

set(SANITIZER_ENABLED ${ENABLE_ASAN})

option(ENABLE_MSAN "Enable build with MemorySanitizer" OFF)
message(STATUS "Build with MSAN: ${ENABLE_MSAN}")

if(SANITIZER_ENABLED AND ENABLE_MSAN)
    message(FATAL_ERROR "Multiple sanitizers are not supported")
elseif(ENABLE_MSAN)
    set(SANITIZER_ENABLED ${ENABLE_MSAN})
endif()

option(ENABLE_TSAN "Enable build with ThreadSanitizer" OFF)
message(STATUS "Build with TSAN: ${ENABLE_TSAN}")

if(SANITIZER_ENABLED AND ENABLE_TSAN)
    message(FATAL_ERROR "Multiple sanitizers are not supported")
elseif(ENABLE_TSAN)
    set(SANITIZER_ENABLED ${ENABLE_TSAN})
endif()

option(ENABLE_UBSAN "Enable build with UndefinedBehaviorSanitizer" OFF)
message(STATUS "Build with UBSAN: ${ENABLE_UBSAN}")

if(SANITIZER_ENABLED AND ENABLE_UBSAN)
    message(FATAL_ERROR "Multiple sanitizers are not supported")
endif()

unset(SANITIZER_ENABLED)

include(GNUInstallDirs)
set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}")
set(CMAKE_INSTALL_LIBDIR "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_INSTALL_BINDIR "${PROJECT_BINARY_DIR}/tmp/bin")
set(CMAKE_INSTALL_INCLUDEDIR "${PROJECT_BINARY_DIR}/tmp/include")
set(CMAKE_INSTALL_DATAROOTDIR "${PROJECT_BINARY_DIR}/tmp/share")

############################################################################################################################
# Find all required libraries to build
############################################################################################################################
include(${PROJECT_SOURCE_DIR}/cmake/dependencies.cmake)

if(WITH_PYTHON_BINDINGS)
    set(Python3_FIND_STRATEGY LOCATION)
    find_package (Python3 COMPONENTS Interpreter Development)
    message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
endif()

find_library(NUMA_LIBRARY NAMES numa)
if (NOT NUMA_LIBRARY)
    message(FATAL_ERROR "NUMA library not found")
endif()

# Bring in UMD and all it's dependencies
add_subdirectory(${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd)

############################################################################################################################
# Constructing interface libs for common compiler flags, header directories, and libraries
#   These interface libs are linked with PUBLIC scope at lowest common target (tt_metal/common) and at tt_metal_libs level
#   in order to propogate to the rest of tt_metal, tt_eager, etc.
############################################################################################################################
add_library(stdlib INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND ENABLE_LIBCXX)
    find_library(LIBC++ c++)
    find_library(LIBC++ABI c++abi)
    if(NOT LIBC++ OR NOT LIBC++ABI)
        message(FATAL_ERROR "libc++ or libc++abi not found. Make sure you have libc++ and libc++abi installed and in your PATH")
    endif()

    target_link_libraries(stdlib INTERFACE ${LIBC++} ${LIBC++ABI})
    target_compile_options(stdlib INTERFACE -stdlib=libc++)
else()
    target_link_libraries(stdlib INTERFACE stdc++)
    target_compile_options(stdlib INTERFACE -fsized-deallocation)
endif()

add_library(metal_common_libs INTERFACE)
target_link_libraries(metal_common_libs INTERFACE
    dl z pthread atomic hwloc numa stdlib # system libraries, hwloc has no cmake support, find_package won't find it
)

# Note on flags:
#   DFMT_HEADER_ONLY must be for every target or else they won't interact with the header only fmt as intended
#   ttnn and tt_lib will break if built with LTO, so leaving -fno-lto in compile options
add_library(linker_flags INTERFACE)

add_library(compiler_warnings INTERFACE)
target_compile_options(compiler_warnings INTERFACE -Werror -Wdelete-non-virtual-dtor -Wreturn-type -Wswitch -Wuninitialized -Wno-unused-parameter)

# add additional compile warning flags depending on the compiler
ADJUST_COMPILER_WARNINGS()

add_library(compiler_flags INTERFACE)
target_link_libraries(compiler_flags INTERFACE linker_flags compiler_warnings stdlib)
target_compile_options(compiler_flags INTERFACE -mavx2 -fPIC -DFMT_HEADER_ONLY -fvisibility-inlines-hidden -fno-lto)

if(ENABLE_CODE_TIMERS)
    target_compile_options(compiler_flags INTERFACE -DTT_ENABLE_CODE_TIMERS)
endif()
if(ENABLE_TRACY)
    target_compile_options(compiler_flags INTERFACE -DTRACY_ENABLE -fno-omit-frame-pointer)
    target_link_options(linker_flags INTERFACE -rdynamic)
endif()
target_compile_options(
    compiler_flags
    INTERFACE $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>
              $<$<BOOL:${ENABLE_MSAN}>:-fsanitize=memory>
              $<$<BOOL:${ENABLE_TSAN}>:-fsanitize=thread>
              $<$<BOOL:${ENABLE_UBSAN}>:-fsanitize=undefined>)
target_link_options(
    linker_flags
    INTERFACE
    $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>
    $<$<BOOL:${ENABLE_MSAN}>:-fsanitize=memory>
    $<$<BOOL:${ENABLE_TSAN}>:-fsanitize=thread>
    $<$<BOOL:${ENABLE_UBSAN}>:-fsanitize=undefined>)

string(TOUPPER $ENV{ARCH_NAME} ARCH_NAME_DEF)
target_compile_options(compiler_flags INTERFACE -DARCH_${ARCH_NAME_DEF})

add_library(metal_header_directories INTERFACE)
target_include_directories(metal_header_directories INTERFACE ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc)
target_include_directories(metal_header_directories SYSTEM INTERFACE ${reflect_SOURCE_DIR} ${flatbuffers_include_dir} ${nng_include_dir})
foreach(lib ${BoostPackages})
    target_include_directories(metal_header_directories INTERFACE ${Boost${lib}_SOURCE_DIR}/include)
endforeach()

if ("$ENV{ARCH_NAME}" STREQUAL "wormhole_b0")
    target_include_directories(metal_header_directories INTERFACE tt_metal/hw/inc/wormhole
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/wormhole/wormhole_b0_defines
        ${UMD_HOME}/device/wormhole
        ${UMD_HOME}/src/firmware/riscv/wormhole
    )
else()
    target_compile_options(compiler_flags INTERFACE -DDISABLE_ISSUE_3487_FIX)
    target_include_directories(metal_header_directories INTERFACE
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/$ENV{ARCH_NAME}
        ${UMD_HOME}/device/$ENV{ARCH_NAME}
        ${UMD_HOME}/src/firmware/riscv/$ENV{ARCH_NAME}
    )
endif()

if(WITH_PYTHON_BINDINGS)
    # Can't use the `REUSE_FROM` option bc tt_lib and ttnn have different build flags :(
    add_library(pch_pybinds INTERFACE)
    target_precompile_headers(pch_pybinds INTERFACE
        ${PROJECT_SOURCE_DIR}/tt_metal/third_party/pybind11/include/pybind11/operators.h
        ${PROJECT_SOURCE_DIR}/tt_metal/third_party/pybind11/include/pybind11/pybind11.h
        ${PROJECT_SOURCE_DIR}/tt_metal/third_party/pybind11/include/pybind11/stl.h
    )
endif()

############################################################################################################################
# Build subdirectories
############################################################################################################################
if(ENABLE_TRACY)
    include(${PROJECT_SOURCE_DIR}/cmake/tracy.cmake)
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/tt_metal)
add_subdirectory(${PROJECT_SOURCE_DIR}/ttnn)

add_subdirectory(${PROJECT_SOURCE_DIR}/tests EXCLUDE_FROM_ALL)

############################################################################################################################
# Install targets for build artifacts and pybinds
#   If built with Tracy, cannot install 'all' since it will pick up install targets from Tracy
# For top level install: cmake --build build --target install  or  make/ninja install -C build
############################################################################################################################
# Install for build artifacts that will upload build/lib
install(TARGETS tt_metal
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT tt_build_artifacts
)
install(TARGETS ttnn
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT tt_build_artifacts
)
if(WITH_PYTHON_BINDINGS)
    # Install .so into src files for pybinds implementation
    install(FILES ${PROJECT_BINARY_DIR}/lib/_ttnn.so
    DESTINATION ${PROJECT_SOURCE_DIR}/ttnn/ttnn
    COMPONENT tt_pybinds
    )
endif()

# Custom clean target for `built` folder for when new kernel changes are pulled
add_custom_target(clean-built
   COMMAND ${CMAKE_COMMAND} -E remove_directory ${PROJECT_SOURCE_DIR}/built
   COMMENT "Cleaning `built` directory"
)

set_target_properties(ttnn PROPERTIES INSTALL_RPATH "$ORIGIN" BUILD_RPATH "$ORIGIN")
