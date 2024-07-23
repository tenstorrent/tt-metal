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
#   Will default to assert build, unless CONFIG env variable is set or manually set -DCMAKE_BUILD_TYPE
############################################################################################################################
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release build is the default" FORCE)
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
option(TT_METAL_VERSIM_DISABLED "Disable TT_METAL_VERSIM" ON)
option(ENABLE_TRACY "Enable Tracy Profiling" OFF)

# Default to building everything as a shared lib
if($ENV{TT_METAL_CREATE_STATIC_LIB})
    option(BUILD_SHARED_LIBS "Create shared library" OFF)
else()
    option(BUILD_SHARED_LIBS "Create shared library" ON)
endif()
message(STATUS "Build shared libs: ${BUILD_SHARED_LIBS}")

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

############################################################################################################################
# Constructing interface libs for common compiler flags, header directories, and libraries
#   These interface libs are linked with PUBLIC scope at lowest common target (tt_metal/common) and at tt_metal_libs level
#   in order to propogate to the rest of tt_metal, tt_eager, etc.
############################################################################################################################
add_library(stdlib INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries(stdlib INTERFACE c++ c++abi)
    target_compile_options(stdlib INTERFACE -stdlib=libc++)
else()
    target_link_libraries(stdlib INTERFACE stdc++)
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
target_link_libraries(compiler_flags INTERFACE compiler_warnings stdlib)
target_compile_options(compiler_flags INTERFACE -mavx2 -fPIC -DFMT_HEADER_ONLY -fvisibility-inlines-hidden -fno-lto)

if(TT_METAL_VERSIM_DISABLED)
    target_compile_options(compiler_flags INTERFACE -DTT_METAL_VERSIM_DISABLED)
endif()
if(ENABLE_CODE_TIMERS)
    target_compile_options(compiler_flags INTERFACE -DTT_ENABLE_CODE_TIMERS)
endif()
if(ENABLE_TRACY)
    target_compile_options(compiler_flags INTERFACE -DTRACY_ENABLE -fno-omit-frame-pointer)
    target_link_options(linker_flags INTERFACE -rdynamic)
endif()

add_library(metal_header_directories INTERFACE)
target_include_directories(metal_header_directories INTERFACE ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc)
target_include_directories(metal_header_directories SYSTEM INTERFACE ${reflect_SOURCE_DIR})
foreach(lib ${BoostPackages})
    target_include_directories(metal_header_directories INTERFACE ${Boost${lib}_SOURCE_DIR}/include)
endforeach()

if ("$ENV{ARCH_NAME}" STREQUAL "wormhole_b0")
    target_include_directories(metal_header_directories INTERFACE tt_metal/hw/inc/wormhole
        tt_metal/hw/inc/wormhole/wormhole_b0_defines
        ${UMD_HOME}/device/wormhole
        ${UMD_HOME}/src/firmware/riscv/wormhole
    )
else()
    target_compile_options(compiler_flags INTERFACE -DDISABLE_ISSUE_3487_FIX)
    target_include_directories(metal_header_directories INTERFACE
        tt_metal/hw/inc/$ENV{ARCH_NAME}
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

# Build umd_device
include(${PROJECT_SOURCE_DIR}/cmake/umd_device.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/tt_metal/hw)
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
