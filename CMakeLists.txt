cmake_minimum_required(VERSION 3.16)
project(tt-metal)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG=DEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DDEBUG=DEBUG")
set(CMAKE_CXX_FLAGS_CI "-O3 -DDEBUG=DEBUG")   # idk if this is the right way to add a custom build type but it works.. also removed --verbose linker option

# Set default values for variables/options
set(CONFIG "assert" CACHE STRING "Default configuration")
set(UMD_HOME "$ENV{TT_METAL_HOME}/tt_metal/third_party/umd")

option(ENABLE_PROFILER "Enable profiler" OFF)
option(ENABLE_TRACY "Enable Tracy" OFF)
option(ENABLE_CODE_TIMERS "Enable code timers" OFF)
option(TT_METAL_VERSIM_DISABLED "Disable TT_METAL_VERSIM" ON)
option(TT_METAL_CREATE_STATIC_LIB "Create static library" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

############################################
# Constructing interface libs for common compiler flags, header directories, and libraries
#   These interface libs are linked with PUBLIC scope at lowest common target (tt_metal/common) and at tt_metal_libs level
#   in order to propogate to the rest of tt_metal, tt_eager and cpp tests.
############################################
add_library(metal_common_libs INTERFACE)
target_link_libraries(metal_common_libs INTERFACE dl z boost_thread boost_filesystem boost_system boost_regex pthread atomic hwloc stdc++)

# note on flags:
#   DFMT_HEADER_ONLY must be for every target or else they won't interact with the header only fmt as intended
add_library(compiler_warnings INTERFACE)
target_compile_options(compiler_warnings INTERFACE -Werror -Wdelete-non-virtual-dtor -Wreturn-type -Wswitch -Wuninitialized -Wno-unused-parameter)

add_library(compiler_flags INTERFACE)
target_link_libraries(compiler_flags INTERFACE compiler_warnings)
target_compile_options(compiler_flags INTERFACE --std=c++17 -MMD -mavx2 -fPIC -DFMT_HEADER_ONLY -fvisibility-inlines-hidden)

if(TT_METAL_VERSIM_DISABLED)
    target_compile_options(compiler_flags INTERFACE -DTT_METAL_VERSIM_DISABLED)
endif()
if(ENABLE_CODE_TIMERS)
    target_compile_options(compiler_flags INTERFACE -DTT_ENABLE_CODE_TIMERS)
endif()
if(ENABLE_PROFILER)
    target_compile_options(compiler_flags INTERFACE -DPROFILER)
endif()
if(ENABLE_TRACY)
    target_compile_options(compiler_flags INTERFACE -DTRACY_ENABLE -fno-omit-frame-pointer)
    target_link_libraries(metal_common_libs INTERFACE tracy)
endif()

add_library(metal_header_directories INTERFACE)
target_include_directories(metal_header_directories INTERFACE tt_metal/hw/inc)
if ("$ENV{ARCH_NAME}" STREQUAL "wormhole_b0")
    target_include_directories(metal_header_directories INTERFACE tt_metal/hw/inc/wormhole
        tt_metal/hw/inc/wormhole/wormhole_b0_defines
        ${UMD_HOME}/device/wormhole
        ${UMD_HOME}/src/firmware/riscv/wormhole
    )
else()
    target_include_directories(metal_header_directories INTERFACE
        tt_metal/hw/inc/$ENV{ARCH_NAME}
        ${UMD_HOME}/device/$ENV{ARCH_NAME}
        ${UMD_HOME}/src/firmware/riscv/$ENV{ARCH_NAME}
    )
endif()

############################################
# Build subdirectories
############################################
# Build umd_device
include(CMakeLists-umd.txt)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/hw)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_metal)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_eager)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests EXCLUDE_FROM_ALL)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/python_env)

# add_subdirectory($ENV{TT_METAL_HOME}/ttnn)
# add_subdirectory($ENV{TT_METAL_HOME}/tt_metal/python_env)
# if(${TT_METAL_ENV} STREQUAL "dev")
#     add_subdirectory(${TT_METAL_HOME}/infra/git_hooks)
# endif()
