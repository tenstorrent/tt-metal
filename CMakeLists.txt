cmake_minimum_required(VERSION 3.16)
project(tt-metal)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG=DEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DDEBUG=DEBUG")
set(CMAKE_CXX_FLAGS_CI "-O3 -DDEBUG=DEBUG")   # idk if this is the right way to add a custom build type, also removed --verbose linker option

# Set default values for variables
set(CONFIG "assert" CACHE STRING "Default configuration")
set(OUT "$ENV{TT_METAL_HOME}/build" CACHE STRING "Build directory")
set(UMD_HOME "$ENV{TT_METAL_HOME}/tt_metal/third_party/umd")

option(ENABLE_PROFILER "Enable profiler" OFF)
option(ENABLE_TRACY "Enable Tracy" OFF)
option(ENABLE_CODE_TIMERS "Enable code timers" OFF)
option(TT_METAL_VERSIM_DISABLED "Disable TT_METAL_VERSIM" ON)
option(TT_METAL_CREATE_STATIC_LIB "Create static library" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COMMON_LIBS dl z boost_thread boost_filesystem boost_system boost_regex pthread atomic hwloc stdc++)

# note on flags:
#   -fvisibility-inlines-hidden <- cannot include this for tests or else it'll complain about hidden symbols... not true anymore?
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
    set(COMMON_LIBS "${COMMON_LIBS} tracy")
endif()

# Build umd_device
include(CMakeLisTs.txt)

include_directories(tt_metal/hw/inc)
if ("$ENV{ARCH_NAME}" STREQUAL "wormhole_b0")
    include_directories(tt_metal/hw/inc/wormhole
                        tt_metal/hw/inc/wormhole/wormhole_b0_defines
                        ${UMD_HOME}/src/firmware/riscv/wormhole)
else()
    include_directories(tt_metal/hw/inc/$ENV{ARCH_NAME}
                        ${UMD_HOME}/src/firmware/riscv/$ENV{ARCH_NAME})
endif()

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
