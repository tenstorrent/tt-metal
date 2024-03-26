cmake_minimum_required(VERSION 3.16)
project(tt-metal)

# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # for static library files (.a)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # for shared library files (.so)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) # for executables

# Set default values for variables
set(CONFIG "assert" CACHE STRING "Default configuration")
set(OUT "$ENV{TT_METAL_HOME}/build" CACHE STRING "Build directory")
set(UMD_HOME "$ENV{TT_METAL_HOME}/tt_metal/third_party/umd")

option(ENABLE_PROFILER "Enable profiler" OFF)
option(ENABLE_TRACY "Enable Tracy" OFF)
option(ENABLE_CODE_TIMERS "Enable code timers" OFF)
option(TT_METAL_VERSIM_DISABLED "Disable TT_METAL_VERSIM" ON)
option(TT_METAL_CREATE_STATIC_LIB "Create static library" OFF)


# we are using c++ 17?
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set(COMPILE_OPTIONS ${COMPILE_OPTIONS} --std=c++17 -Werror -MMD -mavx2 -fPIC -Wdelete-non-virtual-dtor -Wreturn-type -Wswitch -Wuninitialized -Wno-unused-parameter -DFMT_HEADER_ONLY)
# -fvisibility-inlines-hidden <- cannot include this for tests or else it'll complain about hidden symbols
# DFMT_HEADER_ONLY must be for every target or else they won't interact with the header only fmt as intended

set(COMMON_LIBS dl z boost_thread boost_filesystem boost_system boost_regex pthread atomic hwloc stdc++)

# Add compiler flags based on configuration -- Can we use CMAKE_BUILD_TYPE instead?
if(${CONFIG} STREQUAL "release")
    set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -O3)
elseif(${CONFIG} STREQUAL "ci")
    set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -O3 -DDEBUG=DEBUG -Wl,--verbose)
elseif(${CONFIG} STREQUAL "assert")
    set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -O3 -g -DDEBUG=DEBUG)
elseif(${CONFIG} STREQUAL "asan")
    set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -O3 -g -DDEBUG=DEBUG -fsanitize=address)
elseif(${CONFIG} STREQUAL "ubsan")
    set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -O3 -g -DDEBUG=DEBUG -fsanitize=undefined)
elseif(${CONFIG} STREQUAL "debug")
    set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -O0 -g -DDEBUG=DEBUG)
else()
    message(FATAL_ERROR "Unknown value for CONFIG: ${CONFIG}")
endif()

if(TT_METAL_VERSIM_DISABLED)
    set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -DTT_METAL_VERSIM_DISABLED)
endif()
if(ENABLE_CODE_TIMERS)
    set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -DTT_ENABLE_CODE_TIMERS)
endif()
if(ENABLE_PROFILER)
    set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -DPROFILER)
endif()
if(ENABLE_TRACY)
    set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -DTRACY_ENABLE -fno-omit-frame-pointer)
    set(COMMON_LIBS "${COMMON_LIBS} tracy")
endif()

set(METAL_COMPILE_OPTIONS ${COMPILE_OPTIONS} -fvisibility-inlines-hidden)

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

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/python_env)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/hw)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_metal)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_eager)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests EXCLUDE_FROM_ALL)

# add_subdirectory($ENV{TT_METAL_HOME}/ttnn)
# add_subdirectory($ENV{TT_METAL_HOME}/tt_metal/python_env)
# if(${TT_METAL_ENV} STREQUAL "dev")
#     add_subdirectory(${TT_METAL_HOME}/infra/git_hooks)
# endif()
