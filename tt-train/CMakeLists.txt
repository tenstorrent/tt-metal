cmake_minimum_required(VERSION 3.18...4.2)
include(cmake/compilers.cmake)

# When building as a standalone top-level project, inherit the compiler from
# tt-metal's CMakeCache.txt for ABI compatibility. This is necessary because
# build systems like scikit-build-core (used by "pip install -e") set CC/CXX
# to generic values that don't match tt-metal's compiler.
#
# To use a different compiler, disable inheritance:
#   cmake -DTT_TRAIN_INHERIT_COMPILER=OFF -DCMAKE_C_COMPILER=gcc-12 ...
option(TT_TRAIN_INHERIT_COMPILER "Inherit compiler from tt-metal's CMakeCache.txt" ON)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND TT_TRAIN_INHERIT_COMPILER)
    inherit_compiler_from_tt_metal()
endif()

project(
    ml-framework-cpp
    LANGUAGES
        C
        CXX
)
CHECK_COMPILERS()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG=DEBUG -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DDEBUG=DEBUG -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_CI "-O3 -DDEBUG=DEBUG")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

option(ENABLE_LIBCXX "Enable using libc++" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND ENABLE_LIBCXX)
    find_library(LIBC++ c++)
    find_library(LIBC++ABI c++abi)
    if(NOT LIBC++ OR NOT LIBC++ABI)
        message(
            FATAL_ERROR
            "libc++ or libc++abi not found. Make sure you have libc++ and libc++abi installed and in your PATH"
        )
    endif()
    # making it global settings for now
    add_compile_options(-stdlib=libc++)
    add_link_options(-stdlib=libc++)
endif()

message(STATUS "c++ Standard: ${CMAKE_CXX_STANDARD}")

# Add cmake directory to module path for flatbuffers helper
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(cmake/dependencies.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ttnn header files are not clean against narrowing conversion checks
# therefore tt-train likely is not clean against them
# must disable checks to build with tt-metal as super project
add_compile_options("$<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-narrowing>")
add_compile_options("$<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>")

# ttml projects

add_subdirectory(sources)
include(CTest)
enable_testing()
add_subdirectory(tests)
