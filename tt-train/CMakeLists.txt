cmake_minimum_required(VERSION 3.18...3.30)
include(cmake/compilers.cmake)

if(DEFINED ENV{CMAKE_C_COMPILER} AND DEFINED ENV{CMAKE_CXX_COMPILER})
    message(STATUS "Setting C and C++ compiler from environment variables")
    set(CMAKE_C_COMPILER $ENV{CMAKE_C_COMPILER})
    set(CMAKE_CXX_COMPILER $ENV{CMAKE_CXX_COMPILER})
endif()

if(CMAKE_CXX_COMPILER AND CMAKE_C_COMPILER)
    message(STATUS "Using specified C++ compiler: ${CMAKE_CXX_COMPILER}")
    message(STATUS "Using specified C compiler: ${CMAKE_C_COMPILER}")
else()
    message(STATUS "No C or C++ compiler specified, defaulting to Clang-17")
    find_and_set_clang17()
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

# RPATH settings for tt-train executables
# When built as part of the main project (BUILD_TT_TRAIN=ON), the executables are at:
#   build/tt-train/tests/ttml_tests
# And they need to find shared libraries at:
#   build/lib/libtt_metal.so
#   ttnn/ttnn/_ttnncpp.so (in the SOURCE tree, not build tree)
#
# NOTE: CMAKE_BINARY_DIR is always the top-level build directory (e.g., /work/build/).
# CMAKE_SOURCE_DIR is always the top-level source directory (e.g., /work/).
# The tar artifact is created directly from build/ without cmake --install, so
# BUILD_RPATH is what gets embedded in the binaries (not INSTALL_RPATH).
set(CMAKE_BUILD_RPATH
    "${CMAKE_BINARY_DIR}/lib" # /work/build/lib (for libtt_metal.so)
    "${CMAKE_SOURCE_DIR}/ttnn/ttnn" # /work/ttnn/ttnn (for _ttnncpp.so)
)
set(CMAKE_INSTALL_RPATH
    "$ORIGIN/../lib" # FHS: /usr/libexec/tt-train/tests/../lib = /usr/lib64
    "$ORIGIN/../../lib" # From tests/ -> lib/
    "$ORIGIN/../../../ttnn/ttnn" # Tar artifact: build/tt-train/tests/../../../ttnn/ttnn
)

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
