cmake_minimum_required(VERSION 3.18..3.30)
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
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG=DEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DDEBUG=DEBUG")
set(CMAKE_CXX_FLAGS_CI "-O3 -DDEBUG=DEBUG")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

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

include(cmake/dependencies.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 3rd party projects
if(CMAKE_VERSION VERSION_LESS 3.25)
    # FIXME(14681): `SYSTEM` was introduced in v3.25; remove this when we can require v3.25
    add_subdirectory(3rd_party EXCLUDE_FROM_ALL)
else()
    add_subdirectory(3rd_party EXCLUDE_FROM_ALL SYSTEM)
endif()
# ttml projects

add_subdirectory(sources)
include(CTest)
enable_testing()
add_subdirectory(tests)
