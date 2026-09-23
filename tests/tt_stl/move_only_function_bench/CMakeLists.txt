# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Standalone evaluation harness for GitHub issue #57444. Deliberately NOT part of the tt-metal
# build: it depends only on google-benchmark and the two candidate headers, so it configures in
# seconds and can be built with each supported compiler in turn. See README.md.

cmake_minimum_required(VERSION 3.20)
project(move_only_function_bench CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

# Pinned to the same version the main build already uses (third_party/CMakeLists.txt).
FetchContent_Declare(benchmark GIT_REPOSITORY https://github.com/google/benchmark GIT_TAG v1.9.1)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

# fu2 publishes release tags.
FetchContent_Declare(function2 GIT_REPOSITORY https://github.com/Naios/function2 GIT_TAG 4.2.5)

# zoo publishes no tags at all, so it can only be pinned by commit.
FetchContent_Declare(
    zoo
    GIT_REPOSITORY https://github.com/thecppzoo/zoo
    GIT_TAG 7fb5eed6331651e9d94c0f7e9017f2d56ba4b993
)

FetchContent_MakeAvailable(
    benchmark
    function2
)
FetchContent_GetProperties(zoo)
if(NOT zoo_POPULATED)
    FetchContent_Populate(zoo)
endif()

# zoo ships sources we do not want; only its headers are needed.
add_library(zoo_headers INTERFACE)
target_include_directories(zoo_headers INTERFACE ${zoo_SOURCE_DIR}/inc)

# Establishes std::function's real inline buffer for this configuration; the benchmark pins the
# candidates to whatever it reports.
add_executable(sbo_probe sbo_probe.cpp)

add_executable(bench bench.cpp)
target_link_libraries(
    bench
    PRIVATE
        benchmark::benchmark
        function2
        zoo_headers
)

# Item 6: one object file per candidate, for size comparison.
foreach(CANDIDATE STD ZOO FU2)
    string(TOLOWER ${CANDIDATE} CANDIDATE_LOWER)
    add_library(codegen_${CANDIDATE_LOWER} OBJECT codegen_tu.cpp)
    target_compile_definitions(codegen_${CANDIDATE_LOWER} PRIVATE CANDIDATE_${CANDIDATE})
    target_link_libraries(
        codegen_${CANDIDATE_LOWER}
        PRIVATE
            function2
            zoo_headers
    )

    # Item 7: compile-time probe. Built as its own target so it can be timed in isolation.
    add_library(compile_time_${CANDIDATE_LOWER} OBJECT compile_time_tu.cpp)
    target_compile_definitions(compile_time_${CANDIDATE_LOWER} PRIVATE CANDIDATE_${CANDIDATE})
    target_link_libraries(
        compile_time_${CANDIDATE_LOWER}
        PRIVATE
            function2
            zoo_headers
    )
endforeach()
