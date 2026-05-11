# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(planar_concat LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Discover the active Python interpreter.  Override with `-DPython_EXECUTABLE=...`
# at configure time if the default `python3` on PATH isn't what you want.
find_package(
    Python
    3.10
    REQUIRED
    COMPONENTS
        Interpreter
        Development.Module
)

# Find nanobind via the active Python.  Build script can also set nanobind_DIR
# explicitly to skip this lookup.
if(NOT DEFINED nanobind_DIR)
    execute_process(
        COMMAND
            "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
        OUTPUT_VARIABLE nanobind_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _nanobind_lookup_rc
    )
    if(NOT _nanobind_lookup_rc EQUAL 0 OR NOT nanobind_DIR)
        message(
            FATAL_ERROR
            "Could not locate nanobind via ${Python_EXECUTABLE}. "
            "Run `${Python_EXECUTABLE} -m pip install nanobind` and re-configure."
        )
    endif()
endif()
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
    _planar_concat
    NB_STATIC
    bindings.cpp
    planar_concat.cpp
    transpose_avx2.cpp
)

# Compile flags.  -march=x86-64-v3 enables AVX2 + FMA + BMI2 (matches the
# main tt-metal build's target microarchitecture level).  AVX-512 is NOT
# guaranteed by v3 — the implementation here is AVX2-only.
target_compile_options(
    _planar_concat
    PRIVATE
        -O3
        -march=x86-64-v3
        -fno-plt
        -fvisibility=hidden
        -Wall
        -Wextra
        -Wno-unused-parameter
)

# Place the .so directly in the source dir's `build/` for easy import via the
# Python loader stub.  CMake's default puts it in CMAKE_BINARY_DIR which is
# already that location when invoked via `build.sh`.
set_target_properties(
    _planar_concat
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}"
)
