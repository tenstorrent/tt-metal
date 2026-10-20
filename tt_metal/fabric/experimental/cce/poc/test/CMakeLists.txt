# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Native host test of the CCE POC kernel against host-side fakes of the metal
# names it calls. Standalone (not part of metal's build):
#   cmake -S . -B build -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/g++
#   cmake --build build && ctest --test-dir build
# Built as PIE: the fakes map memory at the CCE's low fixed addresses, where a
# non-PIE executable would be loaded.

cmake_minimum_required(VERSION 3.20)
project(cce_poc_kernel_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TT_METAL_HW_INC ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../hw/inc)

set(POC_PATTERN_WORDS 64)

# Metal's KERNEL_COMPILE_TIME_ARG_MAP header, as its JIT writes it.
set(CT_ARG_HEADER ${CMAKE_CURRENT_BINARY_DIR}/named_ct_arg_map_generated.h)
file(WRITE ${CT_ARG_HEADER}
    "// AUTO-GENERATED -- do not edit.\n#pragma once\n\n#define KERNEL_COMPILE_TIME_ARG_MAP {\"pattern_words\",${POC_PATTERN_WORDS}}\n")

add_executable(poc_kernel_test poc_kernel_test.cpp ../poc_kernel.cpp)
target_include_directories(poc_kernel_test PRIVATE fakes)
target_compile_definitions(poc_kernel_test PRIVATE POC_PATTERN_WORDS=${POC_PATTERN_WORDS})
target_include_directories(poc_kernel_test SYSTEM PRIVATE ${TT_METAL_HW_INC})
target_compile_options(poc_kernel_test PRIVATE -Wall -Wextra -Werror -include ${CT_ARG_HEADER})
target_link_options(poc_kernel_test PRIVATE -pie)
target_link_libraries(poc_kernel_test PRIVATE pthread)

enable_testing()
add_test(NAME poc_kernel_test COMMAND poc_kernel_test)
