# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

add_executable(unit_tests_eltwise_helpers)
target_sources(unit_tests_eltwise_helpers PRIVATE test_eltwise_helpers.cpp)

target_link_libraries(
    unit_tests_eltwise_helpers
    PRIVATE
        test_common_libs
        TTNN::CPP
)

target_include_directories(
    unit_tests_eltwise_helpers
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tests
)

tt_reuse_precompile_headers(unit_tests_eltwise_helpers TTNN::PCH)
