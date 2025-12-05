# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

find_package(Python3 REQUIRED COMPONENTS Development)

set(UDM_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_udm_copy.cpp)

add_executable(udm_tests ${UDM_TEST_SRCS})

target_link_libraries(
    udm_tests
    PUBLIC
        test_common_libs
        tt_metal
        tt_metal_udm
        ttnn
        GTest::gtest
        GTest::gtest_main
        Python3::Python
)

target_include_directories(
    udm_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tests
)

set_target_properties(
    udm_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

gtest_discover_tests(udm_tests)
