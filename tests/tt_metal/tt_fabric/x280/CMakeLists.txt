# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_x280_boot test_x280_boot.cpp)

target_link_libraries(
    test_x280_boot
    PRIVATE
        tt_metal
        tt_metal_x280_boot
        gtest
        gtest_main
        test_common_libs
)

add_dependencies(test_x280_boot tt_metal)

target_include_directories(
    test_x280_boot
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
)

set_target_properties(
    test_x280_boot
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric
)
