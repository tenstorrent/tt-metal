# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

add_executable(test_lab_eltwise_binary)

set_target_properties(
    test_lab_eltwise_binary
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${CMAKE_BINARY_DIR}/test/ttnn/lab_examples
)

# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

target_sources(test_lab_eltwise_binary PRIVATE ${TEST_LAB_ELTWISE_BINARY_SOURCES})

target_link_libraries(
    test_lab_eltwise_binary
    PRIVATE
        TT::Metalium
        TTNN::CPP
)

target_include_directories(
    test_lab_eltwise_binary
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tests
)
