# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

add_executable(test_lab_eltwise_binary)
target_sources(
    test_lab_eltwise_binary
    PRIVATE
        ${PROJECT_SOURCE_DIR}/ttnn/examples/lab_eltwise_binary/lab_eltwise_binary.cpp
)

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
