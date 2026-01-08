# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

add_executable(unit_tests_ttnn_udm)
target_sources(
    unit_tests_ttnn_udm
    PRIVATE
        copy/test_udm_copy.cpp
        eltwise/test_udm_add.cpp
        reduction/sharded/test_udm_reduction.cpp
        reduction/interleaved/test_udm_reduction_interleaved.cpp
)

target_link_libraries(
    unit_tests_ttnn_udm
    PRIVATE
        test_common_libs
        tt_metal_udm
        TTNN::CPP
)

target_include_directories(
    unit_tests_ttnn_udm
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tests
)
