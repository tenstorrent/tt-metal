# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

include(sources.cmake)

add_library(tt_metal_noc_estimator OBJECT ${NOC_ESTIMATOR_SRCS})

target_include_directories(
    tt_metal_noc_estimator
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/impl
)

target_link_libraries(tt_metal_noc_estimator PRIVATE yaml-cpp::yaml-cpp)
