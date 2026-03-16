# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

include(sources.cmake)

add_library(tt_metal_noc_estimator STATIC ${NOC_ESTIMATOR_SRCS})

target_include_directories(
    tt_metal_noc_estimator
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/impl
        ${PROJECT_SOURCE_DIR}/tt_metal/api
)

target_link_libraries(tt_metal_noc_estimator PUBLIC yaml-cpp)
