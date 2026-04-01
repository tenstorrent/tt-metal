# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

include(sources.cmake)

add_library(tt_metal_udm STATIC ${UDM_SRCS})

target_include_directories(
    tt_metal_udm
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/impl
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        ${PROJECT_SOURCE_DIR}/tt_metal/llrt
)

target_link_libraries(
    tt_metal_udm
    PUBLIC
        metal_common_libs
        tt_metal
)

# Headers are now part of the public API file set in tt_metal/CMakeLists.txt
# at tt_metal/api/tt-metalium/experimental/udm/
