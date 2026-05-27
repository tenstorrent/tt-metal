# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

include(sources.cmake)

add_library(tt_metal_x280_boot STATIC ${X280_BOOT_SRCS})
add_library(TT::Metalium::Experimental::X280Boot ALIAS tt_metal_x280_boot)

target_include_directories(
    tt_metal_x280_boot
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/impl
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        ${PROJECT_SOURCE_DIR}/tt_metal/llrt
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
)

target_link_libraries(
    tt_metal_x280_boot
    PUBLIC
        metal_common_libs
        tt_metal
)
