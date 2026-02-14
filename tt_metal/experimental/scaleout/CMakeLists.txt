# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set(SCALEOUT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/blitz_decode_pipeline.cpp)

add_library(tt_metal_scaleout OBJECT ${SCALEOUT_SRCS})

target_include_directories(
    tt_metal_scaleout
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
)

target_link_libraries(
    tt_metal_scaleout
    PRIVATE
        TT::Metalium::Common
        TT::Metalium::HostDevCommon
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include/tt_metal FILES_MATCHING PATTERN "*.hpp")
