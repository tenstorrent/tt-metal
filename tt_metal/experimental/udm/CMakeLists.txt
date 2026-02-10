# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_library(tt_metal_udm STATIC ${UDM_SRCS})

target_include_directories(tt_metal_udm PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(
    tt_metal_udm
    PUBLIC
        metal_common_libs
        tt_metal
)

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include/tt_metal FILES_MATCHING PATTERN "*.hpp")
