# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_library(dfb OBJECT ${DFB_SOURCES})

target_link_libraries(
    dfb
    PRIVATE
        common
        TT::Metalium::HostDevCommon
)
