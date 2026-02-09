# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

add_library(dfb OBJECT dataflow_buffer.cpp)

target_link_libraries(
    dfb
    PRIVATE
        common
        TT::Metalium::HostDevCommon
)
