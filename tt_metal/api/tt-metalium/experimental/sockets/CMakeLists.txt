# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Experimental Sockets API Headers
target_sources(
    tt_metal
    PUBLIC
        FILE_SET api
        FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/mesh_socket.hpp
)
