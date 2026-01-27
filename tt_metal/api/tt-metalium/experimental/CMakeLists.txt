# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Experimental API single headers
target_sources(
    tt_metal
    PUBLIC
        FILE_SET api
        FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/inspector.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/profiler.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/pinned_memory.hpp
)

# Load subdirectory experimental headers
add_subdirectory(fabric)
add_subdirectory(lightmetal)
add_subdirectory(sockets)
