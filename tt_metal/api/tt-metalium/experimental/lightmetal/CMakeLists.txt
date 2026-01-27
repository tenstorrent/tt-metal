# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Experimental LightMetal API Headers
target_sources(
    tt_metal
    PUBLIC
        FILE_SET api
        FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/lightmetal_binary.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/lightmetal_capture_utils.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/lightmetal_replay.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/lightmetal_api.hpp
)
