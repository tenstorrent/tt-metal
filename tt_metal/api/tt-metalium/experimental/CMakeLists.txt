# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Experimental API single headers
set(TT_METAL_EXPERIMENTAL_SINGLE_HEADERS
    api/tt-metalium/experimental/inspector.hpp
    api/tt-metalium/experimental/profiler.hpp
    api/tt-metalium/experimental/pinned_memory.hpp
)

# Load subdirectory experimental headers
add_subdirectory(fabric)
add_subdirectory(lightmetal)
add_subdirectory(sockets)

# Combine all experimental headers
set(TT_METAL_EXPERIMENTAL_API
    ${TT_METAL_EXPERIMENTAL_SINGLE_HEADERS}
    ${TT_METAL_EXPERIMENTAL_FABRIC_API}
    ${TT_METAL_EXPERIMENTAL_LIGHTMETAL_API}
    ${TT_METAL_EXPERIMENTAL_SOCKETS_API}
    PARENT_SCOPE
)
