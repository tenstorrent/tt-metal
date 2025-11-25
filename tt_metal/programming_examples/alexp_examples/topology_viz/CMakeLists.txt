# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22...3.30)

project(TopologyVisualization)

# Find TT-Metalium package
if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Topology Visualizer (requires TT-Metal)
# ============================================================================
add_executable(topology_viz topology_viz.cpp)

target_link_libraries(
    topology_viz
    PUBLIC
        TT::Metalium
    PRIVATE
        pthread
        stdc++fs
)

# Disable specific warnings that come from TT-Metal headers
target_compile_options(
    topology_viz
    PRIVATE
        -Wno-error=unused-parameter
        -Wno-unused-parameter
)

# ============================================================================
# Installation
# ============================================================================
install(TARGETS topology_viz RUNTIME DESTINATION bin)

# ============================================================================
# Display build information
# ============================================================================
message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
message(STATUS "  Topology Visualization Tool")
message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
message(STATUS "  Target:")
message(STATUS "    • topology_viz    (Physical topology & ethernet link status)")
message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
