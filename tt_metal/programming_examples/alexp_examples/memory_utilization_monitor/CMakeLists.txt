# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22...3.30)

project(AllocationMonitorTools)

# Find TT-Metalium package
if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Allocation Server (requires TT-Metal)
# ============================================================================
add_executable(allocation_server_poc allocation_server_poc.cpp)

target_link_libraries(allocation_server_poc PUBLIC TT::Metalium)

# Disable specific warnings that come from TT-Metal headers
target_compile_options(
    allocation_server_poc
    PRIVATE
        -Wno-error=unused-parameter
        -Wno-unused-parameter
)

# ============================================================================
# Allocation Monitor Client (standalone, no TT-Metal needed)
# ============================================================================
add_executable(allocation_monitor_client allocation_monitor_client.cpp)

target_link_libraries(allocation_monitor_client PRIVATE pthread)

# ============================================================================
# Optional: Allocation Client Demo
# ============================================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/allocation_client_demo.cpp")
    add_executable(allocation_client_demo allocation_client_demo.cpp)

    target_link_libraries(allocation_client_demo PRIVATE pthread)
endif()

# ============================================================================
# Installation
# ============================================================================
install(
    TARGETS
        allocation_server_poc
        allocation_monitor_client
    RUNTIME
        DESTINATION bin
)

# ============================================================================
# Display build information
# ============================================================================
message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
message(STATUS "  Allocation Monitoring Tools")
message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
message(STATUS "  Targets:")
message(STATUS "    • allocation_server_poc    (with TT::Metalium)")
message(STATUS "    • allocation_monitor_client (standalone)")
message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
