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
# tt-smi: DEPRECATED - Use tt_smi_umd instead
# ============================================================================
# Commented out - tt_smi.cpp is deprecated in favor of tt_smi_umd
# which has better UMD telemetry support and interactive views
#
# add_executable(tt_smi tt_smi.cpp)
# target_link_libraries(tt_smi PUBLIC TT::Metalium PRIVATE pthread stdc++fs)
# target_compile_options(tt_smi PRIVATE -Wno-error=unused-parameter -Wno-unused-parameter)

# ============================================================================
# tt_smi_umd: Enhanced tt-smi with UMD telemetry and ncurses TUI
# ============================================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tt_smi_umd.cpp")
    # Find ncurses library (like nvtop uses)
    find_package(Curses REQUIRED)

    add_executable(tt_smi_umd tt_smi_umd.cpp)

    target_link_libraries(
        tt_smi_umd
        PUBLIC
            TT::Metalium # TT-Metal provides UMD access
        PRIVATE
            pthread
            stdc++fs
            ${CURSES_LIBRARIES} # ncurses for TUI
    )

    target_include_directories(tt_smi_umd PRIVATE ${CURSES_INCLUDE_DIRS})

    # Disable specific warnings
    target_compile_options(
        tt_smi_umd
        PRIVATE
            -Wno-error=unused-parameter
            -Wno-unused-parameter
    )
endif()

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
set(INSTALL_TARGETS
    allocation_server_poc
    allocation_monitor_client
)

# Add tt_smi_umd to install if it was built
if(TARGET tt_smi_umd)
    list(APPEND INSTALL_TARGETS tt_smi_umd)
endif()

install(TARGETS ${INSTALL_TARGETS} RUNTIME DESTINATION bin)

# ============================================================================
# Display build information
# ============================================================================
message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
message(STATUS "  Allocation Monitoring Tools")
message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
message(STATUS "  Targets:")
message(STATUS "    • allocation_server_poc     (with TT::Metalium)")
message(STATUS "    • allocation_monitor_client (standalone)")
if(TARGET tt_smi_umd)
    message(STATUS "    • tt_smi_umd                (nvtop-style TUI with ncurses, UMD telemetry & real-time charts)")
endif()
message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
