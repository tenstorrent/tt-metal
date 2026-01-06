# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

# TT-SMI: Tenstorrent System Management Interface
# Python UI with C++ backend for device monitoring

cmake_minimum_required(VERSION 3.16)

# Option to build Python bindings
option(BUILD_TT_SMI_PYTHON "Build TT-SMI Python bindings" ON)

# C++ backend library (standalone, no Python dependency)
set(TT_SMI_BACKEND_SOURCES tt_smi_backend.cpp)

add_library(tt_smi_backend STATIC ${TT_SMI_BACKEND_SOURCES})

target_include_directories(
    tt_smi_backend
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
)

# Link against UMD for telemetry
target_link_libraries(
    tt_smi_backend
    PUBLIC
        device # UMD device library
)

target_compile_options(
    tt_smi_backend
    PRIVATE
        -std=c++20
        -O3
)

# Python bindings (optional, requires pybind11)
if(BUILD_TT_SMI_PYTHON)
    find_package(pybind11 QUIET)

    if(pybind11_FOUND)
        pybind11_add_module(tt_smi_native
            bindings/native.cpp
        )

        target_link_libraries(tt_smi_native PRIVATE tt_smi_backend)

        target_compile_options(
            tt_smi_native
            PRIVATE
                -std=c++20
                -O3
        )

        # Install Python module to build directory
        set_target_properties(
            tt_smi_native
            PROPERTIES
                OUTPUT_NAME
                    "native"
                LIBRARY_OUTPUT_DIRECTORY
                    "${CMAKE_CURRENT_BINARY_DIR}/bindings"
        )

        message(STATUS "TT-SMI Python bindings enabled")
    else()
        message(WARNING "pybind11 not found - Python bindings disabled. Install with: pip install pybind11")
    endif()
endif()

# Standalone CLI tool (uses C++ backend directly)
add_executable(
    tt_smi_cli
    ../tt_smi.cpp # Original CLI implementation
)

target_link_libraries(tt_smi_cli PRIVATE tt_smi_backend)

target_compile_options(
    tt_smi_cli
    PRIVATE
        -std=c++20
        -O3
)

# Installation
install(TARGETS tt_smi_backend LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

install(FILES tt_smi_backend.hpp DESTINATION include/tt_smi)

if(TARGET tt_smi_native)
    install(TARGETS tt_smi_native LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/python/tt_smi/bindings)
endif()

# Python package installation (optional, via pip)
# To install Python UI:
#   cd memory_utilization_monitor
#   pip install -e .
