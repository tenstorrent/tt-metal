# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)

# Define the tt_cpp library for DeiT components
set(DEIT_TT_CPP_SOURCES
    deit_attention.cpp
    deit_self_attention.cpp
    deit_self_output.cpp
    helper_funcs.cpp
)

set(DEIT_TT_CPP_HEADERS
    deit_config.h
    deit_attention.h
    deit_self_attention.h
    deit_self_output.h
    helper_funcs.h
)

# Create object library for DeiT TT C++ components
add_library(deit_tt_cpp OBJECT ${DEIT_TT_CPP_SOURCES})

# Find required packages
find_package(Torch REQUIRED)

# Set include directories
target_include_directories(deit_tt_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/ttnn
    ${PROJECT_SOURCE_DIR}/ttnn/cpp
    ${PROJECT_SOURCE_DIR}/tt_metal
    ${PROJECT_SOURCE_DIR}/tt_metal/include
    ${TORCH_INCLUDE_DIRS}
)

# Link required libraries
target_link_libraries(deit_tt_cpp PRIVATE
    ttnn
    tt_metal
    ${TORCH_LIBRARIES}
)

# Set C++ standard
set_target_properties(deit_tt_cpp PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add compile definitions if needed
target_compile_definitions(deit_tt_cpp PRIVATE
    FMT_HEADER_ONLY
)

# Set compiler flags for torch compatibility
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(deit_tt_cpp PRIVATE -Wno-unused-parameter)
endif()

# Install headers
install(FILES ${DEIT_TT_CPP_HEADERS}
    DESTINATION include/models/experimental/deit/deit_cpp/tt_cpp
)