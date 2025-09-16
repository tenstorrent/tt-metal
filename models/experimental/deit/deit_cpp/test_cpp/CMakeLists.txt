# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../)

# Add executable with all required source files
add_executable(test_deit_self_attention 
    test_deit_self_attention.cpp
    ../tt_cpp/deit_self_attention.cpp
    ../tt_cpp/deit_self_output.cpp
    ../helper_funcs.cpp
)

# Link libraries - use the same torch libraries as parent
target_link_libraries(test_deit_self_attention
    ${TORCH_LIBRARIES}
    ttnncpp
    tt_metal
    pthread
)

# Set C++ standard
set_target_properties(test_deit_self_attention PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compiler options for debugging
target_compile_options(test_deit_self_attention PRIVATE
    -Wall
    -g
    -O0
    -Wno-unused-parameter
    -Wno-error=unused-parameter
    -Wno-error
)

# Add executable for test_deit_self_output
add_executable(test_deit_self_output 
    test_deit_self_output.cpp
    ../tt_cpp/deit_self_output.cpp
    ../helper_funcs.cpp
)

# Link libraries for test_deit_self_output
target_link_libraries(test_deit_self_output
    ${TORCH_LIBRARIES}
    ttnncpp
    tt_metal
    pthread
)

# Set C++ standard for test_deit_self_output
set_target_properties(test_deit_self_output PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compiler options for debugging
target_compile_options(test_deit_self_output PRIVATE
    -Wall
    -g
    -O0
    -Wno-unused-parameter
    -Wno-error=unused-parameter
    -Wno-error
)

# Add executable for test_deit_attention
add_executable(test_deit_attention 
    test_deit_attention.cpp
    ../tt_cpp/deit_attention.cpp
    ../tt_cpp/deit_self_attention.cpp
    ../tt_cpp/deit_self_output.cpp
    ../helper_funcs.cpp
)

# Link libraries for test_deit_attention
target_link_libraries(test_deit_attention
    ${TORCH_LIBRARIES}
    ttnncpp
    tt_metal
    pthread
)

# Set C++ standard for test_deit_attention
set_target_properties(test_deit_attention PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compiler options for debugging
target_compile_options(test_deit_attention PRIVATE
    -Wall
    -g
    -O0
    -Wno-unused-parameter
    -Wno-error=unused-parameter
    -Wno-error
)

# Add executable for test_deit_patch_embeddings
add_executable(test_deit_patch_embeddings 
    test_deit_patch_embeddings.cpp
    ../tt_cpp/deit_patch_embeddings.cpp
    ../helper_funcs.cpp
)

# Link libraries for test_deit_patch_embeddings
target_link_libraries(test_deit_patch_embeddings
    ${TORCH_LIBRARIES}
    ttnncpp
    tt_metal
    pthread
)

# Set C++ standard for test_deit_patch_embeddings
set_target_properties(test_deit_patch_embeddings PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compiler options for debugging
target_compile_options(test_deit_patch_embeddings PRIVATE
    -Wall
    -g
    -O0
    -Wno-unused-parameter
    -Wno-error=unused-parameter
    -Wno-error
)

# Add executable for test_deit_embeddings
add_executable(test_deit_embeddings 
    test_deit_embeddings.cpp
    ../tt_cpp/deit_embeddings.cpp
    ../tt_cpp/deit_patch_embeddings.cpp
    ../helper_funcs.cpp
)

# Link libraries for test_deit_embeddings
target_link_libraries(test_deit_embeddings
    ${TORCH_LIBRARIES}
    ttnncpp
    tt_metal
    pthread
)

# Set C++ standard for test_deit_embeddings
set_target_properties(test_deit_embeddings PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compiler options for debugging
target_compile_options(test_deit_embeddings PRIVATE
    -Wall
    -g
    -O0
    -Wno-unused-parameter
    -Wno-error=unused-parameter
    -Wno-error
)

# Add executable for test_deit_intermediate
add_executable(test_deit_intermediate 
    test_deit_intermediate.cpp
    ../tt_cpp/deit_intermediate.cpp
    ../helper_funcs.cpp
)

# Link libraries for test_deit_intermediate
target_link_libraries(test_deit_intermediate
    ${TORCH_LIBRARIES}
    ttnncpp
    tt_metal
    pthread
)

# Set C++ standard for test_deit_intermediate
set_target_properties(test_deit_intermediate PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compiler options for debugging
target_compile_options(test_deit_intermediate PRIVATE
    -Wall
    -g
    -O0
    -Wno-unused-parameter
    -Wno-error=unused-parameter
    -Wno-error
)

# Add executable for test_deit_output
add_executable(test_deit_output 
    test_deit_output.cpp
    ../tt_cpp/deit_output.cpp
    ../helper_funcs.cpp
)

# Link libraries for test_deit_output
target_link_libraries(test_deit_output
    ${TORCH_LIBRARIES}
    ttnncpp
    tt_metal
    pthread
)

# Set C++ standard for test_deit_output
set_target_properties(test_deit_output PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compiler options for debugging
target_compile_options(test_deit_output PRIVATE
    -Wall
    -g
    -O0
    -Wno-unused-parameter
    -Wno-error=unused-parameter
    -Wno-error
)

# Add executable for test_deit_layer
add_executable(test_deit_layer 
    test_deit_layer.cpp
    ../tt_cpp/deit_layer.cpp
    ../tt_cpp/deit_attention.cpp
    ../tt_cpp/deit_self_attention.cpp
    ../tt_cpp/deit_self_output.cpp
    ../tt_cpp/deit_intermediate.cpp
    ../tt_cpp/deit_output.cpp
    ../helper_funcs.cpp
)

# Link libraries for test_deit_layer
target_link_libraries(test_deit_layer
    ${TORCH_LIBRARIES}
    ttnncpp
    tt_metal
    pthread
)

# Set C++ standard for test_deit_layer
set_target_properties(test_deit_layer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compiler options for debugging
target_compile_options(test_deit_layer PRIVATE
    -Wall
    -g
    -O0
    -Wno-unused-parameter
    -Wno-error=unused-parameter
    -Wno-error
)

# Add executable for test_deit_encoder
add_executable(test_deit_encoder 
    test_deit_encoder.cpp
    ../tt_cpp/deit_encoder.cpp
    ../tt_cpp/deit_layer.cpp
    ../tt_cpp/deit_attention.cpp
    ../tt_cpp/deit_self_attention.cpp
    ../tt_cpp/deit_self_output.cpp
    ../tt_cpp/deit_intermediate.cpp
    ../tt_cpp/deit_output.cpp
    ../helper_funcs.cpp
)

# Link libraries for test_deit_encoder
target_link_libraries(test_deit_encoder
    ${TORCH_LIBRARIES}
    ttnncpp
    tt_metal
    pthread
)

# Set C++ standard for test_deit_encoder
set_target_properties(test_deit_encoder PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compiler options for debugging
target_compile_options(test_deit_encoder PRIVATE
    -Wall
    -g
    -O0
    -Wno-unused-parameter
    -Wno-error=unused-parameter
    -Wno-error
)