# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../)

# Common variables
set(COMMON_LIBRARIES
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    ttnncpp
    tt_metal
    pthread
)

set(COMMON_COMPILE_OPTIONS
    -Wall
    -g
    -O0
    -Wno-unused-parameter
    -Wno-error=unused-parameter
    -Wno-error
)

set(HELPER_FUNCS ../helper_funcs.cpp)

# Function to create a test executable with common settings
function(add_deit_test TARGET_NAME MAIN_SOURCE)
    # Get additional source files from ARGN (remaining arguments)
    set(ADDITIONAL_SOURCES ${ARGN})

    # Add executable
    add_executable(
        ${TARGET_NAME}
        ${MAIN_SOURCE}
        ${ADDITIONAL_SOURCES}
        ${HELPER_FUNCS}
    )

    # Link libraries
    target_link_libraries(${TARGET_NAME} ${COMMON_LIBRARIES})

    # Set C++ standard
    set_target_properties(
        ${TARGET_NAME}
        PROPERTIES
            CXX_STANDARD
                17
            CXX_STANDARD_REQUIRED
                ON
    )

    # Set compiler options
    target_compile_options(${TARGET_NAME} PRIVATE ${COMMON_COMPILE_OPTIONS})
endfunction()

# Define source file paths
set(TT_CPP_DIR ../tt_cpp)
set(SELF_ATTENTION_SOURCES
    ${TT_CPP_DIR}/deit_self_attention.cpp
    ${TT_CPP_DIR}/deit_self_output.cpp
)
set(ATTENTION_SOURCES
    ${TT_CPP_DIR}/deit_attention.cpp
    ${SELF_ATTENTION_SOURCES}
)
set(LAYER_SOURCES
    ${TT_CPP_DIR}/deit_layer.cpp
    ${ATTENTION_SOURCES}
    ${TT_CPP_DIR}/deit_intermediate.cpp
    ${TT_CPP_DIR}/deit_output.cpp
)
set(ENCODER_SOURCES
    ${TT_CPP_DIR}/deit_encoder.cpp
    ${LAYER_SOURCES}
)
set(EMBEDDINGS_SOURCES
    ${TT_CPP_DIR}/deit_embeddings.cpp
    ${TT_CPP_DIR}/deit_patch_embeddings.cpp
)
set(MODEL_SOURCES
    ${TT_CPP_DIR}/deit_model.cpp
    ${TT_CPP_DIR}/deit_for_image_classification.cpp
    ${ENCODER_SOURCES}
    ${EMBEDDINGS_SOURCES}
    ${TT_CPP_DIR}/deit_pooler.cpp
)

# Create all test executables
add_deit_test(test_deit_self_attention test_deit_self_attention.cpp ${SELF_ATTENTION_SOURCES})
add_deit_test(test_deit_self_output test_deit_self_output.cpp ${TT_CPP_DIR}/deit_self_output.cpp)
add_deit_test(test_deit_attention test_deit_attention.cpp ${ATTENTION_SOURCES})
add_deit_test(test_deit_patch_embeddings test_deit_patch_embeddings.cpp ${TT_CPP_DIR}/deit_patch_embeddings.cpp)
add_deit_test(test_deit_embeddings test_deit_embeddings.cpp ${EMBEDDINGS_SOURCES})
add_deit_test(test_deit_intermediate test_deit_intermediate.cpp ${TT_CPP_DIR}/deit_intermediate.cpp)
add_deit_test(test_deit_output test_deit_output.cpp ${TT_CPP_DIR}/deit_output.cpp)
add_deit_test(test_deit_layer test_deit_layer.cpp ${LAYER_SOURCES})
add_deit_test(test_deit_encoder test_deit_encoder.cpp ${ENCODER_SOURCES})
add_deit_test(test_deit_model test_deit_model.cpp ${MODEL_SOURCES})
add_deit_test(test_deit_for_image_classification test_deit_for_image_classification.cpp ${MODEL_SOURCES})
add_deit_test(test_deit_for_image_classification_with_teacher test_deit_for_image_classification_with_teacher.cpp
    ${TT_CPP_DIR}/deit_for_image_classification_with_teacher.cpp ${MODEL_SOURCES}
)

# Add simple device test
add_deit_test(test_device test_device.cpp)
