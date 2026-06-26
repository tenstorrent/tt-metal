# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)
project(programming_examples_generic_lut_activation)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# ============================================================================
# TEST: Separate .hpp architecture for rational approximations
# This tests if JIT compiler can handle #include directives for LUT headers
# ============================================================================

add_executable(programming_examples_generic_lut_activation_embedded_test_rational)
target_sources(programming_examples_generic_lut_activation_embedded_test_rational PRIVATE generic_lut_activation.cpp)
target_compile_definitions(
    programming_examples_generic_lut_activation_embedded_test_rational
    PRIVATE
        KERNEL_VARIANT="piecewise_rational_embedded/test_rational_2d2_4_test"
)
target_link_libraries(programming_examples_generic_lut_activation_embedded_test_rational PUBLIC TT::Metalium)
target_include_directories(
    programming_examples_generic_lut_activation_embedded_test_rational
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../generic_lut_activation
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/compute
)

# Test target for dual-eval: sigmoid p9 s16 uniform (uses piecewise_generic.cpp kernel)
add_executable(programming_examples_generic_lut_activation_embedded_test_dual_eval)
target_sources(programming_examples_generic_lut_activation_embedded_test_dual_eval PRIVATE generic_lut_activation.cpp)
target_compile_definitions(
    programming_examples_generic_lut_activation_embedded_test_dual_eval
    PRIVATE
        KERNEL_VARIANT="piecewise_p9_embedded/sigmoid_p9_s16_uniform"
        OVERRIDE_KERNEL_PREFIX="tt_metal/programming_examples/"
)
target_link_libraries(programming_examples_generic_lut_activation_embedded_test_dual_eval PUBLIC TT::Metalium)
target_include_directories(
    programming_examples_generic_lut_activation_embedded_test_dual_eval
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# =============================================================================
# ADHOC TARGETS - For dynamic kernel generation at sweep time
# =============================================================================
# These targets are used by sweep scripts with the adhoc workflow:
#   1. run_csv.sh generates kernels/compute/adhoc/adhoc.cpp
#   2. ninja rebuilds just that target (fast incremental build)
#   3. Binary is executed to collect metrics
#
# This eliminates the need for 500+ pre-generated embedded kernels.
# Use one checkout per concurrent worker because adhoc.cpp is checkout-local.

# Primary adhoc target (backward compatible with run_csv.sh)
add_executable(programming_examples_generic_lut_activation_embedded_adhoc)
target_sources(programming_examples_generic_lut_activation_embedded_adhoc PRIVATE generic_lut_activation.cpp)
target_compile_definitions(
    programming_examples_generic_lut_activation_embedded_adhoc
    PRIVATE
        KERNEL_VARIANT="adhoc/adhoc"
)
target_link_libraries(programming_examples_generic_lut_activation_embedded_adhoc PUBLIC TT::Metalium)
target_include_directories(
    programming_examples_generic_lut_activation_embedded_adhoc
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../generic_lut_activation
)

# Parallel adhoc targets for concurrent sweep execution (adhoc0, adhoc1, adhoc2, adhoc3)
foreach(ADHOC_IDX RANGE 0 3)
    set(TARGET_NAME "programming_examples_generic_lut_activation_embedded_adhoc${ADHOC_IDX}")
    add_executable(${TARGET_NAME})
    target_sources(${TARGET_NAME} PRIVATE generic_lut_activation.cpp)
    target_compile_definitions(${TARGET_NAME} PRIVATE KERNEL_VARIANT="adhoc/adhoc${ADHOC_IDX}")
    target_link_libraries(${TARGET_NAME} PUBLIC TT::Metalium)
    target_include_directories(
        ${TARGET_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../generic_lut_activation
    )
endforeach()
