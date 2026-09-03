# Test utilities and small focused tests that live alongside shared test headers.

add_library(tt_metal_test_utils_gha_annotation INTERFACE)
add_library(TT::Metalium::Test::Utils::GhaAnnotation ALIAS tt_metal_test_utils_gha_annotation)
target_include_directories(tt_metal_test_utils_gha_annotation INTERFACE "${PROJECT_SOURCE_DIR}/tests")
target_link_libraries(tt_metal_test_utils_gha_annotation INTERFACE TT::STL)

add_executable(unit_tests_gha_annotation test_gha_annotation.cpp)
target_link_libraries(
    unit_tests_gha_annotation
    PRIVATE
        tt_metal_test_utils_gha_annotation
        gmock
        gtest
        gtest_main
)
# NOTE: deliberately metal_common_pch, not metal_test_pch. metal_test_pch is compiled with
# TT::Metalium's usage requirements (TRACY_IMPORTS, the tt-metalium/ include dirs) and its
# contents #include <tt-metalium/...>. REUSE_FROM does not propagate those to the consumer, so a
# target that doesn't link Metalium can neither validate the .gch (-Winvalid-pch) nor fall back to
# parsing the header textually. This test links only TT::STL + gtest and includes no Metalium
# headers; keep it on the common PCH rather than linking Metalium just to satisfy the PCH.
target_precompile_headers(unit_tests_gha_annotation REUSE_FROM metal_common_pch)
set_target_properties(
    unit_tests_gha_annotation
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

add_dependencies(metal_tests unit_tests_gha_annotation)
