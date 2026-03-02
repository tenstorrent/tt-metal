# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

# Smoke tests
add_library(unit_tests_device_smoke OBJECT)
add_library(TT::Metalium::Test::Device::Smoke ALIAS unit_tests_device_smoke)
TT_ENABLE_UNITY_BUILD(unit_tests_device_smoke)
target_precompile_headers(unit_tests_device_smoke REUSE_FROM metal_common_pch)
target_sources(unit_tests_device_smoke PRIVATE ${UNIT_TESTS_DEVICE_SMOKE_SOURCES})
target_include_directories(
    unit_tests_device_smoke
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(unit_tests_device_smoke PRIVATE test_metal_common_libs)

# Non-smoke tests
add_executable(unit_tests_device)
target_sources(unit_tests_device PRIVATE test_release_ownership.cpp)
target_include_directories(
    unit_tests_device
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(
    unit_tests_device
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
target_link_libraries(
    unit_tests_device
    PRIVATE
        TT::Metalium::Test::Device::Smoke
        test_metal_common_libs
)
