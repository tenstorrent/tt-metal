# Smoke tests
add_library(unit_tests_device_smoke OBJECT)
add_library(TT::Metalium::Test::Device::Smoke ALIAS unit_tests_device_smoke)
TT_ENABLE_UNITY_BUILD(unit_tests_device_smoke)
target_sources(
    unit_tests_device_smoke
    PRIVATE
        test_device_init_and_teardown.cpp
        test_device_cluster_api.cpp
        test_device_pool.cpp
        test_device.cpp
        test_simulator_device.cpp
        test_galaxy_cluster_api.cpp
        test_mock_device_api.cpp
)
target_include_directories(
    unit_tests_device_smoke
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(unit_tests_device_smoke PRIVATE test_metal_common_libs)

# Remaining tests
add_executable(unit_tests_device)
set_target_properties(
    unit_tests_device
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
target_link_libraries(unit_tests_device PRIVATE TT::Metalium::Test::Device::Smoke)
