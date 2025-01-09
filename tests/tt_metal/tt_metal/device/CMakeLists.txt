set(UNIT_TESTS_DEVICE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_device_cluster_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_device_init_and_teardown.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_device_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_galaxy_cluster_api.cpp
)

add_executable(unit_tests_device ${UNIT_TESTS_DEVICE_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_device)

target_link_libraries(unit_tests_device PUBLIC test_metal_common_libs)
target_include_directories(
    unit_tests_device
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    unit_tests_device
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
