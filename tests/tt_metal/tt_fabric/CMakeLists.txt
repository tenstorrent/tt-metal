set(UNIT_TESTS_FABRIC_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_routing_tables.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_cluster_apis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_data_movement/test_basic_fabric_apis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_data_movement/test_basic_1d_fabric.cpp
)

add_executable(fabric_unit_tests ${UNIT_TESTS_FABRIC_SRC})
target_link_libraries(
    fabric_unit_tests
    PRIVATE
        tt_metal
        fabric
        test_common_libs
)

target_include_directories(
    fabric_unit_tests
    PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)

set_target_properties(
    fabric_unit_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric
)

gtest_discover_tests(fabric_unit_tests)
