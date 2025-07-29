set(UNIT_TESTS_FABRIC_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_routing_tables.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_control_plane_logical_to_physical.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_intermesh_apis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_multi_host.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_data_movement/test_basic_fabric_apis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_data_movement/test_basic_1d_fabric.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_data_movement/test_basic_fabric_mux.cpp
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

add_library(test_system_health_smoke OBJECT)
add_library(TT::SystemHealth::Smoke ALIAS test_system_health_smoke)

target_sources(test_system_health_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/system_health/test_system_health.cpp)

target_link_libraries(
    test_system_health_smoke
    PRIVATE
        tt_metal
        test_common_libs
)
add_executable(test_system_health)

target_include_directories(
    test_system_health_smoke
    PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    test_system_health
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric
)
target_link_libraries(test_system_health PRIVATE TT::SystemHealth::Smoke)
