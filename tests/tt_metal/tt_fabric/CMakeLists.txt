set(UNIT_TESTS_FABRIC_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_routing_tables.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_control_plane_logical_to_physical.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_mesh_graph_descriptor.cpp
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

add_library(test_fabric_smoke OBJECT)
add_library(TT::Metalium::Test::Fabric::Smoke ALIAS test_fabric_smoke)

target_sources(test_fabric_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fabric_data_movement/test_basic_fabric_smoke.cpp)

target_include_directories(
    test_fabric_smoke
    PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/perf_microbenchmark/routing
)

target_link_libraries(
    test_fabric_smoke
    PRIVATE
        tt_metal
        test_common_libs
)

add_executable(fabric_smoke_tests)
target_link_libraries(fabric_smoke_tests PRIVATE TT::Metalium::Test::Fabric::Smoke)
set_target_properties(
    fabric_smoke_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric
)

# unit_tests_1d_fabric_latency

add_executable(unit_tests_1d_fabric_latency)
target_include_directories(
    unit_tests_1d_fabric_latency
    PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    unit_tests_1d_fabric_latency
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric
)
target_link_libraries(
    unit_tests_1d_fabric_latency
    PRIVATE
        tt_metal
        fabric
        test_common_libs
)
target_sources(
    unit_tests_1d_fabric_latency
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/fabric_data_movement/test_1d_fabric_loopback_latency.cpp
)

add_executable(test_fabric_1d_bw)
target_include_directories(
    test_fabric_1d_bw
    PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    test_fabric_1d_bw
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric
)
target_link_libraries(
    test_fabric_1d_bw
    PRIVATE
        tt_metal
        fabric
        test_common_libs
)
target_sources(test_fabric_1d_bw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_fabric_edm.cpp)
