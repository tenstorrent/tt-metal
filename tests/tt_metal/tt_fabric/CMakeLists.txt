set(UNIT_TESTS_FABRIC_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_routing_tables.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_mesh_graph_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_topology_mapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_topology_mapper_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_topology_solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_custom_routing_tables.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_multi_host.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_connection_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_router_channel_mapping.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_router_connections.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_router_connection_mapping.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_router_archetypes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_builder_connection_mapping.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_connection_mapping_logic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_fabric_builder_local_connections.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_z_router_integration.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_z_router_device_detection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_fabric_topology_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_data_movement/test_basic_fabric_apis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_data_movement/test_basic_1d_fabric.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_data_movement/test_basic_fabric_mux.cpp
)

add_executable(fabric_unit_tests ${UNIT_TESTS_FABRIC_SRC})

target_link_libraries(
    fabric_unit_tests
    PRIVATE
        tt_metal
        test_common_libs
        yaml-cpp::yaml-cpp
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

set(UNIT_TESTS_PHYSICAL_DISCOVERY_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/physical_discovery/test_physical_system_descriptor.cpp
)

add_executable(test_physical_discovery ${UNIT_TESTS_PHYSICAL_DISCOVERY_SRC})

target_link_libraries(
    test_physical_discovery
    PRIVATE
        tt_metal
        test_common_libs
        yaml-cpp::yaml-cpp
)

target_include_directories(
    test_physical_discovery
    PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)

set_target_properties(
    test_physical_discovery
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric
)

# Fabric Elastic Channels Test
add_executable(fabric_elastic_channels_host_test)

target_sources(
    fabric_elastic_channels_host_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/feature_bringup/fabric_elastic_channels_host_test.cpp
)

target_link_libraries(
    fabric_elastic_channels_host_test
    PRIVATE
        tt_metal
        test_common_libs
)

target_include_directories(
    fabric_elastic_channels_host_test
    PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/feature_bringup
)

set_target_properties(
    fabric_elastic_channels_host_test
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

# ---- Bench: unicast ----
add_executable(
    bench_unicast
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/collectives/unicast/bench_unicast_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/collectives/unicast/run_unicast_once.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/collectives/common/perf_helpers.cpp
)

target_link_libraries(
    bench_unicast
    PRIVATE
        tt_metal
        gtest
        test_common_libs
)

add_dependencies(bench_unicast tt_metal)

target_include_directories(
    bench_unicast
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${PROJECT_BINARY_DIR}/tt_metal/third_party/umd/device/api
        ${PROJECT_BINARY_DIR}/tt_metal/third_party/umd/device/generated
)

set_target_properties(
    bench_unicast
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric
)

# ---- Test: addrgen_write (unified test for all addrgen variants) ----
add_executable(
    test_addrgen_write
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_data_movement/addrgen_write/test_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_data_movement/addrgen_write/unicast_runner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_data_movement/addrgen_write/multicast_runner.cpp
)

target_link_libraries(
    test_addrgen_write
    PRIVATE
        tt_metal
        gtest
        test_common_libs
)

add_dependencies(test_addrgen_write tt_metal)

target_include_directories(
    test_addrgen_write
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${PROJECT_BINARY_DIR}/tt_metal/third_party/umd/device/api
        ${PROJECT_BINARY_DIR}/tt_metal/third_party/umd/device/generated
)

set_target_properties(
    test_addrgen_write
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric
)

# ---- Test: bandwidth telemetry validation ----
add_executable(
    test_bandwidth_telemetry_validation
    ${CMAKE_CURRENT_SOURCE_DIR}/test_bandwidth_telemetry_validation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/collectives/unicast/run_unicast_once.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/collectives/common/perf_helpers.cpp
)

target_link_libraries(
    test_bandwidth_telemetry_validation
    PRIVATE
        tt_metal
        gtest
        test_common_libs
)

add_dependencies(test_bandwidth_telemetry_validation tt_metal)

target_include_directories(
    test_bandwidth_telemetry_validation
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${PROJECT_BINARY_DIR}/tt_metal/third_party/umd/device/api
        ${PROJECT_BINARY_DIR}/tt_metal/third_party/umd/device/generated
)

set_target_properties(
    test_bandwidth_telemetry_validation
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric
)
