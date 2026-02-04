# Source files for tt_metal tt_fabric tests
# Module owners should update this file when adding/removing/renaming source files

set(UNIT_TESTS_FABRIC_SRC
    common/utils.cpp
    common/fabric_worker_kernel_helpers.cpp
    common/fabric_command_interface.cpp
    fabric_router/test_routing_tables.cpp
    fabric_router/test_mesh_graph_descriptor.cpp
    fabric_router/test_topology_mapper.cpp
    fabric_router/test_topology_mapper_utils.cpp
    fabric_router/test_topology_solver.cpp
    fabric_router/test_custom_routing_tables.cpp
    fabric_router/test_multi_host.cpp
    fabric_router/test_connection_registry.cpp
    fabric_router/test_router_channel_mapping.cpp
    fabric_router/test_router_connections.cpp
    fabric_router/test_router_connection_mapping.cpp
    fabric_router/test_router_archetypes.cpp
    fabric_router/test_builder_connection_mapping.cpp
    fabric_router/test_connection_mapping_logic.cpp
    fabric_router/test_fabric_builder_local_connections.cpp
    fabric_router/test_z_router_integration.cpp
    fabric_router/test_z_router_device_detection.cpp
    fabric_router/test_fabric_topology_helpers.cpp
    fabric_data_movement/test_basic_fabric_apis.cpp
    fabric_data_movement/test_basic_1d_fabric.cpp
    fabric_data_movement/test_basic_fabric_mux.cpp
    fabric_data_movement/test_fabric_traffic_generator_kernel.cpp
)

set(UNIT_TESTS_PHYSICAL_DISCOVERY_SRC physical_discovery/test_physical_system_descriptor.cpp)

set(FABRIC_ELASTIC_CHANNELS_HOST_TEST_SOURCES feature_bringup/fabric_elastic_channels_host_test.cpp)

set(TEST_SYSTEM_HEALTH_SMOKE_SOURCES system_health/test_system_health.cpp)

set(TEST_FABRIC_SMOKE_SOURCES fabric_data_movement/test_basic_fabric_smoke.cpp)
