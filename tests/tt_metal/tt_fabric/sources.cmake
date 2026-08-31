# Source files for tt_metal tt_fabric tests
# Module owners should update this file when adding/removing/renaming source files

set(UNIT_TESTS_FABRIC_SRC
    common/utils.cpp
    common/fabric_worker_kernel_helpers.cpp
    common/fabric_command_interface.cpp
    fabric_router/test_routing_tables.cpp
    fabric_router/test_pipeline_builder.cpp
    fabric_router/test_mesh_graph_descriptor.cpp
    fabric_router/test_express_ring_topology.cpp
    fabric_router/test_2d_route_codec.cpp
    fabric_router/test_axis_topology_sweep.cpp
    fabric_router/test_mcast_reverse_tree.cpp
    fabric_router/test_physical_grouping_descriptor.cpp
    fabric_router/test_topology_mapper.cpp
    fabric_router/test_topology_mapper_utils.cpp
    fabric_router/test_topology_solver.cpp
    fabric_router/test_topology_sat_encoder.cpp
    fabric_router/test_custom_routing_tables.cpp
    fabric_router/test_multi_host.cpp
    fabric_router/test_connection_registry.cpp
    fabric_router/test_connection_establishment.cpp
    fabric_router/test_express_connection_wiring.cpp
    fabric_router/test_protected_domain_effects.cpp
    fabric_router/test_fabric_edge_capability.cpp
    fabric_router/test_direction_slot_bijection.cpp
    fabric_router/test_router_wiring_rules.cpp
    fabric_router/test_router_turn_set.cpp
    fabric_router/test_injection_policies.cpp
    fabric_router/test_stream_assignment.cpp
    fabric_router/test_fabric_topology_helpers.cpp
    fabric_router/test_fabric_opt_level.cpp
    fabric_router/test_channel_trimming_capture.cpp
    fabric_router/test_static_sized_channels_allocator.cpp
    disaggregation/test_kv_chunk_address_table_protobuf.cpp
    fabric_data_movement/test_basic_fabric_apis.cpp
    fabric_data_movement/test_basic_1d_fabric.cpp
    fabric_data_movement/test_mesh_multicast_source_inject.cpp
    fabric_data_movement/test_sparse_mcast_perpage.cpp
    fabric_data_movement/test_basic_fabric_mux.cpp
    fabric_data_movement/test_basic_fabric_mux_v2.cpp
    fabric_data_movement/test_fabric_traffic_generator_kernel.cpp
    fabric_router/test_physical_descriptor_builder.cpp
)

set(UNIT_TESTS_PHYSICAL_DISCOVERY_SRC physical_discovery/test_physical_system_descriptor.cpp)

set(TEST_SYSTEM_HEALTH_SMOKE_SOURCES system_health/test_system_health.cpp)

set(TEST_FABRIC_SMOKE_SOURCES fabric_data_movement/test_basic_fabric_smoke.cpp)
