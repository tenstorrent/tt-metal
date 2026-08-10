# Source files for tools/scaleout targets.
# Module owners should update this file when adding/removing/renaming source files.

set(SCALEOUT_PROTO_SCHEMAS
    factory_system_descriptor/schemas/factory_system_descriptor.proto
    cabling_descriptor/schemas/node_config.proto
    cabling_descriptor/schemas/cluster_config.proto
    deployment_descriptor/schemas/deployment.proto
    validation/schemas/ethernet_link_metrics.proto
)

set(SCALEOUT_TOOLS_API_HEADERS
    board/board.hpp
    cabling_generator/cabling_generator.hpp
    cabling_generator/regen_descriptors.hpp
    connector/connector.hpp
    factory_system_descriptor/query.hpp
    factory_system_descriptor/utils.hpp
    node/node_types.hpp
)

set(SCALEOUT_TOOLS_SRCS
    node/node.cpp
    node/node_types.cpp
    board/board.cpp
    cabling_generator/cabling_generator.cpp
    cabling_generator/regen_descriptors.cpp
    connector/connector.cpp
    factory_system_descriptor/query.cpp
    factory_system_descriptor/utils.cpp
)

set(RUN_CLUSTER_VALIDATION_SRCS
    validation/run_cluster_validation.cpp
    validation/utils/ethernet_link_metrics_serialization.cpp
    validation/utils/ethernet_link_api.cpp
    validation/utils/cluster_validation_utils.cpp
)

set(RUN_FABRIC_MANAGER_SRCS
    fabric_manager/run_fabric_manager.cpp
    fabric_manager/utils/fabric_manager_utils.cpp
)

set(SCALEOUT_2D_BIG_MESH_CABLING_GEN_SRCS src/2d_big_mesh_cabling_gen.cpp)

set(RUN_CABLING_GENERATOR_SRCS src/run_cabling_generator.cpp)

set(RUN_REGEN_DESCRIPTORS_SRCS src/run_regen_descriptors.cpp)

set(GENERATE_CLUSTER_DESCRIPTOR_SRCS src/generate_cluster_descriptor.cpp)

set(GENERATE_MGD_LIB_SRCS generate_mgd/generate_mgd.cpp)

set(GENERATE_MGD_SRCS generate_mgd/generate_mgd_main.cpp)

set(GENERATE_RANK_BINDINGS_SRCS src/generate_rank_bindings.cpp)

set(RUN_MPI_STRESS_TEST_SRCS validation/run_mpi_stress_test.cpp)
