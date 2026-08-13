# Source files for tools/tests/scaleout targets.
# Module owners should update this file when adding/removing/renaming source files.

set(TEST_FACTORY_SYSTEM_DESCRIPTOR_SRCS test_factory_system_descriptor.cpp)

set(TEST_DESCRIPTOR_MERGER_SRCS test_descriptor_merger.cpp)

set(TEST_LINK_RETRAINING_SRCS
    test_link_retraining.cpp
    # FIXME: reaches outside this directory via PROJECT_SOURCE_DIR to reuse tools/scaleout
    # sources directly — bad practice, should come from a shared target instead.
    ${PROJECT_SOURCE_DIR}/tools/scaleout/validation/utils/cluster_validation_utils.cpp
    ${PROJECT_SOURCE_DIR}/tools/scaleout/validation/utils/ethernet_link_metrics_serialization.cpp
    ${PROJECT_SOURCE_DIR}/tools/scaleout/validation/utils/ethernet_link_api.cpp
)

set(TEST_CABLING_DESCRIPTOR_MGD_GENERATION_SRCS test_cabling_descriptor_mgd_generation.cpp)

set(TEST_HOST_ID_ASSIGNMENT_SRCS test_host_id_assignment.cpp)

set(TEST_INSTANCE_FILTER_SRCS test_instance_filter.cpp)

set(TEST_GENERATE_RANK_BINDINGS_SRCS test_generate_rank_bindings.cpp)
