set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/tools/scaleout)

include(sources.cmake)

add_executable(test_factory_system_descriptor)
target_sources(test_factory_system_descriptor PRIVATE ${TEST_FACTORY_SYSTEM_DESCRIPTOR_SRCS})

# Add the protobuf generated headers directory for tests
target_include_directories(test_factory_system_descriptor SYSTEM PRIVATE ${Metalium_BINARY_DIR}/tools/scaleout)

target_link_libraries(
    test_factory_system_descriptor
    PRIVATE
        TT::ScaleoutTools
        gmock_main
        enchantum::enchantum
        protobuf::libprotobuf
        yaml-cpp::yaml-cpp
)

# Make sure protobuf files are generated before compiling
add_dependencies(test_factory_system_descriptor scaleout_generated_protos)

# Test for descriptor merger functionality
add_executable(test_descriptor_merger)
target_sources(test_descriptor_merger PRIVATE ${TEST_DESCRIPTOR_MERGER_SRCS})

target_include_directories(test_descriptor_merger SYSTEM PRIVATE ${Metalium_BINARY_DIR}/tools/scaleout)

target_link_libraries(
    test_descriptor_merger
    PRIVATE
        TT::ScaleoutTools
        gmock_main
        enchantum::enchantum
        protobuf::libprotobuf
)

# Make sure protobuf files are generated before compiling descriptor merger tests
add_dependencies(test_descriptor_merger scaleout_generated_protos)

# Test for link retraining validation
add_executable(test_link_retraining)
target_sources(test_link_retraining PRIVATE ${TEST_LINK_RETRAINING_SRCS})

target_include_directories(
    test_link_retraining
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        "$<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>"
)
target_include_directories(
    test_link_retraining
    SYSTEM
    PRIVATE
        ${Metalium_BINARY_DIR}/tools/scaleout # For generated protobuf headers
)

# TT::ScaleoutTools is required explicitly: this target includes <factory_system_descriptor/utils.hpp>
# and calls into scaleout_tools, whose symbols are no longer exported from libtt_metal.so (see the
# visibility comment in tools/scaleout/CMakeLists.txt). The sibling test targets already link it.
target_link_libraries(
    test_link_retraining
    PRIVATE
        TT::ScaleoutTools
        test_common_libs
        tt_metal
        protobuf::libprotobuf
        yaml-cpp::yaml-cpp
        enchantum::enchantum
)

# Make sure protobuf files are generated before compiling
add_dependencies(test_link_retraining scaleout_generated_protos)

# Test for MGD generation from cabling descriptors
add_executable(test_cabling_descriptor_mgd_generation)
target_sources(test_cabling_descriptor_mgd_generation PRIVATE ${TEST_CABLING_DESCRIPTOR_MGD_GENERATION_SRCS})

target_include_directories(
    test_cabling_descriptor_mgd_generation
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${Metalium_BINARY_DIR}/tools/scaleout
        $<TARGET_PROPERTY:fabric,BINARY_DIR>
        "$<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>"
)

# TT::ScaleoutTools is required explicitly. generate_mgd_lib is an OBJECT library, so its own
# PUBLIC link to scaleout_tools does not propagate those objects to consumers -- only usage
# requirements. These symbols used to be satisfied by libtt_metal.so's exports, which are now
# hidden (see the visibility comment in tools/scaleout/CMakeLists.txt).
target_link_libraries(
    test_cabling_descriptor_mgd_generation
    PRIVATE
        TT::ScaleoutTools
        test_common_libs
        tt_metal
        generate_mgd_lib
        protobuf::libprotobuf
        gmock_main
)

# Make sure protobuf files are generated before compiling
add_dependencies(test_cabling_descriptor_mgd_generation scaleout_generated_protos)
add_dependencies(test_cabling_descriptor_mgd_generation fabric_generated_protos)
add_dependencies(test_cabling_descriptor_mgd_generation tt_metal)

# Test for host_id assignment in DFS order (matching Python tt-CableGen behavior)
add_executable(test_host_id_assignment)
target_sources(test_host_id_assignment PRIVATE ${TEST_HOST_ID_ASSIGNMENT_SRCS})

target_include_directories(test_host_id_assignment SYSTEM PRIVATE ${Metalium_BINARY_DIR}/tools/scaleout)

target_link_libraries(
    test_host_id_assignment
    PRIVATE
        TT::ScaleoutTools
        gmock_main
        enchantum::enchantum
        protobuf::libprotobuf
)

# Make sure protobuf files are generated before compiling host_id assignment tests
add_dependencies(test_host_id_assignment scaleout_generated_protos)

# Test for opt-in include/exclude instance filtering
add_executable(test_instance_filter)
target_sources(test_instance_filter PRIVATE ${TEST_INSTANCE_FILTER_SRCS})

target_include_directories(test_instance_filter SYSTEM PRIVATE ${Metalium_BINARY_DIR}/tools/scaleout)

target_link_libraries(
    test_instance_filter
    PRIVATE
        TT::ScaleoutTools
        gmock_main
        enchantum::enchantum
        protobuf::libprotobuf
)

# Make sure protobuf files are generated before compiling instance filter tests
add_dependencies(test_instance_filter scaleout_generated_protos)

add_executable(test_generate_rank_bindings)
target_sources(test_generate_rank_bindings PRIVATE ${TEST_GENERATE_RANK_BINDINGS_SRCS})

target_include_directories(
    test_generate_rank_bindings
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tools/scaleout/src
        "$<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>"
)

target_link_libraries(
    test_generate_rank_bindings
    PRIVATE
        test_common_libs
        tt_metal
        yaml-cpp::yaml-cpp
        gmock_main
        enchantum::enchantum
)
