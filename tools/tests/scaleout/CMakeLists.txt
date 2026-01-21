add_executable(test_factory_system_descriptor test_factory_system_descriptor.cpp)

set_target_properties(
    test_factory_system_descriptor
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tools/scaleout
)

# Add the protobuf generated headers directory for tests
target_include_directories(test_factory_system_descriptor SYSTEM PRIVATE ${CMAKE_BINARY_DIR}/tools/scaleout)

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
add_executable(test_descriptor_merger test_descriptor_merger.cpp)

set_target_properties(
    test_descriptor_merger
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tools/scaleout
)

target_include_directories(test_descriptor_merger SYSTEM PRIVATE ${CMAKE_BINARY_DIR}/tools/scaleout)

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
add_executable(
    test_link_retraining
    test_link_retraining.cpp
    ${PROJECT_SOURCE_DIR}/tools/scaleout/validation/utils/cluster_validation_utils.cpp
    ${PROJECT_SOURCE_DIR}/tools/scaleout/validation/utils/ethernet_link_metrics_serialization.cpp
)

set_target_properties(
    test_link_retraining
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tools/scaleout
)

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
        ${CMAKE_BINARY_DIR}/tools/scaleout # For generated protobuf headers
)

target_link_libraries(
    test_link_retraining
    PRIVATE
        test_common_libs
        tt_metal
        protobuf::libprotobuf
        yaml-cpp::yaml-cpp
        enchantum::enchantum
)

# Make sure protobuf files are generated before compiling
add_dependencies(test_link_retraining scaleout_generated_protos)

# Test for MGD generation from cabling descriptors
add_executable(test_cabling_descriptor_mgd_generation test_cabling_descriptor_mgd_generation.cpp)

set_target_properties(
    test_cabling_descriptor_mgd_generation
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tools/scaleout
)

target_include_directories(
    test_cabling_descriptor_mgd_generation
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_BINARY_DIR}/tools/scaleout
        ${CMAKE_BINARY_DIR}/tt_metal/fabric
        "$<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>"
)

target_link_libraries(
    test_cabling_descriptor_mgd_generation
    PRIVATE
        test_common_libs
        tt_metal
        generate_mgd_lib
        protobuf::libprotobuf
        gmock_main
)

# Make sure protobuf files are generated before compiling
add_dependencies(test_cabling_descriptor_mgd_generation scaleout_generated_protos)
add_dependencies(test_cabling_descriptor_mgd_generation tt_metal)
