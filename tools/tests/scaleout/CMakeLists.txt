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
)

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
        ${CMAKE_BINARY_DIR}/tools/scaleout # For generated protobuf headers
        "$<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>"
)

target_link_libraries(
    test_link_retraining
    PRIVATE
        test_common_libs
        tt_metal
        TT::ScaleoutTools
        protobuf::libprotobuf
)

# Make sure protobuf files are generated before compiling
add_dependencies(test_link_retraining scaleout_generated_protos)
