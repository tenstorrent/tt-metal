add_executable(test_factory_system_descriptor test_factory_system_descriptor.cpp)
add_executable(validate_cluster validate_cluster.cpp)

set_target_properties(
    test_factory_system_descriptor
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tools/scaleout
)

set_target_properties(
    validate_cluster
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tools/scaleout
)

# Add the protobuf generated headers directory for tests
target_include_directories(
    test_factory_system_descriptor
    SYSTEM
    PRIVATE
        ${CMAKE_BINARY_DIR}/tools/scaleout
        ${PROJECT_SOURCE_DIR} # Add project root to find tt_metal headers
)

target_include_directories(
    validate_cluster
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/api
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        ${CMAKE_BINARY_DIR}/tt_metal/api
    SYSTEM
    PRIVATE
        ${CMAKE_BINARY_DIR}/tools/scaleout
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(
    test_factory_system_descriptor
    PRIVATE
        TT::ScaleoutTools
        gmock_main
        enchantum::enchantum
        protobuf::libprotobuf
        tt_metal # Add tt_metal library which includes all necessary dependencies
)

target_link_libraries(
    validate_cluster
    PRIVATE
        TT::ScaleoutTools
        enchantum::enchantum
        protobuf::libprotobuf
        tt_metal # Add tt_metal library which includes all necessary dependencies
)
