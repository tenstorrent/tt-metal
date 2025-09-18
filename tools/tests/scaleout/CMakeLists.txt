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
