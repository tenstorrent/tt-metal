add_executable(test_factory_system_descriptor test_factory_system_descriptor.cpp)

set_target_properties(
    test_factory_system_descriptor
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test
)

target_link_libraries(
    test_factory_system_descriptor
    PRIVATE
        TT::ScaleoutTools
        gmock_main
)
