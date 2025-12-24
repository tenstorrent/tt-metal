# Define the object library for tt-switch tests
add_library(ttswitch_tests OBJECT)
add_library(TT::Metalium::Test::Distributed::TTSwitch ALIAS ttswitch_tests)
TT_ENABLE_UNITY_BUILD(ttswitch_tests)

target_sources(
    ttswitch_tests
    PRIVATE
        main.cpp
        test_mesh_device_ttswitch.cpp
)

target_include_directories(
    ttswitch_tests
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost
)

target_link_libraries(
    ttswitch_tests
    PRIVATE
        test_metal_common_libs
        TTNN::CPP
)

# Create the executable
add_executable(multi_host_ttswitch_tests)
set_target_properties(
    multi_host_ttswitch_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

# Link the object library to the executable
target_link_libraries(multi_host_ttswitch_tests PRIVATE ttswitch_tests)
