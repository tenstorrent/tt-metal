add_library(multi_host_layer_completion OBJECT)
add_library(TT::Metalium::Test::Distributed::MultihostLayerCompletion ALIAS multi_host_layer_completion)
TT_ENABLE_UNITY_BUILD(multi_host_layer_completion)

target_sources(
    multi_host_layer_completion
    PRIVATE
        main.cpp
        test_layer_completion_routing.cpp
)

target_include_directories(
    multi_host_layer_completion
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost
)

target_link_libraries(multi_host_layer_completion PRIVATE test_metal_common_libs)

add_executable(multi_host_layer_completion_tests)
set_target_properties(
    multi_host_layer_completion_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
target_link_libraries(multi_host_layer_completion_tests PRIVATE multi_host_layer_completion)
