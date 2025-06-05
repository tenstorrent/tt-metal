# Define the object library with the name you want
add_library(multi_host_fabric OBJECT)
add_library(TT::Metalium::Test::Distributed::MultiHostFabric ALIAS multi_host_fabric)
TT_ENABLE_UNITY_BUILD(multi_host_fabric)

target_sources(
    multi_host_fabric
    PRIVATE
        main.cpp
        intermesh_routing.cpp
        intermesh_routing_test_utils.cpp
        socket_send_recv.cpp
)

target_include_directories(
    multi_host_fabric
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost
)

target_link_libraries(multi_host_fabric PRIVATE test_metal_common_libs)

add_executable(multi_host_fabric_tests)
set_target_properties(
    multi_host_fabric_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

# Link the object library to the executable
target_link_libraries(multi_host_fabric_tests PRIVATE multi_host_fabric)
