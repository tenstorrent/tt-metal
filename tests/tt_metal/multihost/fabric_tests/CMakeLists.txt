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
        socket_send_recv_utils.cpp
)

target_include_directories(
    multi_host_fabric
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost
)

target_link_libraries(multi_host_fabric PRIVATE test_metal_common_libs)

# Create separate object library for mesh socket components
add_library(mesh_socket_components OBJECT)
add_library(TT::Metalium::Test::Distributed::MeshSocketComponents ALIAS mesh_socket_components)

target_sources(
    mesh_socket_components
    PRIVATE
        mesh_socket_yaml_parser.cpp
        mesh_socket_test_context.cpp
        socket_send_recv_utils.cpp
)

target_include_directories(
    mesh_socket_components
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost
)

target_link_libraries(
    mesh_socket_components
    PRIVATE
        test_metal_common_libs
        yaml-cpp
)

add_executable(multi_host_fabric_tests)
set_target_properties(
    multi_host_fabric_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

# Link the object library to the executable
target_link_libraries(multi_host_fabric_tests PRIVATE multi_host_fabric)

# Create separate executable for mesh socket tests
add_executable(test_mesh_socket_main mesh_socket_test_main.cpp)
set_target_properties(
    test_mesh_socket_main
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

# Link the mesh socket test main with required libraries
target_link_libraries(
    test_mesh_socket_main
    PRIVATE
        mesh_socket_components
        test_metal_common_libs
        yaml-cpp
)

target_include_directories(
    test_mesh_socket_main
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost
)
