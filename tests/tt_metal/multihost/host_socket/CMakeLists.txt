# HostMeshSocket carries a D2D stream over the host network, so the tests only
# exist when that transport was built.
if(NOT TT_METAL_ENABLE_HOST_TRANSPORT)
    return()
endif()

add_library(multi_host_socket_transport OBJECT)
add_library(TT::Metalium::Test::Distributed::HostSocket ALIAS multi_host_socket_transport)
TT_ENABLE_UNITY_BUILD(multi_host_socket_transport)

target_sources(
    multi_host_socket_transport
    PRIVATE
        main.cpp
        host_socket_test_utils.cpp
        test_host_socket.cpp
        test_host_socket_multirank.cpp
        test_host_transport.cpp
        test_socket_hotswap.cpp
        test_host_socket_latency.cpp
)

target_include_directories(
    multi_host_socket_transport
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost
)

target_link_libraries(multi_host_socket_transport PRIVATE test_metal_common_libs)

add_executable(multi_host_socket_transport_tests)
set_target_properties(
    multi_host_socket_transport_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
target_link_libraries(multi_host_socket_transport_tests PRIVATE multi_host_socket_transport)
