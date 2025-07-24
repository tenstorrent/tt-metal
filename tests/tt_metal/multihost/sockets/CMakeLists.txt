# Define the object library with the name you want
add_library(multi_host_sockets OBJECT)
add_library(TT::Metalium::Test::Distributed::MultihostSockets ALIAS multi_host_sockets)
TT_ENABLE_UNITY_BUILD(multi_host_sockets)

target_sources(
    multi_host_sockets
    PRIVATE
        main.cpp
        test_handshaking.cpp
)

target_include_directories(
    multi_host_sockets
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost
)

target_link_libraries(multi_host_sockets PRIVATE test_metal_common_libs)

# Target for all STL tests regardless of duration
add_executable(multi_host_socket_tests)
set_target_properties(
    multi_host_socket_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

# Link the object library to the executable
target_link_libraries(multi_host_socket_tests PRIVATE multi_host_sockets)
