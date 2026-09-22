# The host-to-device leg on its own: one rank, no peer, so this needs no MPI launcher
# even though it lives under multihost/.
add_library(d2h2h2d_socket_h2d_bw_test OBJECT)
TT_ENABLE_UNITY_BUILD(d2h2h2d_socket_h2d_bw_test)

target_sources(d2h2h2d_socket_h2d_bw_test PRIVATE test_h2d_bw.cpp)

target_include_directories(
    d2h2h2d_socket_h2d_bw_test
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tests
)

target_compile_definitions(
    d2h2h2d_socket_h2d_bw_test
    PRIVATE
        TT_METAL_HOST_BRIDGE=1
        TT_DIRECT_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(d2h2h2d_socket_h2d_bw_test PRIVATE test_metal_common_libs)

add_executable(d2h2h2d_socket_h2d_bw)
set_target_properties(
    d2h2h2d_socket_h2d_bw
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
target_link_libraries(d2h2h2d_socket_h2d_bw PRIVATE d2h2h2d_socket_h2d_bw_test)
