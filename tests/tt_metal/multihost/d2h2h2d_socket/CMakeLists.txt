# D2H2H2DSocket -- the oneway_volume run, ported from tt-direct-dev's finalized/test.
#
if(NOT TARGET OpenMPI::MPI)
    message(STATUS "MPI not available: d2h2h2d_socket tests will not be built")
    return()
endif()

add_library(d2h2h2d_socket_test OBJECT)
add_library(TT::Metalium::Test::Distributed::D2H2H2DSocket ALIAS d2h2h2d_socket_test)
TT_ENABLE_UNITY_BUILD(d2h2h2d_socket_test)

target_sources(d2h2h2d_socket_test PRIVATE test_oneway_volume.cpp)

target_include_directories(
    d2h2h2d_socket_test
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost
)

# TT_METAL_HOST_BRIDGE, matching tt_metal/distributed/CMakeLists.txt. The test's own #error
# fires without it rather than emitting several hundred lines about an undeclared Transport.
target_compile_definitions(
    d2h2h2d_socket_test
    PRIVATE
        TT_METAL_HOST_BRIDGE=1
        # Where the device kernels live. Was TT_DIRECT_KERNEL_DIR in the out-of-tree build.sh;
        # in-tree it is this directory, and it is a define rather than a runtime default so a
        # binary run from anywhere still finds its own kernels.
        TT_DIRECT_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(d2h2h2d_socket_test PRIVATE test_metal_common_libs OpenMPI::MPI)

add_executable(d2h2h2d_socket_oneway_volume)
set_target_properties(
    d2h2h2d_socket_oneway_volume
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
target_link_libraries(d2h2h2d_socket_oneway_volume PRIVATE d2h2h2d_socket_test)
