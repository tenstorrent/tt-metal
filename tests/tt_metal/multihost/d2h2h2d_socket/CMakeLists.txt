# D2DSocket -- the oneway_volume run, ported from tt-direct-dev's finalized/test.
#
if(NOT TARGET OpenMPI::MPI)
    message(STATUS "MPI not available: d2h2h2d_socket tests will not be built")
    return()
endif()

add_library(d2h2h2d_socket_d2d_test OBJECT)
add_library(TT::Metalium::Test::Distributed::D2DSocket ALIAS d2h2h2d_socket_d2d_test)
TT_ENABLE_UNITY_BUILD(d2h2h2d_socket_d2d_test)

target_sources(d2h2h2d_socket_d2d_test PRIVATE test_oneway_volume_d2d.cpp)

target_include_directories(
    d2h2h2d_socket_d2d_test
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost
)

target_compile_definitions(
    d2h2h2d_socket_d2d_test
    PRIVATE
        TT_METAL_HOST_BRIDGE=1
        TT_DIRECT_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(
    d2h2h2d_socket_d2d_test
    PRIVATE
        test_metal_common_libs
        OpenMPI::MPI
)

add_executable(d2h2h2d_socket_oneway_volume_d2d)
set_target_properties(
    d2h2h2d_socket_oneway_volume_d2d
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
target_link_libraries(d2h2h2d_socket_oneway_volume_d2d PRIVATE d2h2h2d_socket_d2d_test)
