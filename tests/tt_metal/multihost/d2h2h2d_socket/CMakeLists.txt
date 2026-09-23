# The device-to-host leg on its own: one rank, no peer, so this needs no MPI launcher even
# though it lives under multihost/.
add_library(d2h2h2d_socket_d2h_bw_test OBJECT)
TT_ENABLE_UNITY_BUILD(d2h2h2d_socket_d2h_bw_test)

target_sources(d2h2h2d_socket_d2h_bw_test PRIVATE test_d2h_bw.cpp)

target_include_directories(
    d2h2h2d_socket_d2h_bw_test
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tests
)

# TT_DIRECT_KERNEL_DIR: the test passes an absolute path to CreateKernel, so the kernel is
# found from the build directory without a copy step.
target_compile_definitions(
    d2h2h2d_socket_d2h_bw_test
    PRIVATE
        TT_METAL_HOST_BRIDGE=1
        TT_DIRECT_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

# benchmark, not benchmark_main: this binary keeps its own main so DistributedContext::create
# sees argv before google-benchmark takes its share of it.
target_link_libraries(
    d2h2h2d_socket_d2h_bw_test
    PRIVATE
        test_metal_common_libs
        benchmark::benchmark
)

add_executable(d2h2h2d_socket_d2h_bw)
set_target_properties(
    d2h2h2d_socket_d2h_bw
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
target_link_libraries(
    d2h2h2d_socket_d2h_bw
    PRIVATE
        d2h2h2d_socket_d2h_bw_test
        benchmark::benchmark
)
