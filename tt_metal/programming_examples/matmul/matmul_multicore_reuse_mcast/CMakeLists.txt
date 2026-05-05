add_executable(metal_example_matmul_multicore_reuse_mcast)
target_sources(metal_example_matmul_multicore_reuse_mcast PRIVATE matmul_multicore_reuse_mcast.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    metal_example_matmul_multicore_reuse_mcast
    PRIVATE
        TT::Metalium
        Matmul::Common
)

# Use smaller problem sizes in CI to keep gate times fast (~8x speedup).
# Unset this to build with the original walkthrough values.
target_compile_definitions(metal_example_matmul_multicore_reuse_mcast PRIVATE TT_METAL_CI_MODE)
