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
