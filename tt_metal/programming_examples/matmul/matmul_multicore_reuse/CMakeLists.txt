add_executable(metal_example_matmul_multicore_reuse)
target_sources(metal_example_matmul_multicore_reuse PRIVATE matmul_multicore_reuse.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    metal_example_matmul_multicore_reuse
    PRIVATE
        TT::Metalium
        Matmul::Common
)
