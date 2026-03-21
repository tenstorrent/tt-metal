add_executable(metal_example_high_power_matmul)
target_sources(metal_example_high_power_matmul PRIVATE high_power_matmul.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(metal_example_high_power_matmul PRIVATE TT::Metalium)

if(TARGET TT::CommonPCH)
    target_precompile_headers(metal_example_high_power_matmul REUSE_FROM TT::CommonPCH)
endif()
