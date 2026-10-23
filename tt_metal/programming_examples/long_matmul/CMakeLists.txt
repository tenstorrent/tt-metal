cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_long_matmul)

add_executable(metal_example_long_matmul)
target_sources(metal_example_long_matmul PRIVATE long_matmul.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(metal_example_long_matmul PRIVATE TT::Metalium)

if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(metal_example_long_matmul TT::CommonPCH)
endif()
