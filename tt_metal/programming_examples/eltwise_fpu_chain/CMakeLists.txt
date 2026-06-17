cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_eltwise_fpu_chain)

add_executable(metal_example_eltwise_fpu_chain)
target_sources(metal_example_eltwise_fpu_chain PRIVATE eltwise_fpu_chain.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_eltwise_fpu_chain PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(metal_example_eltwise_fpu_chain TT::CommonPCH)
endif()
