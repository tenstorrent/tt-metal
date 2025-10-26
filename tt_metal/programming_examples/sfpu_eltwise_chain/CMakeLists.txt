cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_sfpu_eltwise_chain)

add_executable(metal_example_sfpu_eltwise_chain)
target_sources(metal_example_sfpu_eltwise_chain PRIVATE sfpu_eltwise_chain.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_sfpu_eltwise_chain PUBLIC TT::Metalium)
