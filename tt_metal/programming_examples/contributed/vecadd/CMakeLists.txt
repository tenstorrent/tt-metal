cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_vecadd)

add_executable(metal_example_vecadd)
target_sources(metal_example_vecadd PRIVATE vecadd.cpp)

find_package(TT-Metalium REQUIRED)
target_link_libraries(metal_example_vecadd PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(metal_example_vecadd TT::CommonPCH)
endif()
