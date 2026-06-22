cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_x280_spsc)

add_executable(metal_example_x280_spsc)
target_sources(metal_example_x280_spsc PRIVATE x280_spsc.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_x280_spsc PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(metal_example_x280_spsc TT::CommonPCH)
endif()
