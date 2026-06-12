cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_x280_d2h)

add_executable(metal_example_x280_d2h)
target_sources(metal_example_x280_d2h PRIVATE x280_d2h.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_x280_d2h PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(metal_example_x280_d2h TT::CommonPCH)
endif()
