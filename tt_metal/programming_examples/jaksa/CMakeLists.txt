cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_jaksa)

add_executable(metal_example_jaksa)
target_sources(metal_example_jaksa PRIVATE jaksa.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_jaksa PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    target_precompile_headers(metal_example_jaksa REUSE_FROM TT::CommonPCH)
endif()
