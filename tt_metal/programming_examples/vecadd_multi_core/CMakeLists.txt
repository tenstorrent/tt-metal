cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_vecadd_multi_core)

add_executable(metal_example_vecadd_multi_core)
target_sources(metal_example_vecadd_multi_core PRIVATE vecadd_multi_core.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_vecadd_multi_core PUBLIC TT::Metalium)
