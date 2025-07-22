cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_vecadd_sharding)

add_executable(metal_example_vecadd_sharding)
target_sources(metal_example_vecadd_sharding PRIVATE vecadd_sharding.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_vecadd_sharding PUBLIC TT::Metalium)
