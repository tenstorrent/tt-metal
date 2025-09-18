cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_add_2_integers_in_compute)

add_executable(metal_example_add_2_integers_in_compute)
target_sources(metal_example_add_2_integers_in_compute PRIVATE add_2_integers_in_compute.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_add_2_integers_in_compute PUBLIC TT::Metalium)
