cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_l2cpu_noc_transfer)

add_executable(metal_example_l2cpu_noc_transfer)
target_sources(metal_example_l2cpu_noc_transfer PRIVATE l2cpu_noc_transfer.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_l2cpu_noc_transfer PUBLIC TT::Metalium)
