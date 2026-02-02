cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_hello_world_datamovement_kernel)

add_executable(metal_example_hello_world_datamovement_kernel)
target_sources(metal_example_hello_world_datamovement_kernel PRIVATE hello_world_datamovement_kernel.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_hello_world_datamovement_kernel PUBLIC TT::Metalium)
