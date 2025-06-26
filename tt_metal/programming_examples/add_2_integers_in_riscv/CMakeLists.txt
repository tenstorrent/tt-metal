cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_add_2_integers_in_riscv)

add_executable(metal_example_add_2_integers_in_riscv)
target_sources(metal_example_add_2_integers_in_riscv PRIVATE add_2_integers_in_riscv.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_add_2_integers_in_riscv PUBLIC TT::Metalium)
