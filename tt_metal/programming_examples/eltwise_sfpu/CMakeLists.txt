cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_eltwise_sfpu)

add_executable(metal_example_eltwise_sfpu)
target_sources(metal_example_eltwise_sfpu PRIVATE eltwise_sfpu.cpp)

find_package(TT-Metalium REQUIRED)
target_link_libraries(metal_example_eltwise_sfpu PUBLIC TT::Metalium)
