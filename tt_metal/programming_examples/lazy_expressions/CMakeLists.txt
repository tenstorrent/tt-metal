cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_lazy_expressions)

add_executable(metal_example_lazy_expressions)
target_sources(metal_example_lazy_expressions PRIVATE lazy_expressions.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_lazy_expressions PUBLIC TT::Metalium)
