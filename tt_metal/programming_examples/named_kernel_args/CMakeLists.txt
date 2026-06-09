cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_named_kernel_args)

add_executable(metal_example_named_kernel_args)
target_sources(metal_example_named_kernel_args PRIVATE named_kernel_args.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_named_kernel_args PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(metal_example_named_kernel_args TT::CommonPCH)
endif()
