cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_sfpu_scalar_repro)

add_executable(metal_example_sfpu_scalar_repro)
target_sources(metal_example_sfpu_scalar_repro PRIVATE sfpu_scalar_repro.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_sfpu_scalar_repro PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(metal_example_sfpu_scalar_repro TT::CommonPCH)
endif()
