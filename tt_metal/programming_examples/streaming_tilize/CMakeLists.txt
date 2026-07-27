cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_streaming_tilize)

add_executable(metal_example_streaming_tilize)
target_sources(metal_example_streaming_tilize PRIVATE streaming_tilize.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_streaming_tilize PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(metal_example_streaming_tilize TT::CommonPCH)
endif()
