cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_risc_scan_bench)

add_executable(metal_example_risc_scan_bench)
target_sources(metal_example_risc_scan_bench PRIVATE risc_scan_bench.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_risc_scan_bench PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(metal_example_risc_scan_bench TT::CommonPCH)
endif()
