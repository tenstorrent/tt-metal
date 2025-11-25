cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_llk_init_uninit_ttsim_tensix_dump)

add_executable(metal_example_llk_init_uninit_ttsim_tensix_dump)
target_sources(metal_example_llk_init_uninit_ttsim_tensix_dump PRIVATE llk_init_uninit_ttsim_tensix_dump.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_llk_init_uninit_ttsim_tensix_dump PUBLIC TT::Metalium)
