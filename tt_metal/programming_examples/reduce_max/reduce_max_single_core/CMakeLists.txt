add_executable(metal_example_reduce_max_single_core)
target_sources(metal_example_reduce_max_single_core PRIVATE reduce_max_single_core.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    metal_example_reduce_max_single_core
    PRIVATE
        TT::Metalium
)
