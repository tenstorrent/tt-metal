add_executable(metal_example_reduce_max_mcast)
target_sources(metal_example_reduce_max_mcast PRIVATE reduce_max_mcast.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    metal_example_reduce_max_mcast
    PRIVATE
        TT::Metalium
)
