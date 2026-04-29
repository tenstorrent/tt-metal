cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_core_map)

add_executable(metal_example_core_map)
target_sources(metal_example_core_map PRIVATE core_map.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_core_map PUBLIC TT::Metalium)

# This example reaches into internal metal headers (llrt/metal_soc_descriptor,
# llrt/tt_cluster, impl/context/metal_context) to verify the physical NOC0 map
# published in the documentation against what the device actually reports. The
# public TT::Metalium interface only exposes `api/` and `api/tt-metalium/`; we
# additionally pull in the Metalium::Metal include directories (which include
# the full tt_metal source tree) so the internal headers resolve. This matches
# what tests do, e.g. test_soc_descriptor in tests/tt_metal/tt_metal/api.
if(TARGET Metalium::Metal)
    target_include_directories(
        metal_example_core_map
        BEFORE
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
    )
endif()

if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(metal_example_core_map TT::CommonPCH)
endif()
