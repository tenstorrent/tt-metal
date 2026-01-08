cmake_minimum_required(VERSION 3.22...3.30)
project(metal_fabric_api_tests)

add_executable(metal_fabric_api_tests)
target_sources(
    metal_fabric_api_tests
    PRIVATE
        fabric_benchmark_runner.cpp
        default_intra_mesh_routing_test_impl.cpp
        incremental_packet_size_test_impl.cpp
)

target_link_libraries(
    metal_fabric_api_tests
    PUBLIC
        tt_metal
        TT::Metalium
)
target_include_directories(
    metal_fabric_api_tests
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/test_utils
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
