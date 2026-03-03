cmake_minimum_required(VERSION 3.22...4.2)
project(programming_examples_tests)

find_package(Protobuf REQUIRED)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

add_executable(package_integration_test)
target_sources(package_integration_test PRIVATE package_integration_test.cpp)
target_link_libraries(
    package_integration_test
    PUBLIC
        TT::Metalium
        protobuf::libprotobuf
)
target_precompile_headers(package_integration_test REUSE_FROM TT::CommonPCH)

add_executable(out_of_tree_test)
target_sources(out_of_tree_test PRIVATE out_of_tree_test.cpp)
target_link_libraries(out_of_tree_test PUBLIC TT::Metalium)
target_precompile_headers(out_of_tree_test REUSE_FROM TT::CommonPCH)

add_executable(mesh_device_test)
target_sources(mesh_device_test PRIVATE mesh_device_test.cpp)
target_link_libraries(mesh_device_test PUBLIC TT::Metalium)
target_precompile_headers(mesh_device_test REUSE_FROM TT::CommonPCH)

# Any other integration tests should be added here
