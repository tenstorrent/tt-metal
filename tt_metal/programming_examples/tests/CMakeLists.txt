cmake_minimum_required(VERSION 3.22...3.30)
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

add_executable(out_of_tree_test)
target_sources(out_of_tree_test PRIVATE out_of_tree_test.cpp)
target_link_libraries(out_of_tree_test PUBLIC TT::Metalium)

# Any other integration tests should be added here
