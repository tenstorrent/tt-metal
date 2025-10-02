cmake_minimum_required(VERSION 3.22...3.30)
project(multi_tensix_distributed_elementwise_add)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# Build simple multi-tensix example
add_executable(multi_tensix_simple)
target_sources(multi_tensix_simple PRIVATE multi_tensix_simple.cpp)
target_link_libraries(multi_tensix_simple PUBLIC TT::Metalium)
target_include_directories(multi_tensix_simple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Build full multi-tensix distributed elementwise add
add_executable(multi_tensix_distributed_elementwise_add)
target_sources(multi_tensix_distributed_elementwise_add PRIVATE multi_tensix_distributed_elementwise_add.cpp)
target_link_libraries(multi_tensix_distributed_elementwise_add PUBLIC TT::Metalium)
target_include_directories(multi_tensix_distributed_elementwise_add PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Set output directory to match other examples
set_target_properties(
    multi_tensix_simple
    multi_tensix_distributed_elementwise_add
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/programming_examples/alexp_examples/e4/multi_tensix_elementwise_add
)
