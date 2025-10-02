cmake_minimum_required(VERSION 3.22...3.30)
project(alexp_elementwise_add)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# Build main distributed elementwise add example
add_executable(alexp_distributed_elementwise_add)
target_sources(alexp_distributed_elementwise_add PRIVATE distributed_elementwise_add.cpp)
target_link_libraries(alexp_distributed_elementwise_add PUBLIC TT::Metalium)

# Build customer version of distributed elementwise add
add_executable(alexp_distributed_elementwise_add_customer)
target_sources(
    alexp_distributed_elementwise_add_customer
    PRIVATE
        distributed_elementwise_add_customer/distributed_elementwise_add.cpp
)
target_link_libraries(alexp_distributed_elementwise_add_customer PUBLIC TT::Metalium)

# Add the fixed version subdirectory
add_subdirectory(distributed_elementwise_add_fixed)

# Set output directory - executables will be in the build directory itself
set_target_properties(
    alexp_distributed_elementwise_add
    alexp_distributed_elementwise_add_customer
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}
)
