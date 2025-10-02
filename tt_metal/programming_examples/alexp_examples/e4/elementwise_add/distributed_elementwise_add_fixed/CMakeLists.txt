cmake_minimum_required(VERSION 3.22...3.30)
project(alexp_distributed_elementwise_add_fixed)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# Build the fixed version of distributed elementwise add
add_executable(alexp_distributed_elementwise_add_fixed)
target_sources(alexp_distributed_elementwise_add_fixed PRIVATE distributed_elementwise_add_fixed.cpp)
target_link_libraries(alexp_distributed_elementwise_add_fixed PUBLIC TT::Metalium)

# Set output directory - executable will be in the build directory itself
set_target_properties(
    alexp_distributed_elementwise_add_fixed
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}
)

# Add some helpful information
message(STATUS "Building fixed version of distributed elementwise add with improved memory management")
