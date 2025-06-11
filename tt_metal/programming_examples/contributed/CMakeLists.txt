cmake_minimum_required(VERSION 3.22...3.30)
project(metal_examples_contributed)

# Add the vecadd example subdirectory
add_subdirectory(vecadd)

# Copy the executable to the root build directory for CI pipeline to find
add_custom_command(
    TARGET metal_example_vecadd
    POST_BUILD
    COMMAND
        ${CMAKE_COMMAND} -E copy vecadd/metal_example_vecadd metal_example_vecadd
)
