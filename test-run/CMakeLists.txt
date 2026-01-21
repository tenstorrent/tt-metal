cmake_minimum_required(VERSION 3.22...3.30)

# Use internal targets - no need to find packages when building from source
add_executable(test_1x32_all_gather_debug)
target_sources(test_1x32_all_gather_debug PRIVATE test_1x32_all_gather_debug.cpp)

# Link against internal targets
target_link_libraries(
    test_1x32_all_gather_debug
    PUBLIC
        TT::Metalium
        TTNN::CPP
)

# Set output directory
set_target_properties(
    test_1x32_all_gather_debug
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/test-run"
)
