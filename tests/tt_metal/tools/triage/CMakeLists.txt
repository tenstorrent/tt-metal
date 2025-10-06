# Standalone C++ operation chain for Inspector testing
# This binary is NOT built with unity build to preserve debug info for addr2line

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/tools/triage)

add_executable(run_operation_chain_cpp)
target_sources(run_operation_chain_cpp PRIVATE run_operation_chain.cpp)

# Link against TTNN - this gives us all the necessary libraries
target_link_libraries(run_operation_chain_cpp PRIVATE TTNN::CPP)

# Include project tests directory
target_include_directories(run_operation_chain_cpp PRIVATE ${PROJECT_SOURCE_DIR}/tests)

# Ensure debug symbols for callstack resolution
target_compile_options(
    run_operation_chain_cpp
    PRIVATE
        -g
        -fno-omit-frame-pointer
)
