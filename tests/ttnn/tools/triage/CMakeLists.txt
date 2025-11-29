# Standalone C++ operation chain for Inspector testing
# This binary is NOT built with unity build to preserve debug info for addr2line

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/ttnn/tools/triage)

add_executable(run_operation_chain_cpp)
target_sources(run_operation_chain_cpp PRIVATE run_operation_chain.cpp)

# Link against TTNN - this gives us all the necessary libraries
target_link_libraries(run_operation_chain_cpp PRIVATE TTNN::CPP)

# Include project tests directory
target_include_directories(run_operation_chain_cpp PRIVATE ${PROJECT_SOURCE_DIR}/tests)

# Ensure debug symbols and NO optimization for callstack resolution
# Use -O0 (no optimization) to completely prevent inlining that would hide
# the actual call sites in backtraces. Even with -O1, template functions from
# decorators.hpp get inlined into user code, making it impossible to see where
# operations were invoked. -O0 is required to preserve all function calls.
target_compile_options(
    run_operation_chain_cpp
    PRIVATE
        -g
        -O0
        -fno-omit-frame-pointer
)
