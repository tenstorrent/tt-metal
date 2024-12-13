add_executable(
    test_distributed
    test_distributed.cpp
    test_distributed_reshape.cpp
)
add_executable(test_distributed_atexit test_distributed_atexit.cpp)

# Set up properties for the target
setup_ttnn_test_target(test_distributed)
setup_ttnn_test_target(test_distributed_atexit)
# Add test to CTest
add_test(NAME test_distributed COMMAND test_distributed)
add_test(NAME test_distributed_atexit COMMAND test_distributed_atexit)
