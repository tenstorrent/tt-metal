# Source files for ttnn benchmark tests
# Module owners should update this file when adding/removing/renaming source files

set(BENCHMARK_SRCS
    "host_tilizer_untilizer/tilizer_untilizer.cpp"
    "padding/pad_rm.cpp"
    "operations/ternary/benchmark_where.cpp"
    "benchmark_host_alloc_on_tensor_readback.cpp"
    "benchmark_host_dtype_conversion.cpp"
    "matmul/test_matmul_benchmark.cpp"
)
