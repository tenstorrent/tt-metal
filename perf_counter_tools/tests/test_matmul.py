############# Matmul (expect fpu util > 0, sfpu util = 0) ##############################

# padded matmul (low fpu util): ../tools/tracy/profile_this.py -o padded_matmul_profile --profiler-capture-perf-counters=fpu -c 'pytest /localdev/sohaibnadeem/tt-metal/tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_padded_2d_matmul'
# matmul benchmark (very high fpu util): ../tools/tracy/profile_this.py -o padded_matmul_profile --profiler-capture-perf-counters=fpu -c 'pytest /localdev/sohaibnadeem/tt-metal/tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf'

