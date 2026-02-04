# Source files for tt_metal llk tests
# Module owners should update this file when adding/removing/renaming source files

set(UNIT_TESTS_LLK_SRC
    test_broadcast.cpp
    test_compute_kernel_sentinel.cpp
    test_copy_block_matmul_partials.cpp
    test_cumsum.cpp
    test_dropout_sfpu_compute.cpp
    test_golden_impls.cpp
    test_mul_reduce_scalar.cpp
    test_pack_rows.cpp
    test_reconfig.cpp
    test_reduce.cpp
    test_sfpu_compute.cpp
    test_single_core_binary_compute.cpp
    test_single_core_matmul_compute.cpp
    test_stochastic_rounding.cpp
    test_single_core_matmul_int8.cpp
    test_transpose.cpp
    test_unary_broadcast.cpp
    test_untilize_tilize.cpp
)
