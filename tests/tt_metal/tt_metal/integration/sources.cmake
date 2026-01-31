# Source files for tt_metal integration tests
# Module owners should update this file when adding/removing/renaming source files

set(UNIT_TESTS_INTEGRATION_SRC
    test_basic_pipeline.cpp
    test_flatten.cpp
    test_sfpu_compute.cpp
    matmul/test_matmul_large_block.cpp
    matmul/test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast.cpp
    matmul/test_matmul_multi_core_multi_dram_inX_mcast.cpp
    matmul/test_matmul_multi_core_X_dram.cpp
    matmul/test_matmul_single_core.cpp
    matmul/test_matmul_X_tile.cpp
    vecadd/test_vecadd_multi_core.cpp
)
