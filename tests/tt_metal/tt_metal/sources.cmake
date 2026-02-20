# Source files for tt_metal tests
# Module owners should update this file when adding/removing/renaming source files

set(UNIT_TESTS_LEGACY_SRC
    test_add_two_ints.cpp
    test_bcast.cpp
    test_bfp8_conversion.cpp
    test_bmm.cpp
    test_compile_program.cpp
    test_compile_sets_kernel_binaries.cpp
    test_core_range_set.cpp
    test_datacopy.cpp
    test_datacopy_bfp8b.cpp
    test_datacopy_output_in_l1.cpp
    test_dm_loopback.cpp
    test_dram_copy_sticks_multi_core.cpp
    test_dram_loopback_single_core.cpp
    test_eltwise_binary.cpp
    test_generic_binary_reader_matmul_large_block.cpp
    test_interleaved_l1_buffer.cpp
    test_interleaved_layouts.cpp
    test_matmul_single_tile_bfp8b.cpp
    test_matmul_single_tile_output_in_l1.cpp
    test_multi_core_kernel.cpp
    test_multi_dm_add_two_ints.cpp
    test_multiple_programs.cpp
    test_quasar_basic_trisc.cpp
    test_sdpa_reduce_c.cpp
    test_single_dm_l1_write.cpp
    test_stress_noc_mcast.cpp
    test_transpose_hc.cpp
    test_untilize_eltwise_binary.cpp
    test_unaligned_read_write_core.cpp
)
