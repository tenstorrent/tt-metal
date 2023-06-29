import argparse
import time
import random
from pathlib import Path
from itertools import chain
from functools import partial

from loguru import logger

from tests.scripts.common import run_single_test, run_process_and_get_result, report_tests, TestEntry, error_out_if_test_report_has_failures, TestSuiteType, get_git_home_dir_str
from tests.scripts.cmdline_args import get_tt_metal_arguments_from_cmdline_args, get_cmdline_args

TT_METAL_TEST_ENTRIES = (
    TestEntry("tt_metal/tests/test_add_two_ints", "test_add_two_ints"),
    TestEntry("tt_metal/tests/test_bfp8_conversion", "test_bfp8_conversion"),

    TestEntry("tt_metal/tests/test_dram_loopback_single_core", "test_dram_loopback_single_core"),
    TestEntry("tt_metal/tests/test_dram_loopback_single_core_db", "test_dram_loopback_single_core_db"),
    #TestEntry("tt_metal/tests/test_dram_loopback_multi_core", "test_dram_loopback_multi_core"),
    #TestEntry("tt_metal/tests/test_dram_loopback_multi_core_db", "test_dram_loopback_multi_core_db"),
    TestEntry("tt_metal/tests/test_dram_to_l1_multicast", "test_dram_to_l1_multicast"),
    TestEntry("tt_metal/tests/test_dram_to_l1_multicast_loopback_src", "test_dram_to_l1_multicast_loopback_src"),

    TestEntry("tt_metal/tests/test_datacopy", "test_datacopy"),
    TestEntry("tt_metal/tests/test_datacopy", "test_datacopy_bfp8b"),
    TestEntry("tt_metal/tests/test_datacopy_output_in_l1", "test_datacopy_output_in_l1"),
    TestEntry("tt_metal/tests/test_dataflow_cb", "test_dataflow_cb"),
    # TestEntry("tt_metal/tests/test_datacopy_multi_core_multi_dram", "test_datacopy_multi_core_multi_dram"),  TODO: pls fix

    TestEntry("tt_metal/tests/test_eltwise_binary", "test_eltwise_binary"),
    TestEntry("tt_metal/tests/test_bcast", "test_bcast"),

    TestEntry("tt_metal/tests/test_matmul_single_tile", "test_matmul_single_tile"),
    TestEntry("tt_metal/tests/test_matmul_single_tile_bfp8b", "test_matmul_single_tile_bfp8b"),
    TestEntry("tt_metal/tests/test_matmul_single_tile_output_in_l1", "test_matmul_single_tile_output_in_l1"),
    TestEntry("tt_metal/tests/test_matmul_multi_tile", "test_matmul_multi_tile"),
    TestEntry("tt_metal/tests/test_matmul_large_block", "test_matmul_large_block"),
    TestEntry("tt_metal/tests/test_matmul_single_core", "test_matmul_single_core"),
    TestEntry("tt_metal/tests/test_matmul_multi_core_single_dram", "test_matmul_multi_core_single_dram"),
    TestEntry("tt_metal/tests/test_matmul_multi_core_multi_dram", "test_matmul_multi_core_multi_dram"),
    TestEntry("tt_metal/tests/test_matmul_multi_core_multi_dram_in0_mcast", "test_matmul_multi_core_multi_dram_in0_mcast"),
    TestEntry("tt_metal/tests/test_matmul_multi_core_multi_dram_in1_mcast", "test_matmul_multi_core_multi_dram_in1_mcast"),
    TestEntry("tt_metal/tests/test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast", "test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast"),
    TestEntry("tt_metal/tests/test_generic_binary_reader_matmul_large_block", "test_generic_binary_reader_matmul_large_block"),

    TestEntry("tt_metal/tests/test_transpose_hc", "test_transpose_hc"),
    TestEntry("tt_metal/tests/test_transpose_wh", "test_transpose_wh"),
    TestEntry("tt_metal/tests/test_reduce_h", "test_reduce_h"),
    TestEntry("tt_metal/tests/test_reduce_w", "test_reduce_w"),
    TestEntry("tt_metal/tests/test_reduce_hw", "test_reduce_hw"),
    TestEntry("tt_metal/tests/test_bmm", "test_bmm"),
    TestEntry("tt_metal/tests/test_flatten", "test_flatten"),
    TestEntry("tt_metal/tests/test_sfpu", "test_sfpu"),

    TestEntry("tt_metal/tests/test_multiple_programs", "test_multiple_programs"),
    TestEntry("tt_metal/tests/test_multi_core_kernel", "test_multi_core_kernel"),

    TestEntry("tt_metal/tests/test_graph_interpreter", "test_graph_interpreter"),
    TestEntry("tt_metal/tests/test_unpack_tilize", "test_unpack_tilize"),
    TestEntry("tt_metal/tests/test_unpack_untilize", "test_unpack_untilize"),
    TestEntry("tt_metal/tests/test_interleaved_layouts", "test_interleaved_layouts"),
    TestEntry("tt_metal/tests/test_interleaved_l1_buffer", "test_interleaved_l1_buffer"),
    TestEntry("tt_metal/tests/test_dram_copy_sticks_multi_core", "test_dram_copy_sticks_multi_core"),
    TestEntry("tt_metal/tests/test_untilize_eltwise_binary", "test_untilize_eltwise_binary"),
    TestEntry("tt_metal/tests/test_dtx_tilized_row_to_col_major", "test_dtx_tilized_row_to_col_major"),
    #TestEntry("tt_metal/tests/test_l1_to_l1_multi_core", "test_l1_to_l1_multi_core"), // TODO (nshanker): fix this test
    TestEntry("tt_metal/tests/test_dtx", "test_dtx"),

    TestEntry("tt_metal/tests/ops/test_eltwise_binary_op", "ops/test_eltwise_binary_op"),
    TestEntry("tt_metal/tests/ops/test_bcast_op", "ops/test_bcast_op"),
    TestEntry("tt_metal/tests/ops/test_reduce_op", "ops/test_reduce_op"),
    TestEntry("tt_metal/tests/ops/test_transpose_op", "ops/test_transpose_op"),
    TestEntry("tt_metal/tests/ops/test_bmm_op", "ops/test_bmm_op"),
    TestEntry("tt_metal/tests/ops/test_eltwise_unary_op", "ops/test_eltwise_unary_op"),
    TestEntry("tt_metal/tests/ops/test_transpose_wh_single_core", "ops/test_transpose_wh_single_core"),
    TestEntry("tt_metal/tests/ops/test_transpose_wh_multi_core", "ops/test_transpose_wh_multi_core"),
    TestEntry("tt_metal/tests/ops/test_tilize_op", "ops/test_tilize_op"),
    TestEntry("tt_metal/tests/ops/test_tilize_op_channels_last", "ops/test_tilize_op_channels_last"),
    TestEntry("tt_metal/tests/ops/test_tilize_zero_padding", "ops/test_tilize_zero_padding"),
    TestEntry("tt_metal/tests/ops/test_tilize_zero_padding_channels_last", "ops/test_tilize_zero_padding_channels_last"),
    TestEntry("tt_metal/tests/ops/test_layernorm_op", "ops/test_layernorm_op"),
    TestEntry("tt_metal/tests/ops/test_softmax_op", "ops/test_softmax_op"),

    TestEntry("tt_metal/tests/tensors/test_host_device_loopback", "tensors/test_host_device_loopback"),
    TestEntry("tt_metal/tests/tensors/test_copy_and_move", "tensors/test_copy_and_move"),
    TestEntry("tt_metal/tests/test_pipeline_across_rows", "test_pipeline_across_rows"),
    TestEntry("tt_metal/tests/test_core_range_set", "test_core_range_set"),

    # Allocator Tests
    TestEntry("tt_metal/tests/allocator/test_free_list_allocator_algo", "allocator/test_free_list_allocator_algo"),
    TestEntry("tt_metal/tests/allocator/test_l1_banking_allocator", "allocator/test_l1_banking_allocator"),

    # DTX Tests
    TestEntry("tt_metal/tests/dtx/tensor", "dtx/tensor"),
    TestEntry("tt_metal/tests/dtx/unit_tests/", "dtx/unit_tests"),
    TestEntry("tt_metal/tests/dtx/overlap", "dtx/overlap"),
    TestEntry("tt_metal/tests/dtx/collapse_transformations", "dtx/collapse_transformations"),
    #TestEntry("tt_metal/tests/dtx/tensor_evaluate", "dtx/tensor_evaluate"),

    # Compile unit tests
    TestEntry("tt_metal/tests/test_compile_sets_kernel_binaries", "test_compile_sets_kernel_binaries"),
    TestEntry("tt_metal/tests/test_compile_program", "test_compile_program"),
)


PROGRAMMING_EXAMPLE_ENTRIES = (
    TestEntry("programming_examples/loopback", "programming_examples/loopback"),
    TestEntry("programming_examples/eltwise_binary", "programming_examples/eltwise_binary"),
)


def run_single_tt_metal_test(test_entry, timeout):
    run_test = partial(run_single_test, "tt_metal", timeout=timeout)

    print(f"\n\n========== RUNNING TT METAL TEST - {test_entry}")

    return run_test(test_entry)


def run_tt_metal_tests(tt_metal_test_entries, timeout):
    make_test_status_entry = lambda test_entry_: (test_entry_, run_single_tt_metal_test(test_entry_, timeout))

    seed = time.time()

    random.seed(seed)
    random.shuffle(tt_metal_test_entries)
    print(f"SHUFFLED TT METAL TESTS - Using order generated by seed {seed}")

    test_and_status_entries = map(make_test_status_entry, tt_metal_test_entries)

    return dict(test_and_status_entries)


def get_tt_metal_test_entries():
    return list(
        TT_METAL_TEST_ENTRIES
    )


def get_programming_example_entries():
    return list(
        PROGRAMMING_EXAMPLE_ENTRIES
    )


def run_single_programming_example(test_entry, timeout):
    run_test = partial(run_single_test, "programming_example", timeout=timeout, build_full_path_to_test=build_programming_example_executable_path)

    print(f"RUNNING PROGRAMMING EXAMPLE - {test_entry}")

    return run_test(test_entry)


def run_programming_examples(programming_example_entries, timeout):
    make_test_status_entry = lambda test_entry_: (test_entry_, run_single_programming_example(test_entry_, timeout))

    seed = time.time()

    random.seed(seed)
    random.shuffle(programming_example_entries)
    print(f"SHUFFLED PROGRAMMING EXAMPLES - Using order generated by seed {seed}")

    test_and_status_entries = map(make_test_status_entry, programming_example_entries)

    return dict(test_and_status_entries)


def build_programming_example_executable_path(namespace, executable_name, extra_params):
    return Path(f"{get_git_home_dir_str()}/build/{executable_name}")


if __name__ == "__main__":
    cmdline_args = get_cmdline_args(TestSuiteType.TT_METAL)

    timeout, tt_arch = get_tt_metal_arguments_from_cmdline_args(cmdline_args)

    logger.warning("We are not yet parameterizing tt_metal tests on tt_arch")

    programming_example_entries = get_programming_example_entries()

    pe_test_report = run_programming_examples(programming_example_entries, timeout)

    tt_metal_test_entries = get_tt_metal_test_entries()

    llb_test_report = run_tt_metal_tests(tt_metal_test_entries, timeout)

    test_report = {**llb_test_report, **pe_test_report}

    report_tests(test_report)

    error_out_if_test_report_has_failures(test_report)
