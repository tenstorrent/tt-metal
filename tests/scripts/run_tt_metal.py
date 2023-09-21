# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import time
import random
from pathlib import Path
from itertools import chain
from functools import partial

from loguru import logger

from tests.scripts.common import (
    run_single_test,
    run_process_and_get_result,
    report_tests,
    TestEntry,
    error_out_if_test_report_has_failures,
    TestSuiteType,
    get_git_home_dir_str,
    filter_empty,
    void_for_whb0,
)
from tests.scripts.cmdline_args import (
    get_tt_metal_arguments_from_cmdline_args,
    get_cmdline_args,
)

TT_METAL_SLOW_DISPATCH_TEST_ENTRIES = (
    TestEntry("tt_metal/tests/test_add_two_ints", "test_add_two_ints"),
    TestEntry("tt_metal/tests/test_bfp8_conversion", "test_bfp8_conversion"),
    TestEntry(
        "tt_metal/tests/test_dram_loopback_single_core",
        "test_dram_loopback_single_core",
    ),
    TestEntry(
        "tt_metal/tests/test_dram_loopback_single_core_db",
        "test_dram_loopback_single_core_db",
    ),
    # TestEntry("tt_metal/tests/test_dram_loopback_multi_core", "test_dram_loopback_multi_core"),
    # TestEntry("tt_metal/tests/test_dram_loopback_multi_core_db", "test_dram_loopback_multi_core_db"),
    TestEntry("tt_metal/tests/test_dram_to_l1_multicast", "test_dram_to_l1_multicast"),
    TestEntry(
        "tt_metal/tests/test_dram_to_l1_multicast_loopback_src",
        "test_dram_to_l1_multicast_loopback_src",
    ),
    TestEntry("tt_metal/tests/test_datacopy", "test_datacopy"),
    TestEntry("tt_metal/tests/test_datacopy", "test_datacopy_bfp8b"),
    TestEntry(
        "tt_metal/tests/test_datacopy_output_in_l1", "test_datacopy_output_in_l1"
    ),
    TestEntry("tt_metal/tests/test_dataflow_cb", "test_dataflow_cb"),
    # TestEntry("tt_metal/tests/test_datacopy_multi_core_multi_dram", "test_datacopy_multi_core_multi_dram"),  TODO: pls fix
    TestEntry("tt_metal/tests/test_bcast", "test_bcast"),
    TestEntry("tt_metal/tests/test_matmul_single_tile", "test_matmul_single_tile"),
    TestEntry(
        "tt_metal/tests/test_matmul_single_tile_bfp8b", "test_matmul_single_tile_bfp8b"
    ),
    TestEntry(
        "tt_metal/tests/test_matmul_single_tile_output_in_l1",
        "test_matmul_single_tile_output_in_l1",
    ),
    TestEntry("tt_metal/tests/test_matmul_multi_tile", "test_matmul_multi_tile"),
    void_for_whb0(
        TestEntry("tt_metal/tests/test_matmul_large_block", "test_matmul_large_block")
    ),
    TestEntry("tt_metal/tests/test_matmul_single_core", "test_matmul_single_core"),
    TestEntry(
        "tt_metal/tests/test_matmul_single_core_small", "test_matmul_single_core_small"
    ),
    void_for_whb0(
        TestEntry(
            "tt_metal/tests/test_matmul_multi_core_single_dram",
            "test_matmul_multi_core_single_dram",
        )
    ),
    void_for_whb0(
        TestEntry(
            "tt_metal/tests/test_matmul_multi_core_multi_dram_in0_mcast",
            "test_matmul_multi_core_multi_dram_in0_mcast",
        )
    ),
    void_for_whb0(
        TestEntry(
            "tt_metal/tests/test_matmul_multi_core_multi_dram_in1_mcast",
            "test_matmul_multi_core_multi_dram_in1_mcast",
        )
    ),
    void_for_whb0(
        TestEntry(
            "tt_metal/tests/test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast",
            "test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast",
        )
    ),
    TestEntry(
        "tt_metal/tests/test_generic_binary_reader_matmul_large_block",
        "test_generic_binary_reader_matmul_large_block",
    ),
    TestEntry("tt_metal/tests/test_transpose_hc", "test_transpose_hc"),
    void_for_whb0(TestEntry("tt_metal/tests/test_transpose_wh", "test_transpose_wh")),
    TestEntry("tt_metal/tests/test_reduce_h", "test_reduce_h"),
    TestEntry("tt_metal/tests/test_reduce_w", "test_reduce_w"),
    TestEntry("tt_metal/tests/test_reduce_hw", "test_reduce_hw"),
    TestEntry("tt_metal/tests/test_bmm", "test_bmm"),
    TestEntry("tt_metal/tests/test_flatten", "test_flatten"),
    TestEntry("tt_metal/tests/test_multiple_programs", "test_multiple_programs"),
    TestEntry("tt_metal/tests/test_multi_core_kernel", "test_multi_core_kernel"),
    TestEntry("tt_metal/tests/test_graph_interpreter", "test_graph_interpreter"),
    TestEntry("tt_metal/tests/test_unpack_tilize", "test_unpack_tilize"),
    TestEntry("tt_metal/tests/test_unpack_untilize", "test_unpack_untilize"),
    TestEntry("tt_metal/tests/test_interleaved_layouts", "test_interleaved_layouts"),
    TestEntry(
        "tt_metal/tests/test_interleaved_l1_buffer", "test_interleaved_l1_buffer"
    ),
    TestEntry(
        "tt_metal/tests/test_dram_copy_sticks_multi_core",
        "test_dram_copy_sticks_multi_core",
    ),
    void_for_whb0(
        TestEntry(
            "tt_metal/tests/test_untilize_eltwise_binary",
            "test_untilize_eltwise_binary",
        )
    ),
    # TestEntry("tt_metal/tests/test_l1_to_l1_multi_core", "test_l1_to_l1_multi_core"), // TODO (nshanker): fix this test
    TestEntry(
        "tt_metal/tests/test_pipeline_across_rows", "test_pipeline_across_rows"
    ),
    TestEntry("tt_metal/tests/test_core_range_set", "test_core_range_set"),
    # Allocator Tests
    TestEntry(
        "tt_metal/tests/allocator/test_free_list_allocator_algo",
        "allocator/test_free_list_allocator_algo",
    ),
    TestEntry(
        "tt_metal/tests/allocator/test_l1_banking_allocator",
        "allocator/test_l1_banking_allocator",
    ),
    # Compile unit tests
    TestEntry(
        "tt_metal/tests/test_compile_sets_kernel_binaries",
        "test_compile_sets_kernel_binaries",
    ),
    TestEntry("tt_metal/tests/test_compile_program", "test_compile_program"),
)

TT_METAL_FAST_DISPATCH_TEST_ENTRIES = (
    TestEntry("tt_metal/tests/test_eltwise_binary", "test_eltwise_binary"),
    TestEntry(
        "tt_metal/tests/test_matmul_multi_core_multi_dram",
        "test_matmul_multi_core_multi_dram",
    ),
)

TT_METAL_COMMON_TEST_ENTRIES = (
    # Allocator Tests
    TestEntry(
        "tt_metal/tests/allocator/test_free_list_allocator_algo",
        "allocator/test_free_list_allocator_algo",
    ),
    TestEntry(
        "tt_metal/tests/allocator/test_l1_banking_allocator",
        "allocator/test_l1_banking_allocator",
    ),
    # Compile unit tests
    TestEntry(
        "tt_metal/tests/test_compile_sets_kernel_binaries",
        "test_compile_sets_kernel_binaries",
    ),
    TestEntry("tt_metal/tests/test_compile_program", "test_compile_program"),
)


PROGRAMMING_EXAMPLE_ENTRIES = (
    TestEntry("programming_examples/loopback", "programming_examples/loopback"),
    TestEntry(
        "programming_examples/eltwise_binary", "programming_examples/eltwise_binary"
    ),
)


def run_single_tt_metal_test(test_entry, timeout):
    run_test = partial(run_single_test, "tt_metal", timeout=timeout)

    logger.info(f"========= RUNNING TT METAL TEST - {test_entry}")

    return run_test(test_entry)


def run_tt_cpp_tests(test_entries, timeout, run_single_test):
    make_test_status_entry = lambda test_entry_: (
        test_entry_,
        run_single_test(test_entry_, timeout),
    )

    seed = time.time()

    random.seed(seed)
    random.shuffle(test_entries)
    logger.info(f"SHUFFLED CPP TESTS - Using order generated by seed {seed}")

    test_and_status_entries = map(make_test_status_entry, test_entries)

    return dict(test_and_status_entries)


@filter_empty
def get_tt_metal_fast_dispatch_test_entries():
    return list(TT_METAL_COMMON_TEST_ENTRIES) + list(
        TT_METAL_FAST_DISPATCH_TEST_ENTRIES
    )


@filter_empty
def get_tt_metal_slow_dispatch_test_entries():
    return list(TT_METAL_COMMON_TEST_ENTRIES) + list(
        TT_METAL_SLOW_DISPATCH_TEST_ENTRIES
    )


@filter_empty
def get_programming_example_entries():
    return list(PROGRAMMING_EXAMPLE_ENTRIES)


def run_single_programming_example(test_entry, timeout):
    run_test = partial(
        run_single_test,
        "programming_example",
        timeout=timeout,
        build_full_path_to_test=build_programming_example_executable_path,
    )

    logger.info(f"RUNNING PROGRAMMING EXAMPLE - {test_entry}")

    return run_test(test_entry)


def run_programming_examples(programming_example_entries, timeout):
    make_test_status_entry = lambda test_entry_: (
        test_entry_,
        run_single_programming_example(test_entry_, timeout),
    )

    seed = time.time()

    random.seed(seed)
    random.shuffle(programming_example_entries)
    logger.info(f"SHUFFLED PROGRAMMING EXAMPLES - Using order generated by seed {seed}")

    test_and_status_entries = map(make_test_status_entry, programming_example_entries)

    return dict(test_and_status_entries)


def build_programming_example_executable_path(namespace, executable_name, extra_params):
    return Path(f"{get_git_home_dir_str()}/build/{executable_name}")


if __name__ == "__main__":
    cmdline_args = get_cmdline_args(TestSuiteType.TT_METAL)

    timeout, tt_arch, dispatch_mode = get_tt_metal_arguments_from_cmdline_args(
        cmdline_args
    )

    logger.warning("We are not yet parameterizing tt_metal tests on tt_arch")

    pe_test_report = {}
    if dispatch_mode == "slow":
        logger.info("Running Programming Example tests")
        programming_example_entries = get_programming_example_entries()
        pe_test_report = run_programming_examples(programming_example_entries, timeout)
        logger.info("Running slow-dispatch mode tests")
        tt_metal_test_entries = get_tt_metal_slow_dispatch_test_entries()
    else:
        logger.info("Running fast-dispatch mode tests")
        tt_metal_test_entries = get_tt_metal_fast_dispatch_test_entries()

    llb_test_report = run_tt_cpp_tests(
        tt_metal_test_entries, timeout, run_single_tt_metal_test
    )

    test_report = {**llb_test_report, **pe_test_report}

    report_tests(test_report)

    error_out_if_test_report_has_failures(test_report)
