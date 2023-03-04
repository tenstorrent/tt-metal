import argparse
import time
import random
from pathlib import Path
from itertools import chain
from functools import partial

from reg_scripts.common import run_single_test, run_process_and_get_result, report_tests, TestEntry, error_out_if_test_report_has_failures, TestSuiteType, get_git_home_dir_str
from reg_scripts.cmdline_args import get_ll_buda_arguments_from_cmdline_args, get_cmdline_args

LL_BUDA_TEST_ENTRIES = (
    TestEntry("ll_buda/tests/test_add_two_ints", "test_add_two_ints"),

    TestEntry("ll_buda/tests/test_dram_loopback_single_core", "test_dram_loopback_single_core"),
    TestEntry("ll_buda/tests/test_dram_loopback_single_core_db", "test_dram_loopback_single_core_db"),
    #TestEntry("ll_buda/tests/test_dram_loopback_multi_core", "test_dram_loopback_multi_core"),
    #TestEntry("ll_buda/tests/test_dram_loopback_multi_core_db", "test_dram_loopback_multi_core_db"),
    TestEntry("ll_buda/tests/test_dram_to_l1_multicast", "test_dram_to_l1_multicast"),
    TestEntry("ll_buda/tests/test_dram_to_l1_multicast_loopback_src", "test_dram_to_l1_multicast_loopback_src"),

    TestEntry("ll_buda/tests/test_datacopy", "test_datacopy"),
    TestEntry("ll_buda/tests/test_dataflow_cb", "test_dataflow_cb"),

    TestEntry("ll_buda/tests/test_eltwise_binary", "test_eltwise_binary"),
    TestEntry("ll_buda/tests/test_bcast", "test_bcast"),

    TestEntry("ll_buda/tests/test_matmul_single_tile", "test_matmul_single_tile"),
    TestEntry("ll_buda/tests/test_matmul_multi_tile", "test_matmul_multi_tile"),
    TestEntry("ll_buda/tests/test_matmul_large_block", "test_matmul_large_block"),
    TestEntry("ll_buda/tests/test_matmul_single_core", "test_matmul_single_core"),
    TestEntry("ll_buda/tests/test_matmul_multi_core_single_dram", "test_matmul_multi_core_single_dram"),
    TestEntry("ll_buda/tests/test_matmul_multi_core_multi_dram", "test_matmul_multi_core_multi_dram"),
    TestEntry("ll_buda/tests/test_matmul_multi_core_multi_dram_in0_mcast", "test_matmul_multi_core_multi_dram_in0_mcast"),
    TestEntry("ll_buda/tests/test_matmul_multi_core_multi_dram_in1_mcast", "test_matmul_multi_core_multi_dram_in1_mcast"),
    TestEntry("ll_buda/tests/test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast", "test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast"),
    TestEntry("ll_buda/tests/test_generic_binary_reader_matmul_large_block", "test_generic_binary_reader_matmul_large_block"),
    TestEntry("ll_buda/tests/test_3x3conv_as_matmul_large_block", "test_3x3conv_as_matmul_large_block"),

    TestEntry("ll_buda/tests/test_transpose_hc", "test_transpose_hc"),
    TestEntry("ll_buda/tests/test_transpose_wh", "test_transpose_wh"),
    TestEntry("ll_buda/tests/test_reduce_h", "test_reduce_h"),
    TestEntry("ll_buda/tests/test_reduce_w", "test_reduce_w"),
    TestEntry("ll_buda/tests/test_reduce_hw", "test_reduce_hw"),
    TestEntry("ll_buda/tests/test_bmm", "test_bmm"),
    TestEntry("ll_buda/tests/test_flatten", "test_flatten"),
    TestEntry("ll_buda/tests/test_sfpu", "test_sfpu"),

    TestEntry("ll_buda/tests/test_multiple_programs", "test_multiple_programs"),
    TestEntry("ll_buda/tests/test_multi_core_kernel", "test_multi_core_kernel"),

    TestEntry("ll_buda/tests/test_graph_interpreter", "test_graph_interpreter"),
    TestEntry("ll_buda/tests/test_unpack_tilize", "test_unpack_tilize"),
    TestEntry("ll_buda/tests/test_unpack_untilize", "test_unpack_untilize"),
    TestEntry("ll_buda/tests/test_interleaved_layouts", "test_interleaved_layouts"),

    TestEntry("ll_buda/tests/ops/test_eltwise_binary_op", "ops/test_eltwise_binary_op"),
    TestEntry("ll_buda/tests/ops/test_bcast_op", "ops/test_bcast_op"),
    TestEntry("ll_buda/tests/ops/test_reduce_op", "ops/test_reduce_op"),
    TestEntry("ll_buda/tests/ops/test_transpose_op", "ops/test_transpose_op"),
    TestEntry("ll_buda/tests/ops/test_bmm_op", "ops/test_bmm_op"),
    TestEntry("ll_buda/tests/ops/test_eltwise_unary_op", "ops/test_eltwise_unary_op"),

    TestEntry("ll_buda/tests/tensors/test_host_device_loopback", "tensors/test_host_device_loopback"),

    # DTX Tests
    TestEntry("ll_buda/tests/dtx/tensor", "dtx/tensor"),
    TestEntry("ll_buda/tests/dtx/unit_tests/", "dtx/unit_tests"),
    TestEntry("ll_buda/tests/dtx/overlap", "dtx/overlap"),
    TestEntry("ll_buda/tests/dtx/collapse_transformations", "dtx/collapse_transformations"),
    #TestEntry("ll_buda/tests/dtx/tensor_evaluate", "dtx/tensor_evaluate"),


)


PROGRAMMING_EXAMPLE_ENTRIES = (
    TestEntry("programming_examples/loopback", "programming_examples/loopback"),
)


def run_single_ll_buda_test(test_entry, timeout):
    run_test = partial(run_single_test, "ll_buda", timeout=timeout)

    print(f"RUNNING LL BUDA TEST - {test_entry}")

    return run_test(test_entry)


def run_ll_buda_tests(ll_buda_test_entries, timeout):
    make_test_status_entry = lambda test_entry_: (test_entry_, run_single_ll_buda_test(test_entry_, timeout))

    seed = time.time()

    random.seed(seed)
    random.shuffle(ll_buda_test_entries)
    print(f"SHUFFLED LL BUDA TESTS - Using order generated by seed {seed}")

    test_and_status_entries = map(make_test_status_entry, ll_buda_test_entries)

    return dict(test_and_status_entries)


def get_ll_buda_test_entries():
    return list(
        LL_BUDA_TEST_ENTRIES
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
    return Path(f"{get_git_home_dir_str()}/build/test/{executable_name}")


if __name__ == "__main__":
    ll_buda_test_entries = get_ll_buda_test_entries()

    cmdline_args = get_cmdline_args(TestSuiteType.LL_BUDA)

    timeout, = get_ll_buda_arguments_from_cmdline_args(cmdline_args)

    llb_test_report = run_ll_buda_tests(ll_buda_test_entries, timeout)

    programming_example_entries = get_programming_example_entries()

    pe_test_report = run_programming_examples(programming_example_entries, timeout)

    test_report = {**llb_test_report, **pe_test_report}

    report_tests(test_report)

    error_out_if_test_report_has_failures(test_report)
