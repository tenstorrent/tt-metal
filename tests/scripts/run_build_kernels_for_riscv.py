from functools import partial
import concurrent.futures
import argparse

from tests.scripts.common import run_single_test, run_process_and_get_result, report_tests, TestEntry, error_out_if_test_report_has_failures, TestSuiteType
from tests.scripts.cmdline_args import get_cmdline_args, get_build_kernels_for_riscv_arguments_from_cmdline_args

BUILD_KERNELS_FOR_RISCV_TEST_ENTRIES = (
)

def run_compile_tests(timeout, tt_arch):

    run_process_and_get_result("rm -rf built")
    run_single_build_kernels_for_riscv_test = partial(run_single_test, "build_kernels_for_riscv", timeout=timeout, tt_arch=tt_arch)

    make_test_status_entry = lambda test_entry_: (test_entry_, run_single_build_kernels_for_riscv_test(test_entry_))

    test_and_status_entries = map(make_test_status_entry, BUILD_KERNELS_FOR_RISCV_TEST_ENTRIES)
    return dict(test_and_status_entries)

if __name__ == "__main__":
    cmdline_args = get_cmdline_args(TestSuiteType.BUILD_KERNELS_FOR_RISCV)

    timeout, tt_arch = get_build_kernels_for_riscv_arguments_from_cmdline_args(cmdline_args)

    test_report = run_compile_tests(timeout, tt_arch)

    report_tests(test_report)

    error_out_if_test_report_has_failures(test_report)
