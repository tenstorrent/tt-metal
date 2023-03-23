from functools import partial
import concurrent.futures
import argparse

from tests.reg_scripts.common import run_single_test, run_process_and_get_result, report_tests, TestEntry, error_out_if_test_report_has_failures, TestSuiteType
from tests.reg_scripts.cmdline_args import get_cmdline_args, get_build_kernels_for_riscv_arguments_from_cmdline_args

BUILD_KERNELS_FOR_RISCV_TEST_ENTRIES = (
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_add_two_ints", "test_build_kernel_add_two_ints"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_dram_copy", "test_build_kernel_dram_copy"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_dram_copy_ncrisc", "test_build_kernel_dram_copy_ncrisc"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_dram_copy_brisc_ncrisc", "test_build_kernel_dram_copy_brisc_ncrisc"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_dram_copy_looped", "test_build_kernel_dram_copy_looped"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_blank", "test_build_kernel_blank"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_datacopy", "test_build_kernel_datacopy"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_dram_to_l1_copy", "test_build_kernel_dram_to_l1_copy"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_l1_to_dram_copy", "test_build_kernel_l1_to_dram_copy"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_copy_pattern", "test_build_kernel_copy_pattern"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_copy_pattern_tilized", "test_build_kernel_copy_pattern_tilized"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_loader_sync", "test_build_kernel_loader_sync"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_loader_sync_db", "test_build_kernel_loader_sync_db"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_eltwise_sync", "test_build_kernel_eltwise_sync"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_remote_read_remote_write_sync", "test_build_kernel_remote_read_remote_write_sync"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_remote_read_remote_write_sync_db", "test_build_kernel_remote_read_remote_write_sync_db"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_risc_read_speed", "test_build_kernel_risc_read_speed"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_risc_write_speed", "test_build_kernel_risc_write_speed"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_risc_rw_speed_banked_dram", "test_build_kernel_risc_rw_speed_banked_dram"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_dataflow_cb_test", "test_build_kernel_dataflow_cb_test"),

    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_transpose_hc", "test_build_kernel_transpose_hc"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_test_debug_print", "test_build_kernel_test_debug_print"),
    TestEntry("build_kernels_for_riscv/tests/test_build_kernel_datacopy_switched_riscs", "test_build_kernel_datacopy_switched_riscs"),
)

def run_compile_tests(num_processes, timeout):

    run_process_and_get_result("rm -rf built_kernels")
    run_single_build_kernels_for_riscv_test = partial(run_single_test, "build_kernels_for_riscv", timeout=timeout)

    if num_processes > 1:
        # clamp the pool to number of inputs
        num_processes = min(num_processes, len(BUILD_KERNELS_FOR_RISCV_TEST_ENTRIES))
        futures = []
        inputs = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            for test_entry in BUILD_KERNELS_FOR_RISCV_TEST_ENTRIES:
                future = executor.submit(run_single_build_kernels_for_riscv_test, test_entry)
                inputs.append(test_entry)
                futures.append(future)
        result = dict(zip(inputs, [f.result() for f in futures]))
        return result
    else:
        # support the old single-process path without executor for debugging
        make_test_status_entry = lambda test_entry_: (test_entry_, run_single_build_kernels_for_riscv_test(test_entry_))

        test_and_status_entries = map(make_test_status_entry, BUILD_KERNELS_FOR_RISCV_TEST_ENTRIES)
        return dict(test_and_status_entries)

if __name__ == "__main__":
    cmdline_args = get_cmdline_args(TestSuiteType.BUILD_KERNELS_FOR_RISCV)

    timeout, num_processes = get_build_kernels_for_riscv_arguments_from_cmdline_args(cmdline_args)

    test_report = run_compile_tests(num_processes, timeout)

    report_tests(test_report)

    error_out_if_test_report_has_failures(test_report)
