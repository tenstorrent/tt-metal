import argparse
from functools import partial

from tests.reg_scripts.common import TestSuiteType


class CmdlineArgs:
    argparser = None


def get_args_from_cmdline_args_(cmdline_args):
    assert cmdline_args == CmdlineArgs, "cmdline_args obj passed does not match expected, use get_cmdline_args()"

    assert cmdline_args.argparser is not None, "get_cmdline_args() should have init'd cmdline_args obj"

    # should be a pure function that you can call over and over
    return CmdlineArgs.argparser.parse_args()


def add_test_type_specific_args_(argparser, test_suite_type=TestSuiteType.UNKNOWN):
    if test_suite_type == TestSuiteType.BUILD_KERNELS_FOR_RISCV:
        argparser.add_argument("-j", help="Use specified number of processes", dest="num_processes", type=int, default=1)
        argparser.add_argument("--num_processes", help="Use specified number of processes", dest="num_processes", type=int, default=1)
    elif test_suite_type == TestSuiteType.LLRT:
        argparser.add_argument("--skip-driver-tests", action="store_true", default=False, help="Skip long-running silicon driver tests")
    elif test_suite_type == TestSuiteType.TT_METAL:
        pass
    else:
        raise Exception("You must specify a test type")

    return argparser


def add_common_args_(argparser):
    argparser.add_argument("--timeout", default=600, type=int, help="Timeout in seconds for each test")

    return argparser


def get_cmdline_args(test_suite_type=TestSuiteType.UNKNOWN):
    # Singleton
    assert CmdlineArgs.argparser is None, f"You can only create a cmdline_args obj once"

    parser = argparse.ArgumentParser(
        prog="RegressionTests",
        description="Run process-based regression tests",
    )

    parser = add_test_type_specific_args_(parser, test_suite_type)
    parser = add_common_args_(parser)

    CmdlineArgs.argparser = parser

    return CmdlineArgs


def get_full_arg_list_with_specific_args_(get_specific_args, cmdline_args):
    parsed_args = get_args_from_cmdline_args_(cmdline_args)

    return (parsed_args.timeout,) + get_specific_args(parsed_args)


def get_empty_args_from_parsed_args_(parsed_args):
    return tuple()


def get_llrt_specific_args_from_parsed_args_(parsed_args):
    return (parsed_args.skip_driver_tests,)


def get_build_kernels_for_riscv_specific_args_from_parsed_args_(parsed_args):
    return (parsed_args.num_processes,)


get_llrt_arguments_from_cmdline_args = partial(get_full_arg_list_with_specific_args_, get_llrt_specific_args_from_parsed_args_)
get_build_kernels_for_riscv_arguments_from_cmdline_args = partial(get_full_arg_list_with_specific_args_, get_build_kernels_for_riscv_specific_args_from_parsed_args_)
get_tt_metal_arguments_from_cmdline_args = partial(get_full_arg_list_with_specific_args_, get_empty_args_from_parsed_args_)
