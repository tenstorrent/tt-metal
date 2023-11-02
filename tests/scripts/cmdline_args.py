# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
from functools import partial

from tests.scripts.common import TestSuiteType


class CmdlineArgs:
    argparser = None


def get_args_from_cmdline_args_(cmdline_args):
    assert cmdline_args == CmdlineArgs, "cmdline_args obj passed does not match expected, use get_cmdline_args()"

    assert cmdline_args.argparser is not None, "get_cmdline_args() should have init'd cmdline_args obj"

    # should be a pure function that you can call over and over
    return CmdlineArgs.argparser.parse_args()


def add_test_type_specific_args_(argparser, test_suite_type=TestSuiteType.UNKNOWN):
    if test_suite_type == TestSuiteType.LLRT:
        argparser.add_argument(
            "--short-driver-tests",
            action="store_true",
            default=False,
            help="Use short-running silicon driver tests instead",
        )
        # Set to 20 minutes for long silicon driver tests
        argparser.add_argument("--timeout", default=1200, type=int, help="Timeout in seconds for each test")
    elif test_suite_type == TestSuiteType.BUILD_KERNELS_FOR_RISCV:
        pass
    elif test_suite_type == TestSuiteType.TT_METAL:
        argparser.add_argument(
            "--dispatch-mode", default="fast", type=str, help="Dispatch mode for tests list differentiation"
        )
        pass
    elif test_suite_type == TestSuiteType.TT_EAGER:
        argparser.add_argument(
            "--dispatch-mode", default="fast", type=str, help="Dispatch mode for tests list differentiation"
        )
        pass
    else:
        raise Exception("You must specify a test type")

    return argparser


def add_common_args_(argparser):
    argparser.add_argument("--timeout", default=600, type=int, help="Timeout in seconds for each test")
    argparser.add_argument(
        "--tt-arch",
        default="grayskull",
        type=str,
        help="Name of silicon arch as a lowercase str, ex. grayskull, wormhole_b0",
    )

    return argparser


def get_cmdline_args(test_suite_type=TestSuiteType.UNKNOWN):
    # Singleton
    assert CmdlineArgs.argparser is None, f"You can only create a cmdline_args obj once"

    parser = argparse.ArgumentParser(
        prog="RegressionTests",
        description="Run process-based regression tests",
        conflict_handler="resolve",
    )

    parser = add_common_args_(parser)
    parser = add_test_type_specific_args_(parser, test_suite_type)

    CmdlineArgs.argparser = parser

    return CmdlineArgs


def get_full_arg_list_with_specific_args_(get_specific_args, cmdline_args):
    parsed_args = get_args_from_cmdline_args_(cmdline_args)

    return (parsed_args.timeout, parsed_args.tt_arch) + get_specific_args(parsed_args)


def get_empty_args_from_parsed_args_(parsed_args):
    return tuple()


def get_llrt_specific_args_from_parsed_args_(parsed_args):
    return (parsed_args.short_driver_tests,)


def get_tt_metal_specific_args_from_parsed_args_(parsed_args):
    return (parsed_args.dispatch_mode,)


get_llrt_arguments_from_cmdline_args = partial(
    get_full_arg_list_with_specific_args_, get_llrt_specific_args_from_parsed_args_
)
get_build_kernels_for_riscv_arguments_from_cmdline_args = partial(
    get_full_arg_list_with_specific_args_, get_empty_args_from_parsed_args_
)
get_tt_metal_arguments_from_cmdline_args = partial(
    get_full_arg_list_with_specific_args_, get_tt_metal_specific_args_from_parsed_args_
)
