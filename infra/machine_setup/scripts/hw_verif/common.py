import subprocess as sp
import copy
import os
import argparse


def run_process_and_get_result(command, extra_env={}, capture_output=True):
    full_env = copy.deepcopy(os.environ)
    full_env.update(extra_env)

    result = sp.run(command, shell=True, capture_output=capture_output, env=full_env)

    return result


def get_smi_log_lines():
    tt_smi_result = run_process_and_get_result("tt-smi -s")
    output = tt_smi_result.stdout.decode()
    assert tt_smi_result.returncode == 0

    s_logname = output.split("/")

    logname = s_logname[1]
    logname = logname.split(".log")
    logname = logname[0]

    log_full_path = f"tt-smi-logs/{logname}.log"

    with open(log_full_path) as f:
        log_lines = f.readlines()

    cleanup_result = run_process_and_get_result("rm -rf tt-smi-logs")
    assert cleanup_result.returncode == 0

    return log_lines


def check_not_empty(list):
    if len(list) == 0:
        return False
    return True


def check_same(list):
    for x in list:
        if x != list[0]:
            return False

    return True


class CmdlineArgs:
    argparser = None


def init_cmdline_args_():
    # Singleton
    assert CmdlineArgs.argparser is None, f"You can only create a cmdline_args obj once"

    parser = argparse.ArgumentParser(
        prog="RegressionTests",
        description="Run process-based regression tests",
        conflict_handler="resolve",
    )

    parser.add_argument(
        "--tt-arch",
        default="gs",
        type=str,
        help="Name of silicon arch as a lowercase str, ex. gs, wh_b0 etc.",
    )

    CmdlineArgs.argparser = parser

    return parser


def get_args_from_cmdline_args_(cmdline_args):
    assert (
        cmdline_args == CmdlineArgs
    ), "cmdline_args obj passed does not match expected, something replaced cmdline_args"

    if not cmdline_args.argparser:
        init_cmdline_args_()

    assert (
        cmdline_args.argparser is not None
    ), "get_cmdline_args() should have init'd cmdline_args obj"

    # should be a pure function that you can call over and over
    return CmdlineArgs.argparser.parse_args()


def get_tt_arch_from_cmd_line():
    args = get_args_from_cmdline_args_(CmdlineArgs)

    return args.tt_arch
