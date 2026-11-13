#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Captures and decodes a hung fabric workload's routers, on every rank.

Usage:
  fabric_debug_tool.py [options] -- <the workload's tt-run command, without the program>
  fabric_debug_tool.py [options]

Run it right after the hang, on hardware nothing has reset or used since. Give it the tt-run command the workload was
launched with, minus the program at the end. The tool appends fabric_capture to that command and runs it, then
fabric_decode the same way, so each tool process gets the workload rank's placement and environment. Without a
command, it runs both tools directly, as one process, for a workload that wasn't launched with tt-run.

Every rank is captured under one capture id. Each tool finds its rank's files in the newest run under
TT_METAL_LOGS_PATH (else its working folder), which must be the workload's logs folder.

Exit status: 0 when both steps succeeded, else the failed step's own exit status: fabric_capture's or fabric_decode's,
or tt-run's when it was the launcher. 2 when this tool's own arguments or setup are wrong.
"""

import argparse
import datetime
import os
import pathlib
import shlex
import subprocess
import sys

from fabric_debug_tool_copy import find_capture_directories, get_default_logs_directory, is_capture_id

EXIT_USAGE = 2
# The command-line options that would rerun discovery on the hung chips.
REFUSED_LAUNCHER_ARGS = ("--force-rediscovery",)


def split_launcher_command(argv):
    """This tool's own arguments, and the launcher command after the first `--` (empty without one)."""
    if "--" not in argv:
        return argv, []
    split = argv.index("--")
    return argv[:split], argv[split + 1 :]


def parse_options(own_args):
    parser = argparse.ArgumentParser(
        prog="fabric_debug_tool.py",
        description="Captures and decodes a hung fabric workload's routers, on every rank.",
        usage="%(prog)s [options] [-- <the workload's tt-run command, without the program>]",
    )
    parser.add_argument(
        "--capture-id",
        help="The capture's id, which names its folder in each run's folder. Defaults to the UTC time now",
    )
    parser.add_argument("--rounds", type=int, help="Passed to fabric_capture. Defaults to fabric_capture's")
    parser.add_argument("--round-interval-ms", type=int, help="Passed to fabric_capture. Defaults to fabric_capture's")
    parser.add_argument(
        "--build-dir",
        type=pathlib.Path,
        help="The build whose tools/fabric_capture and tools/fabric_decode to run. Defaults to $TT_METAL_HOME/build",
    )
    return parser.parse_args(own_args)


def fail(message):
    print(f"fabric_debug_tool: {message}", file=sys.stderr)
    sys.exit(EXIT_USAGE)


def find_tools_directory(build_dir):
    """The build's tools folder, with both tools in it."""
    if build_dir is None:
        tt_metal_home = os.environ.get("TT_METAL_HOME")
        if not tt_metal_home:
            fail("TT_METAL_HOME isn't set; set it, or give the build with --build-dir")
        build_dir = pathlib.Path(tt_metal_home) / "build"
    tools_dir = build_dir.resolve() / "tools"
    for tool in ("fabric_capture", "fabric_decode"):
        tool_path = tools_dir / tool
        if not (tool_path.is_file() and os.access(tool_path, os.X_OK)):
            fail(f"{tool_path} isn't there; build it, or give the build that has it with --build-dir")
    return tools_dir


def make_capture_id():
    """The UTC time now, as fabric_capture makes its default id, so capture ids sort by time."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def get_exit_status(returncode):
    """A process's exit status as a shell reports it: 128 + the signal for one a signal ended."""
    return 128 - returncode if returncode < 0 else returncode


def run_step(step, command):
    """Runs one step, with its output passed straight through, and exits with its status when it fails."""
    print(f"fabric_debug_tool: {step}: {shlex.join(command)}", file=sys.stderr, flush=True)
    try:
        status = get_exit_status(subprocess.run(command).returncode)
    except OSError as error:
        # As a shell reports a command it couldn't start: 127 when it isn't there, 126 when it can't run.
        print(f"fabric_debug_tool: {step} couldn't start {command[0]}: {error.strerror}; stopping", file=sys.stderr)
        sys.exit(127 if isinstance(error, FileNotFoundError) else 126)
    if status != 0:
        print(f"fabric_debug_tool: {step} failed with exit status {status}; stopping", file=sys.stderr)
        sys.exit(status)


def print_results(logs_dir, capture_id):
    capture_dirs = find_capture_directories(logs_dir, capture_id)
    print(f"fabric_debug_tool: capture and decode successful, capture id {capture_id}")
    if capture_dirs:
        print("Artifacts stored in:")
        for capture_dir in capture_dirs:
            print(f"  {capture_dir}")
    else:
        print(
            f"Artifacts stored in each rank's {logs_dir}/generated/fabric_debug/runs/<run_id>/captures/{capture_id}/; "
            "none are on this host's"
        )
    # python3, not this interpreter: the copy runs where the results should land, such as outside a job's container.
    copy_script = pathlib.Path(__file__).resolve().parent / "fabric_debug_tool_copy.py"
    copy_command = ["python3", str(copy_script), capture_id, "--logs-path", str(logs_dir)]
    print(f"To copy them to /scratch/$USER/fabric_debug_tool_artifacts/{capture_id}/ (or another folder with --dest):")
    print(f"  {shlex.join(copy_command)}")


def main(argv):
    own_args, launcher = split_launcher_command(argv)
    options = parse_options(own_args)
    for refused in REFUSED_LAUNCHER_ARGS:
        if refused in launcher:
            fail(f"the launcher command has {refused}, which would rediscover the hung chips; leave it out")
    if options.capture_id is not None and not is_capture_id(options.capture_id):
        fail(f"{options.capture_id!r} isn't a capture id: it names one folder")
    tools_dir = find_tools_directory(options.build_dir)

    logs_path = os.environ.get("TT_METAL_LOGS_PATH")
    if not logs_path:
        print(
            "fabric_debug_tool: warning: TT_METAL_LOGS_PATH isn't set, so each rank looks for its files in its "
            "working folder; that must be where the workload wrote them",
            file=sys.stderr,
        )
    logs_dir = get_default_logs_directory()

    capture_id = options.capture_id or make_capture_id()
    capture_args = ["--capture-id", capture_id]
    if options.rounds is not None:
        capture_args += ["--rounds", str(options.rounds)]
    if options.round_interval_ms is not None:
        capture_args += ["--round-interval-ms", str(options.round_interval_ms)]

    run_step("capture", launcher + [str(tools_dir / "fabric_capture")] + capture_args)
    run_step("decode", launcher + [str(tools_dir / "fabric_decode"), "--capture-id", capture_id])
    print_results(logs_dir, capture_id)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
