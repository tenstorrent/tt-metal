#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Copies one capture's files, and the manifests of each run it captured, to a folder of their own.

Usage: fabric_debug_tool_copy.py <capture_id> [--logs-path <path>] [--dest <dir>]

For each run folder under <logs>/generated/fabric_debug/runs/ that holds captures/<capture_id>/ (one per fabric, so
several when the job is split into sub-contexts), copies the run's manifests and that capture's folder to
<dest>/<capture_id>/<run_id>/, keeping the layout:

  <dest>/<capture_id>/<run_id>/
    fabric_manifest_rank_<r>_of_<n>.json
    captures/<capture_id>/
      fabric_capture_rank_<r>_of_<n>.json, .bin
      fabric_decoded_rank_<r>_of_<n>.json

Run it where <dest> should be, after fabric_debug_tool, with the command it prints. It only reads the logs folder.

Exit status: 0 when it copied the capture, 2 when it copied nothing: no run holds the capture, the destination
already has it, or a copy failed.
"""

import argparse
import getpass
import os
import pathlib
import shutil
import sys

EXIT_FAILED = 2


def get_default_logs_directory():
    """The logs folder tt-metal and the tools use: TT_METAL_LOGS_PATH, else the working folder."""
    return pathlib.Path(os.environ.get("TT_METAL_LOGS_PATH") or os.getcwd()).resolve()


def get_default_destination():
    return pathlib.Path("/scratch") / (os.environ.get("USER") or getpass.getuser()) / "fabric_debug_tool_artifacts"


def make_runs_directory(logs_dir):
    """The folder every fabric run's folder sits in, as fabric_visualizer_files.hpp names it."""
    return logs_dir / "generated" / "fabric_debug" / "runs"


def find_capture_directories(logs_dir, capture_id):
    """The capture's folder in each run folder under the logs folder that has it, by run id."""
    runs_dir = make_runs_directory(logs_dir)
    if not runs_dir.is_dir():
        return []
    run_dirs = sorted(entry for entry in runs_dir.iterdir() if entry.is_dir())
    return [run_dir / "captures" / capture_id for run_dir in run_dirs if (run_dir / "captures" / capture_id).is_dir()]


def fail(message):
    print(f"fabric_debug_tool_copy: {message}", file=sys.stderr)
    sys.exit(EXIT_FAILED)


def is_capture_id(capture_id):
    """Whether `capture_id` can be a capture id: it names one folder, so it can't be a path."""
    return capture_id not in ("", ".", "..") and "/" not in capture_id and os.sep not in capture_id


def copy_run(capture_dir, run_dest):
    """Copies one run's manifests, and the capture's folder in it, to `run_dest`."""
    run_dir = capture_dir.parent.parent
    run_dest.mkdir(parents=True)
    for manifest in sorted(run_dir.glob("fabric_manifest_rank_*.json")):
        shutil.copy2(manifest, run_dest / manifest.name)
    shutil.copytree(capture_dir, run_dest / "captures" / capture_dir.name)


def copy_capture(capture_dirs, dest_dir):
    """Copies every run's files into a temporary folder beside `dest_dir`, then moves it into place, so `dest_dir`
    is never a partial copy."""
    temporary_dir = dest_dir.with_name(f".{dest_dir.name}.tmp.{os.getpid()}")
    try:
        for capture_dir in capture_dirs:
            copy_run(capture_dir, temporary_dir / capture_dir.parent.parent.name)
        os.rename(temporary_dir, dest_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def count_files(directory):
    files = [path for path in directory.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def main(argv):
    parser = argparse.ArgumentParser(
        prog="fabric_debug_tool_copy.py",
        description="Copies one capture's files, and the manifests of each run it captured, to a folder of their own.",
    )
    parser.add_argument("capture_id", help="The capture to copy, as fabric_debug_tool printed it")
    parser.add_argument(
        "--logs-path",
        type=pathlib.Path,
        help="The workload's logs folder. Defaults to TT_METAL_LOGS_PATH, else the working folder",
    )
    parser.add_argument(
        "--dest",
        type=pathlib.Path,
        help="Where the capture's folder goes. Defaults to /scratch/$USER/fabric_debug_tool_artifacts",
    )
    options = parser.parse_args(argv)

    if not is_capture_id(options.capture_id):
        fail(f"{options.capture_id!r} isn't a capture id")
    logs_dir = options.logs_path.resolve() if options.logs_path else get_default_logs_directory()
    dest_dir = (options.dest or get_default_destination()).resolve() / options.capture_id

    capture_dirs = find_capture_directories(logs_dir, options.capture_id)
    if not capture_dirs:
        fail(f"no run under {make_runs_directory(logs_dir)} has capture {options.capture_id}")
    if dest_dir.exists():
        fail(f"{dest_dir} already exists; copy somewhere else with --dest, or remove it first")
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        copy_capture(capture_dirs, dest_dir)
    except shutil.Error as error:
        # copytree's: every file it couldn't copy, as (source, destination, why).
        failures = error.args[0]
        source, _, why = failures[0]
        fail(f"copied nothing: couldn't copy {source} ({why}), and {len(failures) - 1} other file(s)")
    except OSError as error:
        fail(f"copied nothing: {error}")

    num_files, num_bytes = count_files(dest_dir)
    print(f"fabric_debug_tool_copy: copied capture {options.capture_id} from {len(capture_dirs)} run(s) to {dest_dir}")
    print(f"  {num_files} files, {num_bytes / 2**20:.1f} MiB")
    for capture_dir in capture_dirs:
        print(f"  {capture_dir.parent.parent.name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
