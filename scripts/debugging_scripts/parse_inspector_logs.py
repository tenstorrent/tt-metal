#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Script Name: parse_inspector_logs.py

Usage:
    scripts/debugging_scripts/parse_inspector_logs.py [<log-directory>]

Arguments:
    <log-directory>  Path to inspector log directory. Defaults to $TT_METAL_HOME/generated/inspector

Description:
    This script parses inspector logs and transfers them into a structured format.
"""

from dataclasses import dataclass
from functools import cache
import os
import sys
import yaml
from datetime import datetime, timedelta, timezone
from docopt import docopt


# Note: This method is parsing enty by entry and should be used only for debugging large log files.
def fast_parse_yaml_log_file(log_file: str):
    log_entry = ""
    with open(log_file, "r") as f:
        while (line := f.readline()) != "":
            if len(line) > 0 and line[0] != " " and line[0] != "\t" and line[0] != "\n":
                if len(log_entry) > 0:
                    parsed_entry = yaml.safe_load(log_entry)
                    if log_entry[0] == "-":
                        yield parsed_entry[0]
                    else:
                        yield parsed_entry
                log_entry = line
            else:
                log_entry += line
    if len(log_entry) > 0:
        yield yaml.safe_load(log_entry)


@dataclass
class KernelData:
    watcher_kernel_id: int
    name: str
    path: str
    source: str
    program_id: int


@dataclass
class ProgramData:
    id: int
    compiled: bool
    binary_status_per_device: dict[int, str]
    watcher_kernel_ids: list[int]

    def get_device_binary_status(self, device_id: int) -> str:
        return self.binary_status_per_device.get(device_id, "NotSet")


def get_kernels(log_directory: str) -> list[KernelData]:
    yaml_path = os.path.join(log_directory, "kernels.yaml")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    kernels = []
    for entry in data:
        kernel_info = entry.get("kernel", {})
        kernels.append(
            KernelData(
                watcher_kernel_id=int(kernel_info.get("watcher_kernel_id")),
                name=kernel_info.get("name"),
                path=kernel_info.get("path"),
                source=kernel_info.get("source"),
                program_id=int(kernel_info.get("program_id")),
            )
        )
    return kernels


def get_programs(log_directory: str, verbose: bool = False) -> dict[int, ProgramData]:
    yaml_path = os.path.join(log_directory, "programs_log.yaml")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    if verbose:
        print("Programs log:")
        startup_yaml_path = os.path.join(log_directory, "startup.yaml")
        with open(startup_yaml_path, "r") as f:
            startup_data = yaml.safe_load(f)
            for entry, startup_time in startup_data.items():
                assert entry == "startup_time", "Expected 'startup_time' entry in startup.yaml"
                startup_system_clock = (
                    datetime.strptime(startup_time.get("system_clock_iso"), "%Y-%m-%dT%H:%M:%SZ")
                    .replace(tzinfo=timezone.utc)
                    .astimezone()
                )
                startup_clock_ns = int(startup_time.get("high_resolution_clock_ns", 0))
        convert_timestamp = lambda timestamp_ns: startup_system_clock + timedelta(
            microseconds=(timestamp_ns - startup_clock_ns) / 1000
        )
        print_log = lambda timestamp_ns, message: print(
            f"  {convert_timestamp(timestamp_ns).strftime('%Y-%m-%d %H:%M:%S.%f')}: {message}"
        )

    programs: dict[int, ProgramData] = {}
    for entry in data:
        if "program_created" in entry:
            info = entry["program_created"]
            program_id = int(info.get("id"))
            programs[program_id] = ProgramData(
                id=program_id, compiled=False, watcher_kernel_ids=[], binary_status_per_device={}
            )
            if verbose:
                print_log(int(info.get("timestamp_ns")), f"Program {program_id} created")
        elif "program_destroyed" in entry:
            info = entry["program_destroyed"]
            program_id = int(info.get("id"))
            del programs[program_id]
            if verbose:
                print_log(int(info.get("timestamp_ns")), f"Program {program_id} destroyed")
        elif "program_compile_started" in entry:
            info = entry["program_compile_started"]
            program_id = int(info.get("id"))
            if verbose:
                print_log(int(info.get("timestamp_ns")), f"Program {program_id} compile started")
        elif "program_kernel_compile_finished" in entry:
            info = entry["program_kernel_compile_finished"]
            program_id = int(info.get("id"))
            watcher_kernel_id = int(info.get("watcher_kernel_id"))
            programs[program_id].watcher_kernel_ids.append(watcher_kernel_id)
            if verbose:
                print_log(
                    int(info.get("timestamp_ns")),
                    f"Program {program_id} kernel {watcher_kernel_id} compile finished in {info.get('duration_ns')/1000000} ms",
                )
        elif "program_compile_finished" in entry:
            info = entry["program_compile_finished"]
            program_id = int(info.get("id"))
            programs[program_id].compiled = True
            if verbose:
                print_log(
                    int(info.get("timestamp_ns")),
                    f"Program {program_id} compile finished in {info.get('duration_ns')/1000000} ms",
                )
        elif "program_compile_already_exists" in entry:
            info = entry["program_compile_already_exists"]
            program_id = int(info.get("id"))
            programs[program_id].compiled = True
            if verbose:
                print_log(int(info.get("timestamp_ns")), f"Program {program_id} compile already exists")
        elif "program_binary_status_change" in entry:
            info = entry["program_binary_status_change"]
            program_id = int(info.get("id"))
            device_id = int(info.get("device_id"))
            programs[program_id].binary_status_per_device[device_id] = info.get("status")
            if verbose:
                print_log(
                    int(info.get("timestamp_ns")), f"Program {program_id} binary status changed to {info.get('status')}"
                )
    if verbose:
        print()
    return programs


def get_devices_in_use(programs: dict[int, ProgramData]) -> set[int]:
    used_devices = set()
    for program in programs.values():
        # Only include devices with status "Committed"
        committed_devices = {
            device_id for device_id, status in program.binary_status_per_device.items() if status == "Committed"
        }
        used_devices.update(committed_devices)
    return used_devices


class InspectorData:
    def __init__(self, log_directory: str):
        self.log_directory = log_directory

    @cache
    def kernels(self) -> list[KernelData]:
        return get_kernels(self.log_directory)

    def programs(self) -> dict[int, ProgramData]:
        return get_programs(self.log_directory)

    def devices_in_use(self) -> set[int]:
        return get_devices_in_use(self.programs())


@cache
def get_data() -> InspectorData:
    log_directory = os.environ.get("TT_METAL_HOME", "")
    if not log_directory:
        raise ValueError("TT_METAL_HOME environment variable is not set")
    log_directory = os.path.join(log_directory, "generated", "inspector")
    return InspectorData(log_directory)


def main():
    args = docopt(__doc__, argv=sys.argv[1:])
    log_directory = args["<log-directory>"]
    if not log_directory:
        log_directory = os.path.join(os.environ.get("TT_METAL_HOME", ""), "generated", "inspector")

    if not os.path.exists(log_directory):
        print(f"Directory {log_directory} does not exist")
        return

    programs = get_programs(log_directory, verbose=True)
    print("Programs:")
    for program in programs.values():
        print(f"  Program ID {program.id}, compiled: {program.compiled}")
        print(f"    Binary status per device: {program.binary_status_per_device}")
        print(f"    Watcher Kernel IDs: {program.watcher_kernel_ids}")
    print()

    kernels = get_kernels(log_directory)
    print("Kernels:")
    for kernel in kernels:
        print(f"  {kernel.watcher_kernel_id}, pid {kernel.program_id}: {kernel.name} ({kernel.path})")
    devices_in_use = get_devices_in_use(programs)
    print()

    print(f"Devices in use: {devices_in_use}")
    print()


if __name__ == "__main__":
    main()
