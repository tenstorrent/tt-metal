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
from functools import cache, cached_property
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


def read_yaml(yaml_path: str):
    try:
        # Try to use ryml for faster parsing if available
        import ryml
        from ttexalens.util import ryml_to_lazy

        with open(yaml_path, "r") as f:
            content = f.read()
            tree = ryml.parse_in_arena(content)
            data = ryml_to_lazy(tree, tree.root_id())
    except:
        # Fallback to standard yaml library
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
    if data is None:
        return []
    return data


@dataclass
class StartupData:
    startup_system_clock: datetime
    startup_clock_ns: int

    def convert_timestamp(self, timestamp_ns: int) -> datetime:
        return self.startup_system_clock + timedelta(microseconds=(timestamp_ns - self.startup_clock_ns) / 1000)

    def print_log(self, timestamp_ns: int, message: str):
        print(f"  {self.convert_timestamp(timestamp_ns).strftime('%Y-%m-%d %H:%M:%S.%f')}: {message}")


@dataclass
class KernelData:
    watcher_kernel_id: int
    name: str
    path: str
    source: str
    program_id: int


@dataclass
class MeshCoordinate:
    coordinates: list[int]


@dataclass
class MeshDeviceData:
    mesh_id: int
    devices: list[int]
    shape: list[int]
    parent_mesh_id: int | None = None
    initialized: bool = False

    def get_device_id(self, coordinate: MeshCoordinate) -> int:
        assert len(coordinate.coordinates) == len(
            self.shape
        ), f"Coordinate {coordinate.coordinates} does not match mesh shape {self.shape}"
        linear_index = 0
        for dim in range(len(coordinate.coordinates)):
            linear_index = linear_index + coordinate.coordinates[dim] * self.shape[dim]
        return self.devices[linear_index]


@dataclass
class MeshWorkloadProgramData:
    program_id: int
    coordinates: list[MeshCoordinate]


@dataclass
class MeshWorkloadData:
    mesh_workload_id: int
    programs: list[MeshWorkloadProgramData]
    binary_status_per_mesh_device: dict[int, str]

    def get_device_binary_status(self, mesh_id: int) -> str:
        return self.binary_status_per_mesh_device.get(mesh_id, "NotSet")


@dataclass
class ProgramData:
    id: int
    compiled: bool
    binary_status_per_device: dict[int, str]
    watcher_kernel_ids: list[int]

    def get_device_binary_status(self, device_id: int) -> str:
        return self.binary_status_per_device.get(device_id, "NotSet")


def get_kernels(log_directory: str) -> dict[int, KernelData]:
    yaml_path = os.path.join(log_directory, "kernels.yaml")
    data = read_yaml(yaml_path)

    kernels: dict[int, KernelData] = {}
    for entry in data:
        kernel_info = entry.get("kernel", {})
        kernel_data = KernelData(
            watcher_kernel_id=int(kernel_info.get("watcher_kernel_id")),
            name=kernel_info.get("name"),
            path=kernel_info.get("path"),
            source=kernel_info.get("source"),
            program_id=int(kernel_info.get("program_id")),
        )
        kernels[kernel_data.watcher_kernel_id] = kernel_data
    return kernels


def get_startup_data(log_directory: str) -> StartupData:
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
            return StartupData(
                startup_system_clock=startup_system_clock,
                startup_clock_ns=startup_clock_ns,
            )
    raise ValueError("No startup time found in startup.yaml")


def get_programs(log_directory: str, verbose: bool = False) -> dict[int, ProgramData]:
    yaml_path = os.path.join(log_directory, "programs_log.yaml")
    data = read_yaml(yaml_path)
    if verbose:
        print("Programs log:")
        startup = get_startup_data(log_directory)

    programs: dict[int, ProgramData] = {}
    for entry in data:
        if "program_created" in entry:
            info = entry["program_created"]
            program_id = int(info.get("id"))
            programs[program_id] = ProgramData(
                id=program_id, compiled=False, watcher_kernel_ids=[], binary_status_per_device={}
            )
            if verbose:
                startup.print_log(int(info.get("timestamp_ns")), f"Program {program_id} created")
        elif "program_destroyed" in entry:
            info = entry["program_destroyed"]
            program_id = int(info.get("id"))
            if program_id in programs:
                del programs[program_id]
            if verbose:
                startup.print_log(int(info.get("timestamp_ns")), f"Program {program_id} destroyed")
        elif "program_compile_started" in entry:
            info = entry["program_compile_started"]
            program_id = int(info.get("id"))
            if verbose:
                startup.print_log(int(info.get("timestamp_ns")), f"Program {program_id} compile started")
        elif "program_kernel_compile_finished" in entry:
            info = entry["program_kernel_compile_finished"]
            program_id = int(info.get("id"))
            watcher_kernel_id = int(info.get("watcher_kernel_id"))
            if program_id in programs:
                programs[program_id].watcher_kernel_ids.append(watcher_kernel_id)
            if verbose:
                startup.print_log(
                    int(info.get("timestamp_ns")),
                    f"Program {program_id} kernel {watcher_kernel_id} compile finished in {info.get('duration_ns')/1000000} ms",
                )
        elif "program_compile_finished" in entry:
            info = entry["program_compile_finished"]
            program_id = int(info.get("id"))
            if program_id in programs:
                programs[program_id].compiled = True
            if verbose:
                startup.print_log(
                    int(info.get("timestamp_ns")),
                    f"Program {program_id} compile finished in {info.get('duration_ns')/1000000} ms",
                )
        elif "program_compile_already_exists" in entry:
            info = entry["program_compile_already_exists"]
            program_id = int(info.get("id"))
            if program_id in programs:
                programs[program_id].compiled = True
            if verbose:
                startup.print_log(int(info.get("timestamp_ns")), f"Program {program_id} compile already exists")
        elif "program_binary_status_change" in entry:
            info = entry["program_binary_status_change"]
            program_id = int(info.get("id"))
            device_id = int(info.get("device_id"))
            if program_id in programs:
                programs[program_id].binary_status_per_device[device_id] = info.get("status")
            if verbose:
                startup.print_log(
                    int(info.get("timestamp_ns")), f"Program {program_id} binary status changed to {info.get('status')}"
                )
    if verbose:
        print()
    return programs


def get_mesh_devices(log_directory: str, verbose: bool = False) -> dict[int, MeshDeviceData]:
    yaml_path = os.path.join(log_directory, "mesh_devices_log.yaml")
    data = read_yaml(yaml_path)
    if verbose:
        print("Mesh devices log:")
        startup = get_startup_data(log_directory)

    mesh_devices: dict[int, MeshDeviceData] = {}
    for entry in data:
        if "mesh_device_created" in entry:
            info = entry["mesh_device_created"]
            mesh_id = int(info.get("mesh_id"))
            mesh_device = MeshDeviceData(
                mesh_id=mesh_id,
                devices=[int(device_id) for device_id in info.get("devices", [])],
                shape=[int(dim) for dim in info.get("shape", [])],
                parent_mesh_id=int(info.get("parent_mesh_id")) if info.get("parent_mesh_id") is not None else None,
            )
            mesh_devices[mesh_id] = mesh_device
            if verbose:
                startup.print_log(
                    int(info.get("timestamp_ns")),
                    f"Mesh device {mesh_id} created. Devices: {mesh_device.devices}, Shape: {mesh_device.shape}, Parent: {mesh_device.parent_mesh_id}",
                )
        elif "mesh_device_destroyed" in entry:
            info = entry["mesh_device_destroyed"]
            mesh_id = int(info.get("mesh_id"))
            if mesh_id in mesh_devices:
                del mesh_devices[mesh_id]
            if verbose:
                startup.print_log(int(info.get("timestamp_ns")), f"Mesh device {mesh_id} destroyed")
        elif "mesh_device_initialized" in entry:
            info = entry["mesh_device_initialized"]
            mesh_id = int(info.get("mesh_id"))
            if mesh_id in mesh_devices:
                mesh_devices[mesh_id].initialized = True
            if verbose:
                startup.print_log(int(info.get("timestamp_ns")), f"Mesh device {mesh_id} initialized")
    if verbose:
        print()
    return mesh_devices


def get_mesh_workloads(log_directory: str, verbose: bool = False) -> dict[int, MeshWorkloadData]:
    yaml_path = os.path.join(log_directory, "mesh_workloads_log.yaml")
    data = read_yaml(yaml_path)
    if verbose:
        print("Mesh workloads log:")
        startup = get_startup_data(log_directory)

    mesh_workloads: dict[int, MeshWorkloadData] = {}
    for entry in data:
        if "mesh_workload_created" in entry:
            info = entry["mesh_workload_created"]
            mesh_workload_id = int(info.get("mesh_workload_id"))
            mesh_workloads[mesh_workload_id] = MeshWorkloadData(
                mesh_workload_id=mesh_workload_id, programs=[], binary_status_per_mesh_device={}
            )
            if verbose:
                startup.print_log(int(info.get("timestamp_ns")), f"Mesh workload {mesh_workload_id} created")
        elif "mesh_workload_destroyed" in entry:
            info = entry["mesh_workload_destroyed"]
            mesh_workload_id = int(info.get("mesh_workload_id"))
            if mesh_workload_id in mesh_workloads:
                del mesh_workloads[mesh_workload_id]
            if verbose:
                startup.print_log(int(info.get("timestamp_ns")), f"Mesh workload {mesh_workload_id} destroyed")
        elif "mesh_workload_add_program" in entry:
            info = entry["mesh_workload_add_program"]
            mesh_workload_id = int(info.get("mesh_workload_id"))
            program_id = int(info.get("program_id"))
            coordinates = [
                MeshCoordinate(coordinates=[int(c) for c in coord]) for coord in info.get("coordinates", [[]])
            ]
            if mesh_workload_id in mesh_workloads:
                mesh_workloads[mesh_workload_id].programs.append(
                    MeshWorkloadProgramData(program_id=program_id, coordinates=coordinates)
                )
            if verbose:
                startup.print_log(
                    int(info.get("timestamp_ns")),
                    f"Program {program_id} added to mesh workload {mesh_workload_id} with coordinates {coordinates}",
                )
        elif "mesh_workload_set_program_binary_status" in entry:
            info = entry["mesh_workload_set_program_binary_status"]
            mesh_workload_id = int(info.get("mesh_workload_id"))
            mesh_id = int(info.get("mesh_id"))
            if mesh_workload_id in mesh_workloads:
                mesh_workloads[mesh_workload_id].binary_status_per_mesh_device[mesh_id] = info.get("status")
            if verbose:
                startup.print_log(
                    int(info.get("timestamp_ns")),
                    f"Mesh workload {mesh_workload_id} binary status changed to {info.get('status')}",
                )
    if verbose:
        print()
    return mesh_workloads


def update_programs_with_mesh_workloads(
    programs: dict[int, ProgramData],
    mesh_workloads: dict[int, MeshWorkloadData],
    mesh_devices: dict[int, MeshDeviceData],
):
    for mesh_workload in mesh_workloads.values():
        for mesh_id, binary_status in mesh_workload.binary_status_per_mesh_device.items():
            if binary_status != "NotSet":
                for program_data in mesh_workload.programs:
                    program_id = program_data.program_id
                    if program_id not in programs:
                        continue  # Skip if the program is not found
                    program = programs[program_id]
                    for coordinate in program_data.coordinates:
                        device_id = mesh_devices[mesh_id].get_device_id(coordinate)
                        program.binary_status_per_device[device_id] = binary_status


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

    @cached_property
    def mesh_devices(self) -> dict[int, MeshDeviceData]:
        return get_mesh_devices(self.log_directory)

    @cached_property
    def mesh_workloads(self) -> dict[int, MeshWorkloadData]:
        return get_mesh_workloads(self.log_directory)

    @cached_property
    def kernels(self) -> dict[int, KernelData]:
        return get_kernels(self.log_directory)

    @cached_property
    def programs(self) -> dict[int, ProgramData]:
        programs = get_programs(self.log_directory)
        update_programs_with_mesh_workloads(programs, self.mesh_workloads, self.mesh_devices)
        return programs

    @cached_property
    def devices_in_use(self) -> set[int]:
        return get_devices_in_use(self.programs)


def get_log_directory(log_directory: str | None = None) -> str:
    if log_directory is None:
        log_directory = os.environ.get("TT_METAL_INSPECTOR_LOG_PATH", "")
        if log_directory:
            return log_directory
        log_directory = os.environ.get("TT_METAL_HOME", "")
        if not log_directory:
            raise ValueError("TT_METAL_HOME environment variable is not set")
        log_directory = os.path.join(log_directory, "generated", "inspector")
    return log_directory


@cache
def get_data(log_directory: str | None = None) -> InspectorData:
    return InspectorData(get_log_directory(log_directory))


def main():
    args = docopt(__doc__, argv=sys.argv[1:])
    log_directory = get_log_directory(args["<log-directory>"])
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
    for kernel in kernels.values():
        print(f"  {kernel.watcher_kernel_id}, pid {kernel.program_id}: {kernel.name} ({kernel.path})")
    print()

    mesh_devices = get_mesh_devices(log_directory, verbose=True)
    print("Mesh Devices:")
    for mesh_device in mesh_devices.values():
        print(
            f"  Mesh ID {mesh_device.mesh_id}, Parent Mesh ID: {mesh_device.parent_mesh_id}, Initialized: {mesh_device.initialized}"
        )
        print(f"    Devices: {mesh_device.devices}")
        print(f"    Shape: {mesh_device.shape}")
    print()

    mesh_workloads = get_mesh_workloads(log_directory, verbose=True)
    print("Mesh Workloads:")
    for mesh_workload in mesh_workloads.values():
        print(f"  Mesh Workload ID {mesh_workload.mesh_workload_id}")
        print(f"    Programs:")
        for program in mesh_workload.programs:
            print(f"      {program.program_id}: {program.coordinates}")
        print(f"    Binary status per mesh device: {mesh_workload.binary_status_per_mesh_device}")
    print()

    update_programs_with_mesh_workloads(programs, mesh_workloads, mesh_devices)
    print("Programs after updating with mesh workloads:")
    for program in programs.values():
        print(f"  Program ID {program.id}, compiled: {program.compiled}")
        print(f"    Binary status per device: {program.binary_status_per_device}")
        print(f"    Watcher Kernel IDs: {program.watcher_kernel_ids}")
    print()

    devices_in_use = get_devices_in_use(programs)
    print(f"Devices in use: {devices_in_use}")
    print()


if __name__ == "__main__":
    main()
