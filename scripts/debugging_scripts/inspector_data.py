#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    inspector_data [--inspector-log-path=<inspector_log_path>]

Options:
    --inspector-log-path=<inspector_log_path>  Path to the inspector log directory.

Description:
    Provides inspector data for other scripts.
    This script will parse the inspector logs and provide structured data about the devices, workloads, kernels, programs, mesh devices, and mesh workloads.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from triage import triage_singleton, ScriptConfig, run_script
from parse_inspector_logs import get_data as get_logs_data

script_config = ScriptConfig(
    data_provider=True,
)


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
            linear_index = linear_index + coordinate.coordinates[dim] * (1 if dim == 0 else self.shape[dim - 1])
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


class InspectorData(ABC):
    @property
    @abstractmethod
    def mesh_devices(self) -> dict[int, MeshDeviceData]:
        pass

    @property
    @abstractmethod
    def mesh_workloads(self) -> dict[int, MeshWorkloadData]:
        pass

    @property
    @abstractmethod
    def kernels(self) -> dict[int, KernelData]:
        pass

    @property
    @abstractmethod
    def programs(self) -> dict[int, ProgramData]:
        pass

    @property
    @abstractmethod
    def devices_in_use(self) -> set[int]:
        pass


@triage_singleton
def run(args, context) -> InspectorData:
    log_directory = args["--inspector-log-path"]
    return get_logs_data(log_directory)


if __name__ == "__main__":
    run_script()
