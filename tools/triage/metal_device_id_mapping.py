#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    metal_device_id_mapping

Description:
    Mapping between Metal device ID and Unique ID.

    This mapping is necessary when TT_METAL_VISIBLE_DEVICES is used. When TT_METAL_VISIBLE_DEVICES
    restricts devices, Inspector RPC uses remapped metal device IDs (starting from 0), while
    tt-exalens uses the original device IDs from the full device set. This causes a
    mismatch between Metal device IDs and Exalens device IDs.
"""

from inspector_data import run as get_inspector_data, InspectorData
from triage import triage_singleton, ScriptConfig, run_script, log_check
from ttexalens.context import Context


script_config = ScriptConfig(
    data_provider=True,
    depends=["inspector_data"],
)


class MetalDeviceIdMapping:
    def __init__(self, inspector_data: InspectorData):
        unique_ids_result = inspector_data.getMetalDeviceIdMappings()
        self._metal_device_id_to_unique_id: dict[int, int] = {}
        self._unique_id_to_metal_device_id: dict[int, int] = {}

        for mapping in unique_ids_result.mappings:
            metal_device_id = mapping.metalDeviceId
            unique_id = mapping.uniqueId
            log_check(
                metal_device_id not in self._metal_device_id_to_unique_id, "Invalid Inspector data. Duplicated chip id"
            )
            log_check(
                unique_id not in self._unique_id_to_metal_device_id, "Invalid Inspector data. Duplicated unique id"
            )
            self._metal_device_id_to_unique_id[metal_device_id] = unique_id
            self._unique_id_to_metal_device_id[unique_id] = metal_device_id

    def get_unique_id(self, metal_device_id: int) -> int:
        log_check(
            metal_device_id in self._metal_device_id_to_unique_id,
            f"Inspector doesn't know about chip_id: {metal_device_id}",
        )
        return self._metal_device_id_to_unique_id[metal_device_id]

    def get_metal_device_id(self, unique_id: int) -> int:
        log_check(
            unique_id in self._unique_id_to_metal_device_id, f"Inspector doesn't know about unique_id: {unique_id}"
        )
        return self._unique_id_to_metal_device_id[unique_id]

    def has_metal_device_id(self, metal_device_id: int) -> bool:
        return metal_device_id in self._metal_device_id_to_unique_id

    def has_unique_id(self, unique_id: int) -> bool:
        return unique_id in self._unique_id_to_metal_device_id


@triage_singleton
def run(args, context: Context) -> MetalDeviceIdMapping:
    inspector_data = get_inspector_data(args, context)
    return MetalDeviceIdMapping(inspector_data)


if __name__ == "__main__":
    run_script()
