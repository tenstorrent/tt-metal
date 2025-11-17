#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    device_id_mapping

Description:
    Data provider script for mapping between logical chip IDs (device IDs) and hardware unique IDs.
    This mapping is necessary when TT_VISIBLE_DEVICES is used, as it remaps logical device IDs,
    but unique IDs remain stable and can be used for reliable device identification.
"""

from inspector_data import run as get_inspector_data, InspectorData
from triage import triage_singleton, ScriptConfig, run_script, TTTriageError
from ttexalens.context import Context


script_config = ScriptConfig(
    data_provider=True,
    depends=["inspector_data"],
)


class DeviceIdMapping:
    """
    Provides mapping between logical chip IDs (device IDs) and hardware unique IDs.

    Logical chip IDs are remapped when TT_VISIBLE_DEVICES is used, but unique IDs
    remain stable regardless of visible device configuration.
    """

    def __init__(self, inspector_data: InspectorData):
        """
        Initialize the device ID mapping from Inspector RPC data.

        Args:
            inspector_data: InspectorData object to fetch chip_id to unique_id mapping

        Raises:
            TTTriageError: If the mapping cannot be fetched from Inspector RPC
        """
        try:
            unique_ids_result = inspector_data.getChipUniqueIds()
            # Build forward mapping: logical chip_id -> unique_id
            self._chip_id_to_unique_id: dict[int, int] = {}
            # Build reverse mapping: unique_id -> logical chip_id
            self._unique_id_to_chip_id: dict[int, int] = {}

            for mapping in unique_ids_result.mappings:
                chip_id = mapping.chipId
                unique_id = mapping.uniqueId
                self._chip_id_to_unique_id[chip_id] = unique_id
                self._unique_id_to_chip_id[unique_id] = chip_id
        except Exception as e:
            raise TTTriageError(
                f"Failed to get chip_id to unique_id mapping from Inspector RPC: {e}\n"
                "Make sure Inspector RPC is available or serialized RPC data exists.\n"
                "Set TT_METAL_INSPECTOR_RPC=1 when running your Metal application."
            ) from e

    def chip_id_to_unique_id(self, chip_id: int) -> int:
        """
        Convert a logical chip ID to hardware unique ID.

        Args:
            chip_id: Logical chip ID (device ID)

        Returns:
            Hardware unique ID

        Raises:
            KeyError: If chip_id is not found in the mapping
        """
        if chip_id not in self._chip_id_to_unique_id:
            raise KeyError(
                f"Logical chip ID {chip_id} not found in mapping. "
                f"Available chip IDs: {list(self._chip_id_to_unique_id.keys())}"
            )
        return self._chip_id_to_unique_id[chip_id]

    def unique_id_to_chip_id(self, unique_id: int) -> int:
        """
        Convert a hardware unique ID to logical chip ID.

        Args:
            unique_id: Hardware unique ID

        Returns:
            Logical chip ID (device ID)

        Raises:
            KeyError: If unique_id is not found in the mapping
        """
        if unique_id not in self._unique_id_to_chip_id:
            raise KeyError(
                f"Unique ID {unique_id} not found in mapping. "
                f"Available unique IDs: {list(self._unique_id_to_chip_id.keys())}"
            )
        return self._unique_id_to_chip_id[unique_id]

    def remap_chip_ids_to_unique_ids(self, chip_ids: list[int]) -> list[int]:
        """
        Remap a list of logical chip IDs to unique IDs.

        Args:
            chip_ids: List of logical chip IDs

        Returns:
            List of unique IDs in the same order
        """
        return [self.chip_id_to_unique_id(chip_id) for chip_id in chip_ids]

    def remap_unique_ids_to_chip_ids(self, unique_ids: list[int]) -> list[int]:
        """
        Remap a list of unique IDs to logical chip IDs.

        Args:
            unique_ids: List of unique IDs

        Returns:
            List of logical chip IDs in the same order
        """
        return [self.unique_id_to_chip_id(unique_id) for unique_id in unique_ids]

    def get_all_mappings(self) -> dict[int, int]:
        """
        Get the complete mapping from logical chip ID to unique ID.

        Returns:
            Dictionary mapping logical chip_id -> unique_id
        """
        return self._chip_id_to_unique_id.copy()

    def has_chip_id(self, chip_id: int) -> bool:
        """
        Check if a logical chip ID exists in the mapping.

        Args:
            chip_id: Logical chip ID to check

        Returns:
            True if chip_id exists in the mapping, False otherwise
        """
        return chip_id in self._chip_id_to_unique_id

    def has_unique_id(self, unique_id: int) -> bool:
        """
        Check if a unique ID exists in the mapping.

        Args:
            unique_id: Unique ID to check

        Returns:
            True if unique_id exists in the mapping, False otherwise
        """
        return unique_id in self._unique_id_to_chip_id


@triage_singleton
def run(args, context: Context) -> DeviceIdMapping:
    """
    Create and return a DeviceIdMapping instance.

    Args:
        args: Parsed command-line arguments
        context: ttexalens Context object

    Returns:
        DeviceIdMapping instance with chip_id to unique_id mappings
    """
    inspector_data = get_inspector_data(args, context)
    return DeviceIdMapping(inspector_data)


if __name__ == "__main__":
    run_script()
