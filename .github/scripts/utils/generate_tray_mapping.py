#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Generate tray-to-PCIe device mapping and TP2 pairs for Galaxy/UBB systems.

This script reads cluster descriptor YAML and computes:
1. Tray-to-PCIe device mapping based on bus_id upper nibble
2. TP2 (Tensor Parallel 2) Ethernet-connected device pairs

E.g. Galaxy/UBB topology per tray (8 chips):
  4 - 3 - 2 - 1  (bus_id positions 4, 3, 2, 1)
  |   |   |   |
  5 - 6 - 7 - 8  (bus_id positions 5, 6, 7, 8)

Adjacent positions in bus_id order (1-2, 3-4, 5-6, 7-8) are Ethernet-connected.
"""

from __future__ import annotations

from typing import Any

import yaml
import ttnn


# Tray identification based on bus_id upper nibble
# Maps architecture to list of upper nibbles for trays 1-4
UBB_BUS_IDS: dict[str, list[int]] = {
    "wormhole_b0": [0xC0, 0x80, 0x00, 0x40],
    "blackhole": [0x00, 0x40, 0xC0, 0x80],
}

UBB_BOARD_TYPES: set[str] = {"ubb", "ubb_wormhole", "ubb_blackhole"}


def calculate_tray_id(arch: str, bus_id: int) -> int:
    """Calculate tray ID (1-4) from bus_id upper nibble."""
    if arch not in UBB_BUS_IDS:
        return 0
    tray_bus_ids = UBB_BUS_IDS[arch]
    upper_nibble = bus_id & 0xF0
    if upper_nibble in tray_bus_ids:
        return tray_bus_ids.index(upper_nibble) + 1
    return 0


def parse_cluster_descriptor(yaml_path: str) -> tuple[str, dict[int, int], dict[int, int], dict[int, str]]:
    """Parse cluster descriptor YAML and extract chip information."""
    try:
        with open(yaml_path) as f:
            data: dict[str, Any] = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cluster descriptor not found: {yaml_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in cluster descriptor: {e}")

    # Get architecture (use first chip's arch, assumes all chips have same arch)
    arch_map = data.get("arch", {})
    arch = next(iter(arch_map.values()), "unknown") if arch_map else "unknown"

    # Parse chips_with_mmio: list of {chip_id: pcie_id} dicts
    chips_with_mmio: dict[int, int] = {}
    for item in data.get("chips_with_mmio", []):
        if isinstance(item, dict):
            for chip_id, pcie_id in item.items():
                chips_with_mmio[int(chip_id)] = int(pcie_id)

    # Parse chip_to_bus_id: {chip_id: "0xNN"} or {chip_id: int}
    chip_to_bus_id: dict[int, int] = {}
    for chip_id, bus_id_val in data.get("chip_to_bus_id", {}).items():
        if isinstance(bus_id_val, int):
            chip_to_bus_id[int(chip_id)] = bus_id_val
        else:
            chip_to_bus_id[int(chip_id)] = int(bus_id_val, 16)

    # Parse chip_to_boardtype: {chip_id: "ubb"} - keys can be int or str
    chip_to_boardtype: dict[int, str] = {int(k): v for k, v in data.get("chip_to_boardtype", {}).items()}

    return arch, chips_with_mmio, chip_to_bus_id, chip_to_boardtype


def get_ubb_chips_with_tray(
    arch: str,
    chips_with_mmio: dict[int, int],
    chip_to_bus_id: dict[int, int],
    chip_to_boardtype: dict[int, str],
) -> list[dict[str, int]]:
    """Filter and return UBB chips with their tray, pcie_id, and bus_id."""
    result = []
    for chip_id, pcie_id in chips_with_mmio.items():
        board_type = chip_to_boardtype.get(chip_id, "").lower()
        if board_type not in UBB_BOARD_TYPES:
            continue

        bus_id = chip_to_bus_id.get(chip_id, 0)
        tray_id = calculate_tray_id(arch, bus_id)
        if tray_id > 0:
            result.append({"tray_id": tray_id, "pcie_id": pcie_id, "bus_id": bus_id})
    return result


def get_pcie_devices_per_tray(
    arch: str,
    chips_with_mmio: dict[int, int],
    chip_to_bus_id: dict[int, int],
    chip_to_boardtype: dict[int, str],
) -> dict[int, list[int]]:
    """Group PCIe devices by tray based on bus_id.

    Returns empty dict if not on a Galaxy/UBB system.
    """
    result: dict[int, list[int]] = {}
    for chip in get_ubb_chips_with_tray(arch, chips_with_mmio, chip_to_bus_id, chip_to_boardtype):
        result.setdefault(chip["tray_id"], []).append(chip["pcie_id"])
    return result


def get_tp2_device_pairs(
    arch: str,
    chips_with_mmio: dict[int, int],
    chip_to_bus_id: dict[int, int],
    chip_to_boardtype: dict[int, str],
) -> list[list[int]]:
    """
    Get Ethernet-connected device pairs for TP2.

    Returns empty list if not on a Galaxy/UBB system.

    Algorithm:
    1. Group chips by tray using bus_id upper nibble
    2. Sort chips within each tray by bus_id lower nibble (physical position)
    3. Pair adjacent chips: [0,1], [2,3], [4,5], [6,7]
    """
    # Group chips by tray
    chips_by_tray: dict[int, list[dict[str, int]]] = {}
    for chip in get_ubb_chips_with_tray(arch, chips_with_mmio, chip_to_bus_id, chip_to_boardtype):
        chips_by_tray.setdefault(chip["tray_id"], []).append(chip)

    # Create pairs
    pairs = []
    for tray_id in sorted(chips_by_tray.keys()):
        chips = chips_by_tray[tray_id]
        # Sort by bus_id lower nibble (physical position within tray)
        chips.sort(key=lambda c: c["bus_id"] & 0x0F)

        # Pair adjacent chips
        for i in range(0, len(chips) - 1, 2):
            chip_a = chips[i]
            chip_b = chips[i + 1]
            # Order pair by pcie_id
            if chip_a["pcie_id"] < chip_b["pcie_id"]:
                pairs.append([chip_a["pcie_id"], chip_b["pcie_id"]])
            else:
                pairs.append([chip_b["pcie_id"], chip_a["pcie_id"]])

    return pairs


def main():
    # Get cluster descriptor YAML path
    yaml_path = ttnn.cluster.serialize_cluster_descriptor()

    # Parse cluster descriptor
    arch, chips_with_mmio, chip_to_bus_id, chip_to_boardtype = parse_cluster_descriptor(yaml_path)

    # Compute tray mapping and TP2 pairs
    device_mapping = get_pcie_devices_per_tray(arch, chips_with_mmio, chip_to_bus_id, chip_to_boardtype)
    tp2_pairs = get_tp2_device_pairs(arch, chips_with_mmio, chip_to_bus_id, chip_to_boardtype)

    output = {
        "arch": arch,
        "device_mapping": {str(k): sorted(v) for k, v in device_mapping.items()},
        "tp2_pairs": tp2_pairs,
    }

    with open("tray_to_pcie_device_mapping.yaml", "w") as f:
        yaml.dump(output, f)


if __name__ == "__main__":
    main()
