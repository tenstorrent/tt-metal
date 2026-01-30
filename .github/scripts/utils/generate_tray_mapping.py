#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Generate tray-to-PCIe device mapping and TP2 pairs using Python bindings.

This script uses the GetPCIeDevicesPerTray and GetTP2DevicePairsWithBusIds C++ bindings
to generate a YAML file mapping tray IDs to PCIe device IDs and TP2 device
pairs for Galaxy systems.
"""

import yaml
import ttnn


def main():
    # Call the C++ bindings directly to get mappings
    mapping = ttnn._ttnn.device.GetPCIeDevicesPerTray()
    arch = ttnn._ttnn.device.get_arch_name()
    tp2_pairs_with_bus_ids = ttnn._ttnn.device.GetTP2DevicePairsWithBusIds()

    output = {
        "arch": arch,
        "device_mapping": {str(k): sorted(list(v)) for k, v in mapping.items()},
        "tp2_pairs": [[p[0][0], p[1][0]] for p in tp2_pairs_with_bus_ids],
    }

    with open("tray_to_pcie_device_mapping.yaml", "w") as f:
        yaml.dump(output, f)


if __name__ == "__main__":
    main()
