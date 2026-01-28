#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Generate tray-to-PCIe device mapping using Python binding.

This script uses the GetPCIeDevicesPerTray C++ binding to generate
a YAML file mapping tray IDs to PCIe device IDs for Galaxy systems.
"""

import yaml
import ttnn


def main():
    # Call the C++ binding directly to get tray-to-device mapping
    mapping = ttnn._ttnn.device.GetPCIeDevicesPerTray()
    arch = ttnn._ttnn.device.get_arch_name()

    output = {
        "arch": arch,
        "device_mapping": {str(k): sorted(list(v)) for k, v in mapping.items()},
    }

    with open("tray_to_pcie_device_mapping.yaml", "w") as f:
        yaml.dump(output, f)

    print("Generated tray_to_pcie_device_mapping.yaml")
    print(yaml.dump(output))


if __name__ == "__main__":
    main()
