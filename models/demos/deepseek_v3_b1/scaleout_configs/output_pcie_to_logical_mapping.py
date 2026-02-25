#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Output PCI device ID -> logical ID mapping for the current host.
UMD TT_VISIBLE_DEVICES expects logical IDs (BDF-sorted indices), not PCI IDs.
Run via mpirun once per host to collect mappings for all hosts.
"""
import argparse
import sys

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", required=True, help="Hostname for this mapping (from hostfile)")
    parser.add_argument("--output", required=True, help="Output YAML file path")
    args = parser.parse_args()

    try:
        import tt_umd
    except ImportError as e:
        print(f"tt_umd not available: {e}", file=sys.stderr)
        sys.exit(1)

    cluster_desc = tt_umd.TopologyDiscovery.create_cluster_descriptor()
    chips_with_mmio = cluster_desc.get_chips_with_mmio()
    # chips_with_mmio: chip_id (logical) -> pcie_id; we need pcie_id -> logical_id
    pcie_to_logical = {int(pcie_id): int(chip_id) for chip_id, pcie_id in chips_with_mmio.items()}

    with open(args.output, "w") as f:
        yaml.dump({args.hostname: pcie_to_logical}, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
