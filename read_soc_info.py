#!/usr/bin/env python3
"""
Read SOC Descriptor using UMD Python API
Shows: Grid size, L1 memory, harvesting, core counts
"""

import sys

sys.path.insert(0, "/home/ttuser/aperezvicente/tt-metal-apv/tt_metal/third_party/umd/build/nanobind")

import tt_umd


def main():
    # Enumerate devices
    pci_ids = tt_umd.PCIDevice.enumerate_devices()
    print(f"Found {len(pci_ids)} device(s): {pci_ids}\n")

    if len(pci_ids) == 0:
        print("No devices found!")
        return

    # Summary across all devices
    total_tensix = 0
    total_harvested = 0
    total_l1 = 0

    # Read SOC info from ALL devices
    for pci_id in pci_ids:
        dev = tt_umd.TTDevice.create(pci_id)
        dev.init_tt_device()

        print("=" * 70)
        print(f"Device {pci_id} - SOC Descriptor Information")
        print("=" * 70)

        # Get basic device info
        arch = dev.get_arch()
        board = dev.get_board_type()
        print(f"\nArchitecture: {arch}")
        print(f"Board Type:   {board}")

        # Create SOC descriptor
        soc = tt_umd.SocDescriptor(dev)

        # Get core counts by type
        print("\n" + "=" * 70)
        print("Core Counts (after harvesting)")
        print("=" * 70)

        core_types = {
            "TENSIX (worker)": tt_umd.CoreType.TENSIX,
            "ETH (ethernet)": tt_umd.CoreType.ETH,
            "DRAM": tt_umd.CoreType.DRAM,
            "PCIE": tt_umd.CoreType.PCIE,
            "ARC": tt_umd.CoreType.ARC,
        }

        tensix_cores = []
        for name, core_type in core_types.items():
            cores = soc.get_cores(core_type)
            print(f"{name:20s}: {len(cores):3d} cores")
            if core_type == tt_umd.CoreType.TENSIX:
                tensix_cores = cores

        # Calculate grid dimensions from Tensix cores
        if tensix_cores:
            # Cores are already in NOC0 coordinates, just count the dimensions
            max_x = max(c.x for c in tensix_cores) + 1
            max_y = max(c.y for c in tensix_cores) + 1

            print(f"\nTensix Grid Bounding Box: {max_x} × {max_y}")
            print(f"Active Tensix Cores: {len(tensix_cores)}")

            # Show first few cores as examples
            print(f"\nExample core coordinates (NOC0): ", end="")
            for i, c in enumerate(tensix_cores[:5]):
                print(f"({c.x},{c.y})", end=" ")
            if len(tensix_cores) > 5:
                print("...")

        # Get harvested cores
        print("\n" + "=" * 70)
        print("Harvesting Information")
        print("=" * 70)

        harvested_tensix = 0
        for name, core_type in core_types.items():
            harvested = soc.get_harvested_cores(core_type)
            if len(harvested) > 0:
                print(f"{name:20s}: {len(harvested)} harvested cores")
                if core_type == tt_umd.CoreType.TENSIX:
                    harvested_tensix = len(harvested)

        all_harvested = soc.get_all_harvested_cores()
        print(f"\nTotal harvested:      {len(all_harvested)} cores")

        # Calculate L1 memory
        print("\n" + "=" * 70)
        print("L1 Memory (Tensix cores only)")
        print("=" * 70)

        # From wormhole_implementation.hpp
        l1_per_tensix = 1499136  # bytes
        device_l1 = l1_per_tensix * len(tensix_cores)

        print(f"L1 per core:  {l1_per_tensix:,} bytes ({l1_per_tensix/(1024**2):.2f} MiB)")
        print(f"Tensix cores: {len(tensix_cores)}")
        print(f"Total L1:     {device_l1:,} bytes ({device_l1/(1024**2):.1f} MiB)")

        # Accumulate totals
        total_tensix += len(tensix_cores)
        total_harvested += harvested_tensix
        total_l1 += device_l1

        print("\n")

    # Print system-wide summary
    print("=" * 70)
    print("SYSTEM-WIDE SUMMARY (All Devices)")
    print("=" * 70)
    print(f"Total devices:        {len(pci_ids)}")
    print(
        f"Total Tensix cores:   {total_tensix} active + {total_harvested} harvested = {total_tensix + total_harvested}"
    )
    print(f"Total L1 memory:      {total_l1:,} bytes ({total_l1/(1024**2):.1f} MiB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
