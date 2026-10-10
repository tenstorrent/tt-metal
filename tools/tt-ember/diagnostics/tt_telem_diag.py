#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

from tt_umd import TopologyDiscovery

def summarize_obj(name, obj):
    print(f"\n--- {name} ---")
    print(f"type: {type(obj)}")
    try:
        l = len(obj)  # works for lists/tuples/dicts
        print(f"len: {l}")
    except Exception:
        pass

    # Print a few likely methods/attrs without dumping everything
    candidates = [
        "get_chip_locations", "get_arch", "get_pci_device", "get_device",
        "devices", "get_devices", "chip_ids", "get_chip_ids",
        "enumerate_devices", "enumerate_devices_info",
        "get_device_info", "get_pci_device_id",
    ]
    present = [c for c in candidates if hasattr(obj, c)]
    if present:
        print("interesting attrs:", ", ".join(present))
    else:
        print("interesting attrs: (none of the common ones)")

def main():
    res = TopologyDiscovery.discover()
    print("TopologyDiscovery.discover() type:", type(res))

    if not isinstance(res, tuple):
        summarize_obj("discover_result", res)
        return 0

    print("tuple length:", len(res))
    for i, item in enumerate(res):
        summarize_obj(f"tuple[{i}]", item)

        # If it's a container, show element types for first few
        if isinstance(item, (list, tuple)) and len(item) > 0:
            print("  element[0] type:", type(item[0]))
        if isinstance(item, dict) and len(item) > 0:
            k0 = next(iter(item.keys()))
            print("  key[0] type:", type(k0), "value[0] type:", type(item[k0]))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
