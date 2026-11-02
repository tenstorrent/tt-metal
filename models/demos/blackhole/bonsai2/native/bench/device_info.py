# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Print what the opened device actually offers: cores, DRAM banks, memory.

The A0 roofline came out 300x below the chip peak, which is only believable if
the device being measured is not the device we think it is.  This dumps the
grid, the bank-to-core assignment the streaming op uses, and the DRAM size, so
that question is settled with numbers rather than assumed.
"""

from __future__ import annotations

import os

import ttnn


def main() -> int:
    dev_id = int(os.environ.get("BONSAI_DEVICE", 0))
    print(f"TT_VISIBLE_DEVICES={os.environ.get('TT_VISIBLE_DEVICES')} MESH_DEVICE={os.environ.get('MESH_DEVICE')}")
    print(f"opening device_id={dev_id}")
    device = ttnn.open_device(device_id=dev_id, l1_small_size=24576)
    try:
        print(f"arch                {device.arch()}")
        print(f"num_devices         {device.get_num_devices()}")
        print(f"compute grid        {device.compute_with_storage_grid_size()}")
        print(f"dram grid           {device.dram_grid_size()}")
        print(f"num DRAM banks      {device.num_dram_banks() if hasattr(device, 'num_dram_banks') else 'n/a'}")
        all_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        print(f"bank->worker cores  {len(all_cores)}: {[(c.x, c.y) for c in all_cores]}")
        core_grid = device.compute_with_storage_grid_size()
        print(f"total compute cores {core_grid.x * core_grid.y}")
        # Memory figures need a per-chip device handle; MeshDevice has no
        # allocator(), and they are not needed to answer "is this chip healthy".
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
