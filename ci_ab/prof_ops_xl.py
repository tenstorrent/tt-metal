#!/usr/bin/env python3
"""Scratch, not for merge. topk_large_indices GLM rows (160 x 524288 and 160 x 102400, k 2048) on the full grid and on a
108 core grid (rows 0..8 full width plus row 9 columns 0..8) under the device profiler: signpost per case, N_MEAS calls."""
import gc
import os

import torch
import ttnn
from tracy import signpost

N_WARM, N_MEAS = 2, int(os.environ.get("OPS_NMEAS", "8"))


def grid108(dev):
    g = dev.compute_with_storage_grid_size()
    rs = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, 8)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(8, 9)),
        ]
    )
    assert rs.num_cores() == 108, rs.num_cores()
    return rs


def case(rows, n, k, g108):
    def setup(dev):
        torch.manual_seed(2007)
        x = torch.randn((rows, n), dtype=torch.float32).to(torch.bfloat16)
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        return t, (grid108(dev) if g108 else None)

    def run(x, grid):
        if grid is None:
            return ttnn.experimental.topk_large_indices(x, k=k)
        return ttnn.experimental.topk_large_indices(x, k=k, sub_core_grids=grid)

    return setup, run


CASES = {
    "glm_r160_n524288_k2048_full": case(160, 524288, 2048, False),
    "glm_r160_n524288_k2048_g108": case(160, 524288, 2048, True),
    "glm_r160_n102400_k2048_full": case(160, 102400, 2048, False),
    "glm_r160_n102400_k2048_g108": case(160, 102400, 2048, True),
}


def main():
    names = os.environ.get("OPS_CASES", "").split() or list(CASES)
    dev = ttnn.open_device(device_id=0)
    try:
        for name in names:
            setup, run = CASES[name]
            gc.collect()
            try:
                args = setup(dev)
                for _ in range(N_WARM):
                    run(*args)
                ttnn.synchronize_device(dev)
                signpost(header=name)
                for _ in range(N_MEAS):
                    run(*args)
                ttnn.synchronize_device(dev)
                signpost(header="end")
                ttnn.ReadDeviceProfiler(dev)
                print(f"CASE {name} ok", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                ttnn.synchronize_device(dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
