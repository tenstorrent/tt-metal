#!/usr/bin/env python3
"""Scratch, not for merge. ttnn.topk multi-core (Blackhole tree merge) device time under the device profiler: per case a
signpost, N_MEAS measured calls after N_WARM warm-up calls, then an "end" signpost. Cases from $TREE_CASES. With
TREE_PROBE=1 each case's new kernel names in $TT_METAL_CACHE are printed (mock probe)."""
import gc
import glob
import os

import torch
import ttnn
from tracy import signpost

N_WARM, N_MEAS = 2, int(os.environ.get("TREE_NMEAS", "8"))


def case(h, w, k, stable=False, largest=True, grid=None):
    def setup(dev):
        torch.manual_seed(2007)
        t = torch.randn((1, 1, h, w), dtype=torch.bfloat16)
        x = ttnn.from_torch(t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        g = None
        if grid is not None:
            g = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0], grid[1] + 1))])
        return x, g

    def run(x, g):
        kw = {"sub_core_grids": g} if g is not None else {}
        return ttnn.topk(x, k, dim=-1, largest=largest, sorted=True, stable=stable, **kw)

    return setup, run


CASES = {
    # sampling_1d multi-device: the per-device vocab shard padded to 16384 (Llama on 8 devices), max_top_k 32; #57396's table
    "w16384_h32_k32": case(32, 16384, 32),
    "w16384_h32_k64": case(32, 16384, 64),
    "w16384_h32_k32_stable": case(32, 16384, 32, stable=True),
    "w8192_h64_k64": case(64, 8192, 64),
    # test_topk_low_tile_row_multicore and the ab_dump shapes (<= 2 tile rows, pow2 width >= 1024)
    "w1024_h32_k32": case(32, 1024, 32),
    "w1024_h32_k64": case(32, 1024, 64),
    "w2048_h32_k8": case(32, 2048, 8),
    "w2048_h32_k32": case(32, 2048, 32),
    "w2048_h32_k64": case(32, 2048, 64),
}
# test_topk_multicore_tree_merge: 64 x 16384 on 2, 4, 8 and 32 local cores
for name, g in (("n2", (2, 1)), ("n4", (4, 1)), ("n8", (4, 2)), ("n32", (8, 4))):
    for k in (32, 64):
        for st in (False, True):
            CASES[f"tree_{name}_k{k}" + ("_stable" if st else "")] = case(64, 16384, k, stable=st, grid=g)


def kernels(cache):
    return {p.split(os.sep)[-2] for p in glob.glob(os.path.join(cache, "*", "kernels", "*", "*"))
            if os.path.isdir(p)} if cache else set()


def kernel_names(cache):
    return {p.split(os.sep)[-3] for p in glob.glob(os.path.join(cache, "*", "kernels", "*", "*"))}


def main():
    names = os.environ.get("TREE_CASES", "").split() or list(CASES)
    probe = os.environ.get("TREE_PROBE") == "1"
    cache = os.environ.get("TT_METAL_CACHE", "")
    dev = ttnn.open_device(device_id=0)
    try:
        for name in names:
            setup, run = CASES[name]
            gc.collect()
            before = set(glob.glob(os.path.join(cache, "*", "kernels", "*", "*"))) if probe else set()
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
                args[0].deallocate()
                print(f"CASE {name} ok", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                ttnn.synchronize_device(dev)
            if probe:
                new = set(glob.glob(os.path.join(cache, "*", "kernels", "*", "*"))) - before
                print(f"PROBE {name} " + " ".join(sorted({p.split(os.sep)[-2] for p in new})), flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
