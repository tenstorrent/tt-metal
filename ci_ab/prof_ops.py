#!/usr/bin/env python3
"""Scratch, not for merge. Device time of ttnn.experimental.topk_large_indices (the Blackhole topk_xl op: DeepSeek V3.2 indexer
rows, MiniMax block scores, sampling) per case under the device profiler: signpost, N_MEAS measured calls after N_WARM warm-up
calls, then an "end" signpost. Cases from $OPS_CASES (all if empty)."""
import gc
import os

import torch
import ttnn
from tracy import signpost

N_WARM, N_MEAS = 2, int(os.environ.get("OPS_NMEAS", "8"))


def case_xl(rows, n, k, valid_length=None):
    def setup(dev):
        torch.manual_seed(2007)
        x = torch.randn((rows, n), dtype=torch.float32).to(torch.bfloat16)
        return (ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev),)

    def run(x):
        return ttnn.experimental.topk_large_indices(x, k=k, valid_length=valid_length)

    return setup, run


CASES = {
    # DeepSeek V3.2 indexer (index_topk 2048): n = tokens, 2 chunks a row at 4K
    "ds_idx_r640_n4096_k2048": case_xl(640, 4096, 2048),
    "ds_idx_r640_n8192_k2048": case_xl(640, 8192, 2048),
    "ds_idx_r640_n12288_k2048": case_xl(640, 12288, 2048),
    "ds_idx_r640_n6144_k2048_odd": case_xl(640, 6144, 2048),
    "ds_idx_r64_n4096_k2048": case_xl(64, 4096, 2048),
    "ds_idx_r1_n4096_k2048": case_xl(1, 4096, 2048),
    "ds_idx_r640_n4096_k2048_valid3000": case_xl(640, 4096, 2048, valid_length=3000),
    # K 1024 and 512 rows, production prefill and bounded cache
    "r640_n4096_k1024": case_xl(640, 4096, 1024),
    "r640_n8192_k1024": case_xl(640, 8192, 1024),
    "r640_n2048_k512": case_xl(640, 2048, 512),
    "r640_n4096_k512": case_xl(640, 4096, 512),
    "r640_n1024_k512": case_xl(640, 1024, 512),
    "prefill_r640_n51200_k1536": case_xl(640, 51200, 1536),
    "bounded_r2_n102400_k1536_v56320": case_xl(2, 102400, 1536, valid_length=56320),
}


def main():
    table = CASES
    names = os.environ.get("OPS_CASES", "").split() or list(table)
    dev = ttnn.open_device(device_id=0)
    try:
        for name in names:
            setup, run = table[name]
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
