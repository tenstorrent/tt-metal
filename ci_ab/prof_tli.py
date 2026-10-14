#!/usr/bin/env python3
"""Scratch, not for merge. topk_large_indices device time under the device profiler (per case a signpost, N_MEAS calls after
N_WARM, an "end" signpost), or with TLI_DUMP=<dir> the indices of every case saved for an arm-to-arm compare."""
import gc
import os
import sys

import torch
import ttnn

N_WARM, N_MEAS = 2, int(os.environ.get("OPS_NMEAS", "8"))
# (rows, n, k, valid_length): production shapes (MiniMax-M3 prefill and bounded cache, DeepSeek V3 indexer, MSA blocks,
# RT-DETR queries) and the K' 1024 / 2048 multi-chunk rows the split serves.
CASES = {
    "prefill_640x51200_k1536": (640, 51200, 1536, None),
    "bounded_2x102400_v56320_k1536": (2, 102400, 1536, 56320),
    "dsv3_64x131072_k2048": (64, 131072, 2048, None),
    "dsv3_128x16384_k2048": (128, 16384, 2048, None),
    "dsv3_32x65536_v40000_k2048": (32, 65536, 2048, 40000),
    "k1024_256x16384": (256, 16384, 1024, None),
    "k1024_2x8192": (2, 8192, 1024, None),
    "msa_4096x1024_k16": (4096, 1024, 16, None),
    "msa_1024x4096_k16": (1024, 4096, 16, None),
    "rtdetr_1x8400_k304": (1, 8400, 304, None),
    "k512_640x4096": (640, 4096, 512, None),
    "k512_640x1024": (640, 1024, 512, None),
    "k512_640x1536": (640, 1536, 512, None),
    "k1024_640x2048": (640, 2048, 1024, None),
    "k1024_640x3072": (640, 3072, 1024, None),
    "k2048_640x4096": (640, 4096, 2048, None),
    "k2048_640x6144": (640, 6144, 2048, None),
    "dsv3_192x16384_k2048": (192, 16384, 2048, None),
    "dsv3_200x8192_k2048": (200, 8192, 2048, None),
    "k512_160x4096": (160, 4096, 512, None),
}


def main():
    dump = os.environ.get("TLI_DUMP")
    names = os.environ.get("OPS_CASES", "").split() or list(CASES)
    if not dump:
        from tracy import signpost
    dev = ttnn.open_device(device_id=0)
    out = {}
    try:
        for name in names:
            rows, n, k, vl = CASES[name]
            gc.collect()
            try:
                torch.manual_seed(2026)
                x = torch.randn((rows, n), dtype=torch.float32).to(torch.bfloat16)
                t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
                kw = {} if vl is None else {"valid_length": vl}
                run = lambda: ttnn.experimental.topk_large_indices(t, k=k, **kw)
                if dump:
                    out[name] = ttnn.to_torch(run(), dtype=torch.uint32).to(torch.int64)
                    print(f"CASE {name} ok", flush=True)
                    continue
                for _ in range(N_WARM):
                    run()
                ttnn.synchronize_device(dev)
                signpost(header=name)
                for _ in range(N_MEAS):
                    run()
                ttnn.synchronize_device(dev)
                signpost(header="end")
                ttnn.ReadDeviceProfiler(dev)
                print(f"CASE {name} ok", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                ttnn.synchronize_device(dev)
    finally:
        ttnn.close_device(dev)
    if dump:
        torch.save(out, dump)
        print(f"DUMP DONE {len(out)}", flush=True)


if __name__ == "__main__":
    main()
