#!/usr/bin/env python3
"""Scratch, not for merge. topk_large_indices device time under the device profiler: per case a signpost, then N_MEAS
measured calls after N_WARM warm-up calls. Cases from $TLI_CASES (space separated)."""
import gc
import os

import torch
import ttnn
from tracy import signpost

N_WARM, N_MEAS = 2, int(os.environ.get("TLI_NMEAS", "8"))


def case(rows, n, k, valid_length=None, meta=False):
    def setup(dev):
        torch.manual_seed(0)
        x = torch.randn((rows, n), dtype=torch.float32).to(torch.bfloat16)
        args = [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)]
        if meta:
            v = torch.tensor([valid_length], dtype=torch.int64).reshape(1, 1, 1, 1)
            args.append(ttnn.from_torch(v, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG))
        return args

    def run(x, m=None):
        if m is not None:
            return ttnn.experimental.topk_large_indices(x, k=k, valid_length_tensor=m)
        kw = {"valid_length": valid_length} if valid_length is not None else {}
        return ttnn.experimental.topk_large_indices(x, k=k, **kw)

    return setup, run


CASES = {
    # test_topk_large_indices.py production perf rows
    "prefill_640x51200_k1536": case(640, 51200, 1536),
    "bounded_2x102400_v56320_k1536": case(2, 102400, 1536, 56320),
    "decode_160x524288_k2048": case(160, 524288, 2048),
    "decode_meta_160x524288_k2048": case(160, 524288, 2048, 524288, meta=True),
    "glm95k_160x102400_k2048": case(160, 102400, 2048, 102400),
    # DeepSeek V4 Pro indexer, INDEX_TOPK 1024
    "v4pro_640x3328_v3310_k1024": case(640, 3328, 1024, 3310),
    "v4pro_640x2560_k1024": case(640, 2560, 1024),
    "v4pro_640x8192_k1024": case(640, 8192, 1024),
    "v4pro_160x32768_k1024": case(160, 32768, 1024),
    # K' 512 and the column body
    "k512_1280x8192_k128": case(1280, 8192, 128),
    "msa_1280x8192_k16": case(1280, 8192, 16),
    "msa_1280x1024_v440_k16": case(1280, 1024, 16, 440),
    "msa_1280x480_v440_k16": case(1280, 480, 16, 440),
    "rtdetr_1x8400_k304": case(1, 8400, 304),
    "routed_32x128256_k64": case(32, 128256, 64),
}


def main():
    names = os.environ.get("TLI_CASES", "").split() or list(CASES)
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
                ttnn.ReadDeviceProfiler(dev)
                for a in args:
                    a.deallocate()
                print(f"CASE {name} ok", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                ttnn.synchronize_device(dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
