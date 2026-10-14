#!/usr/bin/env python3
"""Scratch, not for merge. topk_large_indices device time under the device profiler (per case a signpost, OPS_NMEAS calls
after 2 warm-up calls, an "end" signpost), or with TLI_DUMP=<file> every case's indices saved for an arm-to-arm compare."""
import gc
import os

import torch
import ttnn

N_WARM, N_MEAS = 2, int(os.environ.get("OPS_NMEAS", "8"))


def case(rows, n, k, valid_length=None, meta=False):
    return rows, n, k, valid_length, meta


# K' 2048 production shapes (DeepSeek V3.2 indexer prefill, bounded cache and decode, GLM 5.3 indexer, DeepSeek V3
# rows of 2 to 64 chunks) and the DeepSeek V4 Pro indexer's K' 1024 rows as controls (their kernels do not change).
CASES = {
    "prefill_640x51200_k1536": case(640, 51200, 1536),
    "bounded_2x102400_v56320_k1536": case(2, 102400, 1536, 56320),
    "decode_160x524288_k2048": case(160, 524288, 2048),
    "decode_meta_160x524288_k2048": case(160, 524288, 2048, 524288, meta=True),
    "glm95k_160x102400_k2048": case(160, 102400, 2048, 102400),
    "dsv3_64x131072_k2048": case(64, 131072, 2048),
    "dsv3_128x16384_k2048": case(128, 16384, 2048),
    "dsv3_192x16384_k2048": case(192, 16384, 2048),
    "dsv3_200x8192_k2048": case(200, 8192, 2048),
    "dsv3_32x65536_v40000_k2048": case(32, 65536, 2048, 40000),
    "k2048_640x4096": case(640, 4096, 2048),
    "k2048_640x6144": case(640, 6144, 2048),
    "v4pro_640x3328_v3310_k1024": case(640, 3328, 1024, 3310),
    "v4pro_640x2560_k1024": case(640, 2560, 1024),
    "v4pro_640x8192_k1024": case(640, 8192, 1024),
    "v4pro_160x32768_k1024": case(160, 32768, 1024),
}


def setup(dev, rows, n, k, vl, meta):
    torch.manual_seed(2026)
    x = torch.randn((rows, n), dtype=torch.float32).to(torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    del x
    if meta:
        v = torch.tensor([vl], dtype=torch.int64).reshape(1, 1, 1, 1)
        m = ttnn.from_torch(v, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return [t, m], lambda: ttnn.experimental.topk_large_indices(t, k=k, valid_length_tensor=m)
    kw = {} if vl is None else {"valid_length": vl}
    return [t], lambda: ttnn.experimental.topk_large_indices(t, k=k, **kw)


def main():
    dump = os.environ.get("TLI_DUMP")
    names = os.environ.get("OPS_CASES", "").split() or list(CASES)
    if not dump:
        from tracy import signpost
    dev = ttnn.open_device(device_id=0)
    out = {}
    try:
        for name in names:
            gc.collect()
            args = []
            try:
                args, run = setup(dev, *CASES[name])
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
                msg = " ".join(str(e).split())
                size = msg[msg.find("Program size"):][:120] if "Program size" in msg else ""
                print(f"CASE {name} FAILED {type(e).__name__}: {msg[:200]} {size}", flush=True)
                ttnn.synchronize_device(dev)
            finally:
                for a in args:
                    a.deallocate()
    finally:
        ttnn.close_device(dev)
    if dump:
        torch.save(out, dump)
        print(f"DUMP DONE {len(out)}", flush=True)


if __name__ == "__main__":
    main()
