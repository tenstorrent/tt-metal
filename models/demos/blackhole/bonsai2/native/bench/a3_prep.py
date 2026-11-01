# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A3 prep kernel: x' = H(D x) scattered into every worker's block-masked buffer.

Checks each worker's [8, K] shard against mask_x(bf16(rotate(x))), and times the op.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a3_prep [--widths 5120,17408]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .a1_bfp2_shard import all_kernel_us
from .a1_calib import device_kernel_us

GGUF = os.path.expanduser("~/avan/models/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="5120,17408")
    ap.add_argument("--wpb", type=int, default=4)
    ap.add_argument("--upc", default="1,2")
    ap.add_argument("--calls", type=int, default=9)
    args = ap.parse_args()
    for key, val in (("TT_METAL_DEVICE_PROFILER", "1"), ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
                     ("TT_METAL_PROFILER_MID_RUN_DUMP", "1")):
        os.environ.setdefault(key, val)

    import ttnn
    from models.demos.blackhole.bonsai2.native.a3 import op as a3
    from models.demos.blackhole.bonsai2.native.gguf_ternary import read_gguf
    from models.demos.blackhole.bonsai2.native.hadamard import BlockHadamard

    had = BlockHadamard.from_metadata(read_gguf(GGUF).kv)
    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    rc = 0
    try:
        workers = a3.worker_cores(device, args.wpb)
        for k in [int(w) for w in args.widths.split(",")]:
            m = a3.hadamard_tensor(device, had, k)
            x8 = a3.masked_input_buffer(device, workers, k)
            x = torch.randn(1, 1, 32, k).bfloat16().float()
            want = a3.mask_x(torch.from_numpy(had.rotate(x[0, 0, :1].numpy()).astype(np.float32)).bfloat16().float()
                             .flatten()).numpy()
            for src, mem in (("tile32 L1", ttnn.L1_MEMORY_CONFIG), ("tiny L1", ttnn.L1_MEMORY_CONFIG)):
                tile32 = src.startswith("tile32")
                xin = x if tile32 else x[:, :, :1]
                xt = ttnn.from_torch(xin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem,
                                     tile=None if tile32 else ttnn.Tile([1, 32]))
                for upc in [int(u) for u in args.upc.split(",")]:
                    try:
                        a3.prep(xt, m, x8, workers, src_tile32=tile32, units_per_core=upc)
                        ttnn.synchronize_device(device)
                        got = ttnn.to_torch(x8).float().numpy().reshape(len(workers), 8, k)
                        err = float(np.abs(got - want[None]).max())
                        np.savez(os.path.expanduser(f"~/avan/native2bit/agentA/tmp/prep_{k}.npz"), got=got, want=want,
                                 x=x[0, 0, 0].numpy(), signs=had.signs[k])
                        ulp = 2.0 ** (np.floor(np.log2(np.abs(want).max())) - 7)  # bf16 ulp at the max
                        devs = []
                        for scatter in (True, False):
                            device_kernel_us(ttnn, device)
                            for _ in range(args.calls):
                                a3.prep(xt, m, x8, workers, src_tile32=tile32, units_per_core=upc, scatter=scatter)
                            ttnn.synchronize_device(device)
                            devs.append(float(np.median(all_kernel_us(ttnn, device))))
                        for dbg in (1, 2, 3):
                            device_kernel_us(ttnn, device)
                            for _ in range(args.calls):
                                a3.prep(xt, m, x8, workers, src_tile32=tile32, units_per_core=upc, debug=dbg)
                            ttnn.synchronize_device(device)
                            devs.append(float(np.median(all_kernel_us(ttnn, device))))
                        print(f"   debug: no M reads {devs[2]:.1f}, no x reads {devs[3]:.1f}, neither {devs[4]:.1f}")
                        a3.prep(xt, m, x8, workers, src_tile32=tile32, units_per_core=upc)
                        dev, dev_noscatter = devs[:2]
                        ok = err <= ulp
                        rc |= 0 if ok else 1
                        print(f"K={k:5d} {src:11s} upc={upc} {dev:6.1f} us (no scatter {dev_noscatter:5.1f})  max|err| {err:.2e} (1 bf16 ulp at max "
                              f"{ulp:.2e}) zero rows ok {bool((got[:, :, :] * (want[None] == 0)).max() == 0)} "
                              f"{'ok' if ok else 'FAIL'}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"K={k:5d} {src:11s} upc={upc} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                        rc = 1
                ttnn.deallocate(xt)
            ttnn.deallocate(m)
            ttnn.deallocate(x8)
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
