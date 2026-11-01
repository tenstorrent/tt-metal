# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A3 TernaryLinear on one projection (default the LM head, output.weight 5120 x 248320) vs the
float64 reference rotate(x) @ (codes * scales).T, plus device time.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a3_lmhead [--name output.weight]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .a1_bfp2_shard import all_kernel_us
from .a1_calib import device_kernel_us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="output.weight", help="GGUF tensor, or gdn_in:<layer> / attn_qkv:<layer> for the reordered rows")
    ap.add_argument("--calls", type=int, default=5)
    ap.add_argument("--cache", default=os.path.expanduser("~/avan/native2bit/agentA/tt_cache/pq2_0/a3_linear"))
    args = ap.parse_args()
    for key, val in (("TT_METAL_DEVICE_PROFILER", "1"), ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
                     ("TT_METAL_PROFILER_MID_RUN_DUMP", "1")):
        os.environ.setdefault(key, val)

    import ttnn
    from models.demos.blackhole.bonsai2.native import cpu_ref
    from models.demos.blackhole.bonsai2.native.a3.linear import TernaryLinear, attn_qkv_rows, gdn_in_rows
    from models.demos.blackhole.bonsai2.native.a3.mlp import Shared
    from models.demos.blackhole.bonsai2.native.gguf_ternary import read_ternary

    os.makedirs(args.cache, exist_ok=True)
    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    rc = 0
    try:
        sh = Shared(device)
        if args.name.split(":")[0] in ("gdn_in", "attn_qkv"):
            kind, layer = args.name.split(":")[0], int(args.name.split(":")[1])
            codes, scales = (gdn_in_rows if kind == "gdn_in" else attn_qkv_rows)(sh, layer)
            cache_name = f"blk.{layer}.gdn_in" if kind == "gdn_in" else f"blk.{layer}.attn_qkv_fused"
            lin = TernaryLinear(sh, cache_name, args.cache, out_mem=ttnn.DRAM_MEMORY_CONFIG, rows=(codes, scales))
            w = codes.astype(np.float64) * np.repeat(scales.astype(np.float64), 128, 1)
        else:
            lin = TernaryLinear(sh, args.name, args.cache, out_mem=ttnn.DRAM_MEMORY_CONFIG)
            w = read_ternary(sh.gguf, args.name).dequant().astype(np.float64)
        x = torch.randn(1, lin.k).bfloat16().float().numpy()
        ref = (sh.had.rotate(x) @ w.T).flatten()  # = cpu_ref.project_ternary
        xt = ttnn.from_torch(torch.from_numpy(x).reshape(1, 1, 1, lin.k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y = ttnn.to_torch(lin.forward(xt)).float().numpy().reshape(-1)[: lin.n]
        p = cpu_ref.pcc(y, ref)
        top = int(np.argmax(y)) == int(np.argmax(ref))
        device_kernel_us(ttnn, device)
        for _ in range(args.calls):
            ttnn.deallocate(lin.forward(xt))
        ttnn.synchronize_device(device)
        d = np.array(all_kernel_us(ttnn, device)).reshape(args.calls, -1)
        ok = p >= 0.9999
        rc = 0 if ok else 1
        print(f"{args.name} [{lin.n}, {lin.k}]: pcc {p:.7f} argmax match {top} {'ok' if ok else 'FAIL'}; "
              f"device us per program (median): {np.median(d, 0).round(1).tolist()} total {np.median(d.sum(1)):.1f}",
              flush=True)
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
