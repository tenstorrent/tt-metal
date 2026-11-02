# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Op-level gate for a3/prefill.py on real PQ2_0 layers.

  dequant  bf16 output vs codes * fp16 scale rounded to bf16 (expect bit-exact), bfp8 PCC
  rotate   vs BlockHadamard.rotate in float64
  mlp      prefill_mlp at S rows vs the float64 cpu_ref chain, and row 0 vs the A3 decode MLP
  timing   wall ms per call (synchronized, median of 5 after a warmup)

Usage: python -m models.demos.blackhole.bonsai2.native.a3.check_prefill [--layers 0,47] [--seqs 128,2048]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,47")
    ap.add_argument("--seqs", default="128,2048")
    args = ap.parse_args()

    import ttnn
    from models.demos.blackhole.bonsai2.native import cpu_ref
    from models.demos.blackhole.bonsai2.native.a3 import prefill as pf
    from models.demos.blackhole.bonsai2.native.a3.mlp import Shared, TernaryMLP
    from models.demos.blackhole.bonsai2.native.gguf_ternary import read_ternary

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    print(f"pid {os.getpid()} TT_VISIBLE_DEVICES={os.environ.get('TT_VISIBLE_DEVICES')}", flush=True)
    os.system(f"ls -l /proc/{os.getpid()}/fd 2>/dev/null | grep -o '/dev/tenstorrent/[0-9]*' | sort -u")

    def timed(f, n=5):
        f()
        ttnn.synchronize_device(device)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            r = f()
            ttnn.synchronize_device(device)
            ts.append((time.perf_counter() - t0) * 1e3)
            for t in (r if isinstance(r, tuple) else (r,)):
                ttnn.deallocate(t)
        return float(np.median(ts))

    rc = 0
    try:
        sh = Shared(device)
        rot = pf.rotator(sh)
        cache = os.path.join(os.environ["TT_CACHE_PATH"], "a3_mlp")
        rng = np.random.default_rng(0)
        for layer in [int(v) for v in args.layers.split(",")]:
            mlp = TernaryMLP(sh, layer, cache)
            t = {n: read_ternary(sh.gguf, f"blk.{layer}.ffn_{n}.weight") for n in ("gate", "up", "down")}
            # dequant, bf16: bit-exact vs codes * scale rounded to bf16
            wg, wu = pf.dequant(mlp.w_gu, mlp.s_gu, 5120, 17408, gu=True, dtype=ttnn.bfloat16)
            wd = pf.dequant(mlp.w_d, mlp.s_d, 17408, 5120, dtype=ttnn.bfloat16)
            for name, dev in (("gate", wg), ("up", wu), ("down", wd)):
                ref = torch.from_numpy(t[name].dequant().T.copy()).bfloat16()
                got = ttnn.to_torch(dev).reshape(ref.shape)
                exact = (got == ref).float().mean().item()
                print(f"L{layer} dequant {name} bf16: exact {exact * 100:.4f}%  pcc {cpu_ref.pcc(got.float().numpy(), ref.float().numpy()):.7f}", flush=True)
                rc |= exact < 1.0
                ttnn.deallocate(dev)
            wd8 = pf.dequant(mlp.w_d, mlp.s_d, 17408, 5120)
            ref = t["down"].dequant().T
            print(f"L{layer} dequant down bfp8: pcc {cpu_ref.pcc(ttnn.to_torch(wd8).float().numpy().reshape(ref.shape), ref):.7f}", flush=True)
            ttnn.deallocate(wd8)

            for s in [int(v) for v in args.seqs.split(",")]:
                x = torch.from_numpy(rng.standard_normal((s, 5120)).astype(np.float32)).bfloat16()
                xt = ttnn.from_torch(x.reshape(1, 1, s, 5120), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                xr = ttnn.to_torch(rot(xt)).float().numpy().reshape(s, 5120)
                xr_ref = sh.had.rotate(x.float().numpy())
                print(f"L{layer} S={s} rotate 5120: pcc {cpu_ref.pcc(xr, xr_ref):.7f} max|d| {np.abs(xr - xr_ref).max():.3e}", flush=True)
                y = ttnn.to_torch(pf.prefill_mlp(mlp, xt)).float().numpy().reshape(s, 5120)
                xb = x.float().numpy()
                g = cpu_ref.project_ternary(t["gate"], sh.had, xb)
                u = cpu_ref.project_ternary(t["up"], sh.had, xb)
                ref = cpu_ref.project_ternary(t["down"], sh.had, g / (1 + np.exp(-g)) * u)
                p = cpu_ref.pcc(y, ref)
                # decode path on row 0 (A3, exact ternary)
                x0 = ttnn.from_torch(x[:1].repeat(32, 1).reshape(1, 1, 32, 5120), dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
                y0 = ttnn.to_torch(mlp.forward(x0)).float().numpy()[0, 0, 0]
                p0 = cpu_ref.pcc(y[0], y0)
                rel = float(np.abs(y - ref).max() / np.abs(ref).max())  # scale-sensitive: PCC isn't (SCALE16 bug, 01:34)
                print(f"L{layer} S={s} prefill_mlp: pcc vs float64 {p:.6f}; row0 vs A3 decode {p0:.6f}; "
                      f"decode vs float64 {cpu_ref.pcc(y0, ref[0]):.6f}; max rel err {rel:.4f}", flush=True)
                rc |= p < 0.999 or p0 < 0.999 or not rel < 0.2
                ms = {
                    "dequant_gu": timed(lambda: pf.dequant(mlp.w_gu, mlp.s_gu, 5120, 17408, gu=True)),
                    "dequant_down": timed(lambda: pf.dequant(mlp.w_d, mlp.s_d, 17408, 5120)),
                    "rot5120": timed(lambda: rot(xt)),
                    "mlp": timed(lambda: pf.prefill_mlp(mlp, xt)),
                }
                print(f"L{layer} S={s} wall ms: " + "  ".join(f"{k} {v:.3f}" for k, v in ms.items()), flush=True)
                ttnn.deallocate(xt)
                ttnn.deallocate(x0)
    finally:
        ttnn.close_device(device)
    print("PASS" if not rc else "FAIL", flush=True)
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
