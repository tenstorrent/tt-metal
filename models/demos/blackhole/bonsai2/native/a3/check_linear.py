# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Op-level gate for prefill_linear on every TernaryLinear site: prefill rows vs float64
((codes * scales) @ H(D x)), and row 0 vs the A3 decode op. Wall ms per prefill call.

Usage: python -m models.demos.blackhole.bonsai2.native.a3.check_linear [--seqs 128,2048]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="128,2048")
    args = ap.parse_args()
    import ttnn
    from models.demos.blackhole.bonsai2.native import cpu_ref
    from models.demos.blackhole.bonsai2.native.a3 import linear as al
    from models.demos.blackhole.bonsai2.native.a3 import prefill as pf
    from models.demos.blackhole.bonsai2.native.a3.mlp import Shared
    from models.demos.blackhole.bonsai2.native.gguf_ternary import read_ternary

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    cache = os.path.join(os.environ["TT_CACHE_PATH"], "a3_linear")
    rc = 0
    try:
        sh = Shared(device)

        def gg(name):
            t = read_ternary(sh.gguf, name)
            return t.codes, t.scales

        sites = [
            ("gdn_out L0", "blk.0.ssm_out.weight", None),
            ("gdn_in L0", "blk.0.gdn_in", al.gdn_in_rows(sh, 0)),
            ("attn_qkv L3", "blk.3.attn_qkv_fused", al.attn_qkv_rows(sh, 3)),
            ("attn_o L3", "blk.3.attn_output.weight", None),
        ]
        if os.environ.get("CL_CACHE_HITS") == "1":
            # The model runs one cached program per site shape over all layers (runtime args change, compile args
            # don't). Same shapes again on other layers: these calls are program-cache hits with new tensors.
            sites += [("gdn_out L1", "blk.1.ssm_out.weight", None), ("gdn_out L2", "blk.2.ssm_out.weight", None),
                      ("attn_o L7", "blk.7.attn_output.weight", None), ("gdn_in L1", "blk.1.gdn_in", al.gdn_in_rows(sh, 1))]
        rng = np.random.default_rng(0)
        for label, name, rows in sites:
            codes, scales = rows if rows is not None else gg(name)
            shape = None if rows is None else (codes.shape[0], codes.shape[1])
            tl = al.TernaryLinear(sh, name, cache, rows=None if rows is None else (lambda r=rows: r), shape=shape)
            w = codes.astype(np.float64) * np.repeat(scales.astype(np.float64), 128, axis=1)  # [N, K]
            for s in [int(v) for v in args.seqs.split(",")]:
                x = torch.from_numpy(rng.standard_normal((s, tl.k)).astype(np.float32)).bfloat16()
                xt = ttnn.from_torch(x.reshape(1, s, tl.k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
                y = ttnn.to_torch(pf.prefill_linear(tl, xt)).float().numpy().reshape(s, tl.n)
                y2 = ttnn.to_torch(pf.prefill_linear(tl, xt)).float().numpy().reshape(s, tl.n)  # same call again: cache hit
                same = float(np.mean(y2 == y))
                ref = sh.had.rotate(x.float().numpy()) @ w.T
                x0 = ttnn.from_torch(x[:1].repeat(32, 1).reshape(1, 1, 32, tl.k), dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
                y0 = ttnn.to_torch(tl.forward(x0)).float().numpy().reshape(-1, tl.n)[0]
                p, p0 = cpu_ref.pcc(y, ref), cpu_ref.pcc(y[0], y0)
                rel = float(np.abs(y - ref).max() / np.abs(ref).max())  # scale-sensitive: PCC isn't (SCALE16 bug, 01:34)
                f = lambda: pf.prefill_linear(tl, xt)  # noqa: E731
                ttnn.deallocate(f())
                ttnn.synchronize_device(device)
                ts = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    r = f()
                    ttnn.synchronize_device(device)
                    ts.append((time.perf_counter() - t0) * 1e3)
                    ttnn.deallocate(r)
                print(f"{label} [{tl.k}->{tl.n}] S={s}: pcc vs float64 {p:.6f}; row0 vs A3 decode {p0:.6f}; "
                      f"{np.median(ts):.3f} ms; repeat call bit-same {same * 100:.2f}%; max rel err {rel:.4f}", flush=True)
                rc |= p < 0.999 or p0 < 0.999 or same < 1.0 or not rel < 0.05
                ttnn.deallocate(xt)
                ttnn.deallocate(x0)
    finally:
        ttnn.close_device(device)
    print("PASS" if not rc else "FAIL", flush=True)
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
