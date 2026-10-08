# SPDX-License-Identifier: Apache-2.0
"""Single-chip probe: error of ttnn.matmul (x bf16 [32,4096] @ W [4096,2048]) vs an fp64 reference for each math fidelity and weight dtype,
to learn which fidelity phases carry the mantissa bits of bf16 activations / bfp8 weights on Blackhole. fp32 DEST accumulation on.
Env: PROBE_OUT (jsonl)."""
import json
import os

import torch

import ttnn

OUT = os.environ.get("PROBE_OUT", "/tmp/fid_probe.jsonl")
FID = dict(
    LoFi=ttnn.MathFidelity.LoFi,
    HiFi2=ttnn.MathFidelity.HiFi2,
    HiFi3=ttnn.MathFidelity.HiFi3,
    HiFi4=ttnn.MathFidelity.HiFi4,
)


def rel(a, b):
    return float((a - b).norm() / b.norm())


def test_probe():
    dev = ttnn.open_device(device_id=0)
    try:
        g = torch.Generator().manual_seed(0)
        K, N, M = 4096, 2048, 32
        x = torch.randn(M, K, generator=g).bfloat16()
        w = (torch.randn(K, N, generator=g) * 0.02).bfloat16()
        tx = ttnn.from_torch(x.reshape(1, 1, M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        for wname, wdt in (("bfp8", ttnn.bfloat8_b), ("bf16", ttnn.bfloat16)):
            tw = ttnn.from_torch(w.reshape(1, 1, K, N), dtype=wdt, layout=ttnn.TILE_LAYOUT, device=dev)
            wq = ttnn.to_torch(tw).reshape(K, N).double()  # exactly what the device holds
            ref = x.double() @ wq
            for fname, f in FID.items():
                ck = ttnn.init_device_compute_kernel_config(
                    dev.arch(), math_fidelity=f, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
                )
                o = ttnn.matmul(tx, tw, compute_kernel_config=ck, dtype=ttnn.float32)
                o = ttnn.to_torch(o).reshape(M, N).double()
                r = dict(w=wname, fid=fname, rel=rel(o, ref))
                print("PROBE", json.dumps(r), flush=True)
                open(OUT, "a").write(json.dumps(r) + "\n")
    finally:
        ttnn.close_device(dev)
