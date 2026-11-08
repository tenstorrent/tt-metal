# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""G1 gate for the Hunyuan3D-2.1 shape DiT: single calls on device vs the CPU fp32 goldens (dump_goldens.py).

usage: TT_VISIBLE_DEVICES=<BDF> python models/experimental/hunyuan3d_21/tests/run_dit_vs_golden.py GOLDEN_DIR CKPT CACHE
Prints PCC and relative L2 error per saved call (steps 0, mid, last) and the warm call time.
"""

import json
import sys
import time

import torch

import ttnn
from models.experimental.hunyuan3d_21.reference import dit_host as host
from models.experimental.hunyuan3d_21.tt.dit import TtHunyuanDiT


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def main():
    gdir, ckpt, cache = sys.argv[1:4]
    t0 = time.time()
    state = host.load_dit_state(ckpt, cache)
    t_load = time.time() - t0
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        t0 = time.time()
        model = TtHunyuanDiT(mesh_device=mesh)
        model.load_torch_state_dict(state)
        t_build = time.time() - t0
        cond = torch.load(f"{gdir}/cond.pt")["main"]  # conditioner output [1, 1370, 1024]
        cond = torch.cat([cond, torch.zeros_like(cond)])  # CFG: unconditional_embedding is zeros (encode_cond)
        res = {"load_s": round(t_load, 1), "build_s": round(t_build, 1), "calls": []}
        for call in torch.load(f"{gdir}/dit_calls.pt"):
            times = []
            for _ in range(2):  # first call compiles
                s = time.time()
                out = model(call["x"], call["t"], cond)
                times.append(time.time() - s)
            g = call["out"]
            res["calls"].append(
                {
                    "step": call["step"],
                    "pcc": round(pcc(out, g), 6),
                    "rel": round(rel(out, g), 5),
                    "pcc_cond": round(pcc(out[0], g[0]), 6),
                    "pcc_uncond": round(pcc(out[1], g[1]), 6),
                    "first_s": round(times[0], 2),
                    "warm_s": round(times[1], 2),
                }
            )
            print(json.dumps(res["calls"][-1]), flush=True)
        print("DIT_G1", json.dumps(res))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
