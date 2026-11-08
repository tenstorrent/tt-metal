# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-block relative error of the TT DiT and of CPU bf16 against CPU fp32 for one saved call (block_dump.py).
usage: python compare_blocks.py GOLDEN_DIR CKPT CACHE [call_index]"""
import sys

import torch

import ttnn
from models.experimental.hunyuan3d_21.reference import dit_host as host
from models.experimental.hunyuan3d_21.tt.dit import Precision, TtHunyuanDiT

gdir, ckpt, cache = sys.argv[1:4]
idx = int(sys.argv[4]) if len(sys.argv) > 4 else 1
flags = dict(kv.split("=") for kv in sys.argv[5:])  # e.g. fp32_gate=0
prec = Precision(**{k: v == "1" for k, v in flags.items()})
print("precision", prec)
call = torch.load(f"{gdir}/dit_calls.pt")[idx]
ref = torch.load(f"{gdir}/blocks_step{call['step']}.pt")
cond = torch.load(f"{gdir}/cond.pt")["main"]
cond = torch.cat([cond, torch.zeros_like(cond)])
rel = lambda a, b: float((a.double() - b.double()).norm() / b.double().norm())
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    m = TtHunyuanDiT(mesh_device=mesh, precision=prec)
    m.load_torch_state_dict(host.load_dit_state(ckpt, cache))
    m.collect = []
    m(call["x"], call["t"], cond)
    print("block  tt_rel   bf16_rel  ratio   (t-token tt / bf16)")
    for i, h in enumerate(m.collect):
        if i < 13:
            continue
        f32, b16 = ref["fp32"][i], ref["bf16"][i]
        a, b = rel(h, f32), rel(b16, f32)
        at, bt = rel(h[:, -1:], f32[:, -1:]), rel(b16[:, -1:], f32[:, -1:])
        print(f"{i:5d}  {a:.5f}  {b:.5f}  {a / b:5.2f}   ({at:.5f} / {bt:.5f})", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
