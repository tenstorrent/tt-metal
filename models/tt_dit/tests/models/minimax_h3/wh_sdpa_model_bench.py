# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The model's WH SDPA (wh_sdpa_minimax_h3.WhSdpa) on one chip as a 1x1 mesh: timing, and a check against torch.

    python wh_sdpa_model_bench.py HEADS Q_ROWS KEYS VALID [--readers=4] [--repeats=3] [--check=HEADS]

The check compares the first HEADS heads with fp32 torch (meaningful at small shapes: at 110592 random keys the
softmax is nearly flat and the release's exp approximation dominates the PCC).
"""

import sys
import time

import torch

import ttnn
from models.tt_dit.models.transformers.minimax_h3.wh_sdpa_minimax_h3 import WhSdpa


def opt(name: str, default: str) -> str:
    return next((x.split("=", 1)[1] for x in sys.argv if x.startswith(f"--{name}=")), default)


def main() -> None:
    heads, q_rows, keys, valid = [int(x) for x in sys.argv[1:5]]
    readers, repeats, check = int(opt("readers", "2")), int(opt("repeats", "3")), int(opt("check", "0"))
    torch.manual_seed(0)
    q = torch.randn(1, heads, q_rows, 128).bfloat16()
    k = torch.randn(1, heads, keys, 128).bfloat16()
    v = torch.randn(1, heads, keys, 128).bfloat16()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        dram = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tq, tk = ttnn.from_torch(q, **dram), ttnn.from_torch(k, **dram)
        tvt = ttnn.from_torch(v.transpose(-2, -1).contiguous(), **dram)
        n = ttnn.from_torch(
            torch.tensor([[[[valid]]]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        op = WhSdpa(mesh, readers=readers)
        times = []
        out = None
        for _ in range(repeats):
            if out is not None:
                ttnn.deallocate(out)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            out = op(tq, tk, tvt, n)
            ttnn.synchronize_device(mesh)
            times.append((time.perf_counter() - t0) * 1e3)
        print("times ms:", " ".join(f"{t:.2f}" for t in times))
        print(
            f"H{heads} Nq {q_rows} keys {keys} valid {valid} readers {readers}: median {sorted(times)[len(times) // 2]:.2f} ms"
        )
        if check:
            o = ttnn.to_torch(out).float()
            for h in range(check):
                s = (q[0, h].float() @ k[0, h, :valid].float().T) / 128**0.5
                ref = torch.softmax(s, dim=-1) @ v[0, h, :valid].float()
                got = o[0, h]
                pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
                rel = ((got - ref).norm() / ref.norm()).item()
                print(f"head {h}: PCC {pcc:.6f} rel {rel:.3e} max abs {(got - ref).abs().max():.3e}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
