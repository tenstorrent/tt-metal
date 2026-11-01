# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug: v2 ternary_linear on gate alone (gu=0, tiny out) and fused gate|up vs cpu_ref."""
import os

import numpy as np
import torch


def main():
    import ttnn
    from models.demos.blackhole.bonsai2.native import cpu_ref
    from models.demos.blackhole.bonsai2.native.a3 import op as a3
    from models.demos.blackhole.bonsai2.native.a3.mlp import Shared
    from models.demos.blackhole.bonsai2.native.gguf_ternary import read_ternary

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    try:
        sh = Shared(device)
        x = torch.randn(1, 5120).bfloat16().float().numpy()
        g = read_ternary(sh.gguf, "blk.0.ffn_gate.weight")
        u = read_ternary(sh.gguf, "blk.0.ffn_up.weight")
        rg = cpu_ref.project_ternary(g, sh.had, x).flatten()
        ru = cpu_ref.project_ternary(u, sh.had, x).flatten()
        xt = ttnn.from_torch(torch.from_numpy(np.repeat(x, 32, 0)).reshape(1, 1, 32, 5120), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
        a3.prep(xt, sh.m[5120], sh.x8[5120], sh.workers, src_tile32=True)
        w, s = a3.weights_for_device(device, g.codes.T.astype(np.float32), g.scales.T.astype(np.float32))
        o = ttnn.from_torch(torch.zeros(1, 1, 1, 17408), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            tile=ttnn.Tile([1, 32]), device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
        a3.ternary_linear(sh.x8[5120], w, s, o, sh.workers, gu=False)
        yg = ttnn.to_torch(o).float().numpy().flatten()
        print("gate alone, tiny out: pcc", cpu_ref.pcc(yg, rg), flush=True)
        o32 = ttnn.from_torch(torch.zeros(1, 1, 32, 17408), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
        a3.ternary_linear(sh.x8[5120], w, s, o32, sh.workers, gu=False)
        print("gate alone, tile32 out: pcc", cpu_ref.pcc(ttnn.to_torch(o32).float().numpy()[0, 0, 0], rg), flush=True)
        gc = a3.interleave_cols(g.codes.T.astype(np.float32), u.codes.T.astype(np.float32))
        gs = a3.interleave_cols(g.scales.T.astype(np.float32), u.scales.T.astype(np.float32))
        w2, s2 = a3.weights_for_device(device, gc, gs)
        a3.ternary_linear(sh.x8[5120], w2, s2, o, sh.workers, gu=True)
        yh = ttnn.to_torch(o).float().numpy().flatten()
        rh = rg / (1 + np.exp(-rg)) * ru
        print("fused gu: pcc vs silu(g)*u", cpu_ref.pcc(yh, rh), " vs g", cpu_ref.pcc(yh, rg), " vs u",
              cpu_ref.pcc(yh, ru), " vs silu(g)", cpu_ref.pcc(yh, rg / (1 + np.exp(-rg))), " vs g*u",
              cpu_ref.pcc(yh, rg * ru), flush=True)
        np.savez(os.path.expanduser("~/avan/native2bit/agentA/tmp/gu_dbg.npz"), yh=yh, rg=rg, ru=ru, yg=yg)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
