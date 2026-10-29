# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does the DFlash2 drafter's extend write every row's K and V into its paged context cache? Per page-table width:
extend 8 random rows at a block-aligned slot (and 3 rows at an unaligned one), read the cache back (device 0) and
report the rows whose K / V stayed zero. Also: does ttnn.pad of the [1, R, NKV, HD] K/V alias its input?

usage: python -m qwen36_resident.tests.dfl_kv_write [--nb 64,65,144,145,160]
"""
import argparse
import faulthandler
import sys

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.dflash import build_drafter, tap_buffers
from qwen36_resident.tests.long_ctx import close_mesh, log, open_mesh

ap = argparse.ArgumentParser()
ap.add_argument("--nb", default="64,65,144,145,160")
a = ap.parse_args()
faulthandler.dump_traceback_later(600, repeat=True, file=sys.stderr)
mesh = open_mesh()
try:
    ck = Q.Checkpoint(Q.checkpoint_dir())
    dr = build_drafter(mesh, ck)
    nt, R, H = len(dr.taps), dr.block, 5120
    g = torch.Generator().manual_seed(3)

    # pad aliasing
    x = ttnn.from_torch(
        torch.randn(1, R, 2, 128),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    xp = ttnn.pad(x, [1, R, 32, 128], [0, 0, 0, 0], 0.0)
    log(
        f"pad: in addr {x.buffer_address()} out addr {xp.buffer_address()} (alias {x.buffer_address() == xp.buffer_address()})"
    )

    def rows_state(kv, positions):
        out = []
        for li, (kc, vc) in enumerate(kv):
            k = ttnn.to_torch(ttnn.get_device_tensors(kc)[0]).float()
            v = ttnn.to_torch(ttnn.get_device_tensors(vc)[0]).float()
            kz = [p for p in positions if k[p // 64, :, p % 64].norm() == 0]
            vz = [p for p in positions if v[p // 64, :, p % 64].norm() == 0]
            out.append((li, kz, vz))
        return out

    for nb in [int(s) for s in a.nb.split(",")]:
        taps = tap_buffers(mesh, nt, R)
        dr.alloc(torch.arange(nb, dtype=torch.int32).reshape(1, nb), 64, taps)
        for s0, n in ((64, 8), (200, 3), (72, 8)):
            host = tap_buffers(mesh, nt, R, torch.randn(nt, R, H, generator=g) * 2)
            for t, buf in zip(host, taps):
                ttnn.copy(t, buf)
                ttnn.deallocate(t)
            dr.extend_context(s0, n, traced=False)
            ttnn.synchronize_device(mesh)
            st = rows_state(dr.kv, list(range(s0, s0 + n)))
            log(f"nb={nb} extend {n} rows at {s0}: zero rows (layer, K, V) {[s for s in st if s[1] or s[2]]}")
        dr.free()
        for t in taps:
            ttnn.deallocate(t)
finally:
    close_mesh(mesh)
