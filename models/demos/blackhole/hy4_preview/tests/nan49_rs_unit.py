"""NAN49: standalone all-reduce (RS+AG, ring, ping-pong semaphores) on a 1x8 slot: does ping-pong set 1 of the reduce-scatter give NaN/garbage without any other op?"""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    mesh4 = os.environ.get("MESH4") == "1"
    if mesh4:
        from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
        from models.demos.gpt_oss.config import mesh_4x8

        full, md = None, open_mesh()
    else:
        full, md = open_row()
    try:
        extra = int(os.environ.get("EXTRA_SEMS", "0"))
        _hold = [
            ttnn.create_global_semaphore(
                md, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}), 0
            )
            for _ in range(extra)
        ]
        ccl = CCLManager(
            md,
            num_links=int(os.environ.get("LINKS", "2")),
            topology=getattr(ttnn.Topology, os.environ.get("TOPO", "Ring")),
        )
        mc = mesh_4x8() if mesh4 else mesh_1x8()
        res(
            event="sem_addr",
            rs=[hex(ttnn.get_global_semaphore_address(s)) for s in ccl.rs_ping_pong_semaphores],
            ag=[hex(ttnn.get_global_semaphore_address(s)) for s in ccl.ag_ping_pong_semaphores],
            bar=[hex(ttnn.get_global_semaphore_address(s)) for s in ccl.barrier_semaphore],
        )
        n = int(os.environ.get("N", "512"))
        g = torch.Generator().manual_seed(1)
        R = 4 if mesh4 else 1
        for it in range(int(os.environ.get("ITERS", "8"))):
            x = torch.randn(R * 8, 1, n, 6144, generator=g)  # per-device partials
            t = (
                upload(md, x.reshape(8, 1, n, 6144).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, shard_dim=0)
                if False
                else ttnn.from_torch(
                    x.to(torch.bfloat16),
                    device=md,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            rs_i, ag_i, b_i = ccl.rs_ping_pong_idx, ccl.ag_ping_pong_idx, ccl.barrier_idx
            o = mc.allreduce(t, ccl, axis=1)
            ttnn.synchronize_device(md)
            xs = x.to(torch.bfloat16).float().reshape(R, 8, n, 6144)
            want_r = xs.sum(1)
            bad = []
            for d, td in enumerate(ttnn.get_device_tensors(o)):
                h = ttnn.to_torch(td).float().reshape(n, 6144)
                want = want_r[d // 8]
                fin = bool(torch.isfinite(h).all())
                err = float((h - want).abs().max()) if fin else float("inf")
                if (not fin) or err > 0.5:
                    bad.append((d, fin, round(err, 3) if fin else None))
            res(event="iter", it=it, rs_idx=rs_i, ag_idx=ag_i, bar_idx=b_i, bad=bad)
            ttnn.deallocate(o)
    finally:
        (close_mesh(md) if mesh4 else close_row(full, md))


if __name__ == "__main__":
    main()
