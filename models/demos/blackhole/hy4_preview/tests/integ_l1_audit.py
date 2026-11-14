"""integ L1 audit (1x8 slot): persistent L1 allocated by XMerge (HY4_XM_NSPLIT), fused-iHC constants (one call at T=1 and T=4), reported via the allocator."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt.hy4_ihc_fused import Hy4IHCFused


def l1(md):
    v = ttnn.get_memory_view(md, ttnn.BufferType.L1)
    return dict(alloc_per_bank=v.total_bytes_allocated_per_bank, free_per_bank=v.total_bytes_free_per_bank, largest_free=v.largest_contiguous_bytes_free_per_bank)


def R(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
    try:
        R(event="empty", **l1(md))
        from models.demos.blackhole.hy4_preview.tt import hy4_xmerge

        xm = hy4_xmerge.XMerge(md, 1, nsplit=int(os.environ.get("HY4_XM_NSPLIT", "8")))
        R(event="xmerge", **l1(md))
        D = 6144
        ih = Hy4IHCFused(md, torch.randn(8, 4 * D), torch.randn(8), (torch.tensor([1.0, 1.0]) , torch.tensor([1.0, 1.0])) if False else torch.ones(2, 1) if False else [torch.tensor(0.5), torch.tensor(0.5)])
        for T in (1, 4):
            x = ttnn.from_torch(torch.randn(1, 1, 4 * T, D), device=md, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(md))
            h, _, pp = ih.pre(x, ih.norm_weight(torch.ones(D), T))
            ttnn.synchronize_device(md)
            R(event=f"ihc_T{T}", **l1(md))
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
