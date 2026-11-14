"""fuse-sparse-merge B2: threshold selection building blocks. TEST=h1: pass-1 histogram + latency. Replicated data on 1x8 slot, checks chip 0."""
import json, os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tt"))
from dk_dev import timed_traced
from dm_block import close_row, open_row
import ttnn
import hy4_sel as SEL

def emit(d): print("RES " + json.dumps(d), flush=True)

def skey(x):  # x int32 bf16 bits
    return torch.where((x & 0x8000) != 0, (~x) & 0xFFFF, x | 0x8000)

def main():
    full, md = open_row()
    try:
        for n in [int(v) for v in os.environ.get("NS", "131072,32768").split(",")]:
            for S in [int(v) for v in os.environ.get("SS", "1,2").split(",")]:
                sc = (torch.randn(S, n) * 3).to(torch.bfloat16)
                sd = ttnn.from_torch(sc.reshape(1, 1, S, n), device=md, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(md))
                out = SEL.hist1(sd, S)
                got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).reshape(8, S, 256)[0].long()
                bits = sc.view(torch.int16).long() & 0xFFFF
                hi = skey(bits) >> 8
                ref = torch.stack([torch.bincount(hi[s], minlength=256) for s in range(S)])
                us = timed_traced(md, lambda: SEL.hist1(sd, S))
                emit({"test": "h1", "n": n, "S": S, "exact": bool((got == ref).all()), "us": us})
    finally:
        close_row(full, md)
main()
