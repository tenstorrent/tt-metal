"""Benchmark single-card decode perf: program cache on, sparse vs dense MoE.
Run: TT_DEVICE=1 /home/yito/work/run_zaya.sh python models/demos/zaya1_8b/tests/run_decode_bench.py
"""
import os
import time
import torch
import ttnn

from models.demos.zaya1_8b.tt.model_args import ZayaWeights
from models.demos.zaya1_8b.tt.model import ZayaModel
from models.demos.zaya1_8b.tt.cache import ZayaCache

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "reference", "golden")
N = 8


def time_decode(model, ids, sparse, warmup=2, n=N):
    c = ZayaCache()
    cur = model.prefill(ids, c)
    for _ in range(warmup):                      # warmup populates program cache
        cur = model.decode_step(cur, c, sparse=sparse)
    t = time.time()
    for _ in range(n):
        cur = model.decode_step(cur, c, sparse=sparse)
    return (time.time() - t) / n


def main():
    ids = torch.load(os.path.join(GOLDEN, "inputs.pt"), weights_only=False)["input_ids"].to(torch.int32)
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 1)))
    device.enable_program_cache()
    try:
        model = ZayaModel(device, w=ZayaWeights(), verbose=False)

        sp = time_decode(model, ids, sparse=True)
        ce_sp = device.num_program_cache_entries()
        dn = time_decode(model, ids, sparse=False)
        ce_dn = device.num_program_cache_entries()

        print(f"\n=== decode perf (program cache ON, n={N}) ===")
        print(f"  sparse MoE decode: {sp*1000:8.1f} ms/tok ({1/sp:.2f} tok/s)  cache_entries={ce_sp}")
        print(f"  dense  MoE decode: {dn*1000:8.1f} ms/tok ({1/dn:.2f} tok/s)  cache_entries={ce_dn}")
        print(f"  dense/sparse ratio: {dn/sp:.2f}x  (>1 means sparse faster)")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
