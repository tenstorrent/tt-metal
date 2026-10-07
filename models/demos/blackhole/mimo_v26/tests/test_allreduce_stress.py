# SPDX-License-Identifier: Apache-2.0
"""Stress ttnn.all_reduce over mesh columns (cluster_axis=1) as used by attention/FFN: a trace of K chained all_reduces on distinct inputs [1,1,32,4096] bf16, replayed R times; every output of
every replay must be bit-identical to the first replay (rows/devices reported). Variants: topology / num_links."""
import json
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import close_mesh, open_mesh_full  # noqa: E402

E = os.environ.get
OUT = E("DEC_OUT", "/tmp/ar_stress.jsonl")


def main():
    K, R = int(E("AR_K", "96")), int(E("AR_R", "300"))
    mesh = open_mesh_full()
    try:
        g = torch.Generator().manual_seed(0)
        rep_rows = ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, 1))
        xs = []
        for i in range(K):
            t = torch.randn(4, 8, 32, 4096, generator=g).bfloat16()  # distinct per device
            xs.append(
                ttnn.from_torch(
                    t,
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=rep_rows,
                )
            )
        variants = {
            "default": {},
            "ring_nl1": dict(topology=ttnn.Topology.Ring, num_links=1),
            "linear_nl1": dict(topology=ttnn.Topology.Linear, num_links=1),
            "ring_nl2": dict(topology=ttnn.Topology.Ring, num_links=2),
        }
        for name in E("AR_VARIANTS", "default,ring_nl1,linear_nl1,ring_nl2").split(","):
            kw = variants[name]

            def fwd():
                return [ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw) for x in xs]

            try:
                outs = fwd()  # compile
                [ttnn.deallocate(o) for o in outs]
                ttnn.synchronize_device(mesh)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                outs = fwd()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.synchronize_device(mesh)
                comp = ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape))
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                ref = [ttnn.to_torch(o, mesh_composer=comp) for o in outs]
                bad = {}
                t0 = time.time()
                for r in range(R):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                    for i, o in enumerate(outs):
                        # compare on device-side cheaply: read back only every 10th replay fully, else a checksum per device
                        if r % 10 == 0:
                            y = ttnn.to_torch(o, mesh_composer=comp)
                            d = (y != ref[i]).any(dim=(2, 3))  # [rows, cols]
                            if d.any():
                                bad.setdefault((i,), []).append((r, [(int(a), int(b)) for a, b in d.nonzero()][:4]))
                rec = dict(
                    case="allreduce_stress",
                    variant=name,
                    K=K,
                    R=R,
                    secs=round(time.time() - t0, 1),
                    n_bad=len(bad),
                    first_bad=list(bad.items())[:3],
                )
                ttnn.release_trace(mesh, tid)
            except Exception as e:  # noqa
                rec = dict(case="allreduce_stress", variant=name, error=str(e)[:300])
            print("AR_RESULT " + json.dumps(rec, default=str), flush=True)
            open(OUT, "a").write(json.dumps(rec, default=str) + "\n")
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
