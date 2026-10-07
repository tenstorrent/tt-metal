# SPDX-License-Identifier: Apache-2.0
"""rms_decode_sharded vs the interleaved ttnn.rms_norm on decode rows (mesh, real-ish weight): max abs / rel diff, and traced device time of 50 repeated norms of each kind."""
import json
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import close_mesh, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.tt.layer import EPS, _k_norm, rms_decode_sharded  # noqa: E402


def main():
    mesh = open_mesh_full()
    try:
        g = torch.Generator().manual_seed(0)
        for rows in (1, 8, 32):
            x = (torch.randn(1, 1, 32, 4096, generator=g) * 3).bfloat16()
            x[:, :, rows:] = 0
            w = (torch.rand(1, 1, 128, 32, generator=g) + 0.5).bfloat16()
            xt = ttnn.from_torch(
                x,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            xt = ttnn.reshape(xt, (1, 1, rows, 4096)) if rows != 32 else xt
            wt = ttnn.from_torch(
                w,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            a = ttnn.rms_norm(xt, epsilon=EPS, weight=wt, compute_kernel_config=_k_norm())
            b = rms_decode_sharded(xt, wt)
            ha = ttnn.to_torch(ttnn.get_device_tensors(a)[0]).float()
            hb = ttnn.to_torch(ttnn.get_device_tensors(b)[0]).float()
            res = dict(
                rows=rows,
                max_abs=float((ha - hb).abs().max()),
                rel_l2=float((ha - hb).norm() / ha.norm()),
                equal=bool(torch.equal(ha, hb)),
            )

            def timed(f):
                f()
                ttnn.synchronize_device(mesh)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                for _ in range(50):
                    ttnn.deallocate(f())
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.synchronize_device(mesh)
                ts = []
                for _ in range(6):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                    ts.append(time.perf_counter() - t0)
                ttnn.release_trace(mesh, tid)
                return round(sorted(ts)[2] / 50 * 1e6, 1)

            res["plain_us_per_norm"] = timed(
                lambda: ttnn.rms_norm(xt, epsilon=EPS, weight=wt, compute_kernel_config=_k_norm())
            )
            res["sharded_us_per_norm_incl_2_moves"] = timed(lambda: rms_decode_sharded(xt, wt))
            print("NORM_RESULT " + json.dumps(res), flush=True)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
