# SPDX-License-Identifier: Apache-2.0
"""Prototype of the on-device token feedback: global argmax over the vocab-sharded logits (8 column shards) entirely on device, in-place update of the token / position tensors, and the same
inside a trace (replayed). Reference: host argmax over the concatenated shards (lowest vocab id on ties)."""
import json
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import close_mesh, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.tt.decode import global_argmax  # noqa: E402


def main():
    mesh = open_mesh_full()
    try:
        g = torch.Generator().manual_seed(0)
        V = 19072
        for U in (1, 8, 32):
            lg = (torch.randn(4, 8, 1, U, V, generator=g) * 2).bfloat16()
            lg[0, 3, 0, 0, 5] = 50.0  # clear winner in column 3
            if U > 1:
                lg[1, 2, 0, 1, 7] = 40.0
                lg[1, 6, 0, 1, 9] = 40.0  # tie across columns -> lowest column (2)
            xt = ttnn.from_torch(
                lg.reshape(4, 8, U, V),
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, 1)),
            )
            xt = ttnn.reshape(xt, (1, 1, U, V))
            tok = ttnn.from_torch(
                torch.zeros(4, U, dtype=torch.int32),
                device=mesh,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, None)),
            )
            fn = lambda: global_argmax(xt, tok, V)
            fn()
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            dts = ttnn.get_device_tensors(tok)
            got = torch.stack([ttnn.to_torch(dts[r * 8]).long().reshape(U) for r in range(4)])
            # reference
            full = lg.reshape(4, 8, U, V).permute(0, 2, 1, 3).reshape(4, U, 8 * V).float()
            ref = full.argmax(-1)
            print(
                "ARGMAX_RESULT",
                json.dumps(
                    dict(
                        U=U,
                        equal=bool(torch.equal(got, ref)),
                        mismatches=int((got != ref).sum()),
                        got0=got[0, :3].tolist(),
                        ref0=ref[0, :3].tolist(),
                    )
                ),
                flush=True,
            )
            ttnn.release_trace(mesh, tid)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
