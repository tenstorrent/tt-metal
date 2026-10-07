# SPDX-License-Identifier: Apache-2.0
"""Stress the traced decode MoE of ONE real layer (DEC_LAYER=5): fixed inputs (distinct random bf16 hidden per mesh row, 8 live slots per row), the same trace replayed R times; every replay must
be bit-identical to the first. Stages (MiMoMoEDecode.forward upto): 0 router+spread, 1 counts, 5 MoE before the column all-reduce (per-column partial sums), 99 full layer MoE.
Reports per stage the replays that differ, the mesh rows / columns affected."""
import json
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import close_mesh, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.reference.loader import Ckpt  # noqa: E402
from models.demos.blackhole.mimo_v26.tt import weights as W  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.decode import LiveMask  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.layer import MiMoDecoderLayer  # noqa: E402

E = os.environ.get
OUT = E("DEC_OUT", "/tmp/moe_stress.jsonl")


def main():
    L, R = int(E("DEC_LAYER", "5")), int(E("DEC_REPS", "400"))
    mesh = open_mesh_full()
    try:
        lyr = MiMoDecoderLayer(mesh, L, Ckpt(), W.WeightCache(mesh), 256, ring_blocks=12)
        g = torch.Generator().manual_seed(1)
        x = (torch.randn(4, 1, 32, 4096, generator=g) * 0.5).bfloat16()
        xd = ttnn.from_torch(
            x,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, None)),
        )
        live = LiveMask(mesh)
        alive = torch.zeros(4, 32, dtype=torch.bool)
        alive[:, :8] = True
        keep = live.set(alive)
        ttnn.synchronize_device(mesh)
        lyr.moe_decode(xd, live.t).deallocate(True)  # builds MiMoMoEDecode, compiles
        dec = lyr.moe_dec
        comp = ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape))
        for stage in [int(v) for v in E("DEC_STAGES", "99,5").split(",")]:

            def fwd():
                o = dec.forward(xd, live.t, upto=stage)
                return o if isinstance(o, tuple) else (o,)

            os.environ["MIMO_CHECK_CAP"] = "0"
            [t.deallocate(True) for t in fwd()]
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            outs = fwd()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ref = [ttnn.to_torch(o, mesh_composer=comp) for o in outs]
            bad = []
            t0 = time.time()
            for r in range(R):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                for i, o in enumerate(outs):
                    y = ttnn.to_torch(o, mesh_composer=comp)
                    live_rows = y[:, :, :, :8] if y.dim() == 4 and y.shape[2] >= 8 else y
                    d = (y != ref[i]).flatten(2).any(-1) if y.dim() >= 3 else (y != ref[i])
                    if d.any():
                        bad.append((r, i, [(int(a), int(b)) for a, b in d.nonzero()][:6]))
            rec = dict(
                case="moe_decode_stress",
                layer=L,
                stage=stage,
                reps=R,
                secs=round(time.time() - t0, 1),
                n_bad_replays=len({b[0] for b in bad}),
                first_bad=bad[:4],
            )
            print("MS_RESULT " + json.dumps(rec, default=str), flush=True)
            open(OUT, "a").write(json.dumps(rec, default=str) + "\n")
            ttnn.release_trace(mesh, tid)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
