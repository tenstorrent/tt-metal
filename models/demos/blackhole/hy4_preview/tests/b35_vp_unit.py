# SPDX-License-Identifier: Apache-2.0
"""B35 unit: value gather as push + flag-wait assemble (hy4_pool.push_wait) vs the host model of all_gather_async(dim=3): every device holds different bf16 rows; several launches in a row, bit exact."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt import hy4_qpush

S = int(os.environ.get("S", "4"))
KL = int(os.environ.get("KLOC", "512"))
REPS = int(os.environ.get("REPS", "6"))
R, C = 4, 8


def main():
    md = open_mesh(trace=int(100e6))
    try:
        g = torch.Generator().manual_seed(3)
        pb = KL * 2
        push = hy4_qpush.Push(md, S, pb, max(1, min(4352 // pb, 4)))
        ok_all = True
        for it in range(REPS):
            v = torch.randn(R, C, S, KL, generator=g).to(torch.bfloat16)
            vh = v.permute(0, 2, 1, 3).reshape(R * S, 1, 1, C * KL)
            vt = ttnn.from_torch(
                vh,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=md,
                mesh_mapper=ttnn.ShardTensor2dMesh(md, mesh_shape=(R, C), dims=(0, 3)),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            push(vt)
            av = HP.push_wait(push, S)
            ttnn.synchronize_device(md)
            dts = ttnn.get_device_tensors(av)
            eq = True
            for r in range(R):
                exp = v[r].permute(1, 0, 2).reshape(S, C * KL)  # [S, 8*KL]
                for c in range(C):
                    got = ttnn.to_torch(dts[r * C + c]).reshape(S, C * KL)
                    if not torch.equal(got.view(torch.int16), exp.view(torch.int16)):
                        eq = False
            ok_all &= eq
            print("RES " + json.dumps(dict(event="case", it=it, equal=eq)), flush=True)
        print("RES " + json.dumps(dict(event="done", ok=ok_all)), flush=True)
    finally:
        close_mesh(md)


main()
