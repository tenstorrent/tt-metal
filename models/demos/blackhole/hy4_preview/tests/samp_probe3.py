"""device-sampling probe 3: greedy sampler variants eager vs traced on the 4x8 mesh (fp32 logits [1,1,1,15104] per device). VARIANT env: v2 (RM dim-2 all-gather + transpose, no tiny TILE reshapes)."""
import json
import types

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

NC, ROWS, SHARD = 8, 4, 15104


def res(**kw):
    print("RES " + json.dumps(kw, default=str), flush=True)


def main():
    md = open_mesh(trace=50_000_000)
    try:
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        ch = types.SimpleNamespace(mc=mc, ccl=ccl, md=md)
        from models.demos.blackhole.hy4_preview.tt.hy4_device_loop import DeviceLoop

        shell = types.SimpleNamespace(md=md, ch=ch)
        # DeviceLoop needs a decoder; use only its greedy(): build a bare object
        dl = DeviceLoop.__new__(DeviceLoop)
        dl.md, dl.ch = md, ch
        g = torch.Generator().manual_seed(4)
        x = torch.randn(ROWS, 1, 1, NC * SHARD, generator=g) * 3
        x[1, 0, 0, 5] = x[1, 0, 0, 20000] = 50.0  # tie across columns (lowest index must win)
        x[2, 0, 0, 100] = x[2, 0, 0, 101] = 60.0  # tie inside a column
        x[3, 0, 0, NC * SHARD - 1] = 70.0
        lt = ttnn.from_torch(
            x,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 3), mesh_shape=(ROWS, NC)),
        )
        want = x.reshape(ROWS, -1).argmax(-1).tolist()

        def read(t):
            return [int(ttnn.to_torch(ttnn.get_device_tensors(t)[r * NC]).reshape(-1)[0]) for r in range(ROWS)]

        fn = dl.greedy_v2
        o = fn(lt)
        ttnn.synchronize_device(md)
        res(event="eager", got=read(o), want=want)
        o = fn(lt)
        ttnn.synchronize_device(md)
        res(event="eager2", got=read(o))
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        o = fn(lt)
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        res(event="captured")
        for i in range(4):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            res(event="replay", i=i, got=read(o), want=want)
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
