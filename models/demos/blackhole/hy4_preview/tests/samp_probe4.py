"""device-sampling probe 4: where does the full-vocabulary sampler hang in a TRACE on the 4x8 mesh? STOP env = Hy4DeviceSampler stage name (greedy_part, scale_exp, search_k, search_p, prefix, '' = whole); PLAIN=1 plain path.
Synthetic fp32 logits, capture, replay x3 with a sync each (progress lines identify the stage that hangs)."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt.hy4_sampler import Hy4DeviceSampler
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

NC, ROWS, SHARD = 8, 4, 15104


def res(**kw):
    print("RES " + json.dumps(kw, default=str), flush=True)


def main():
    stop = os.environ.get("STOP", "") or None
    plain = os.environ.get("PLAIN", "1") == "1"
    md = open_mesh(trace=50_000_000)
    try:
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        fs = Hy4DeviceSampler(md, mc, ccl, users_per_row=1)
        fs.alloc_params()
        fs.set_params([1 / 0.9] * 4, [0] * 4, [1.0] * 4, [0.37, 0.5, 0.9, 0.1], [0.0] * 4)
        g = torch.Generator().manual_seed(4)
        x = torch.randn(ROWS, 1, 1, NC * SHARD, generator=g) * 3
        lt = ttnn.from_torch(
            x,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 3), mesh_shape=(ROWS, NC)),
        )
        fs.stop = stop
        res(event="start", stop=stop, plain=plain)
        o = fs.forward(lt, fs.params, plain=plain)
        ttnn.synchronize_device(md)
        res(event="eager_ok", shape=list(o.shape))
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        i0 = ccl.ag_ping_pong_idx
        o = fs.forward(lt, fs.params, plain=plain)
        res(event="ag_calls_parity", parity=(ccl.ag_ping_pong_idx - i0) % 2)
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        res(event="captured")
        if os.environ.get("BTB", "0") == "1":
            res(event="btb_start", ag_idx=ccl.ag_ping_pong_idx)
            for i in range(60):
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            res(event="btb_done")
            return
        for i in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            res(event="replay", i=i)
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
