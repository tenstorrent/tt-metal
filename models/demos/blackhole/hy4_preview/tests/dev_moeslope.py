# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""moe-perf: moe_compute time vs the number of distinct active experts on the BUSIEST chip (forced routing, no router). n experts on each of 32/n chips (4 tokens x 8 = 32 activations).
Gives the fixed cost + per-expert slope of dispatch+moe_compute. Env: HY4_T=1, HY4_NLIST=1,2,3,4,8 + lever env (MOE_COMPUTE_*)."""
import json
import os
import statistics
import time

import torch

import ttnn
from models.common.moe.tt_moe_decode import moe_compute_kwargs
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh, p
from models.demos.blackhole.hy4_preview.tt.hy4_moe import Hy4MoEBlock
from models.demos.blackhole.hy4_preview.tt.hy4_moe_place import experts_of_chip

T = int(os.environ.get("HY4_T", "1"))
NLIST = [int(v) for v in os.environ.get("HY4_NLIST", "1,2,3,4,8").split(",")]
OUT = os.environ.get("OUT", "/tmp/moeslope.json")
HID = 6144
REP, NREP = 10, 15


def routing_for(n, B):
    """token t gets 8 experts; the active set is n experts on each of 32/n chips (rows x cols); B tokens x 8 acts = 8B activations in total, each distinct per token (n=1 needs 8B <= 32 chips)."""
    nchip = max(1, (8 * B) // n)
    chips = [(r, c) for c in range(8) for r in range(4)][:nchip]
    acts = []
    for r, c in chips:
        for e in experts_of_chip(r, c)[:n]:
            acts.append(e)
    acts = acts[: 8 * B]
    ids = torch.tensor(acts, dtype=torch.int64).reshape(B, 8)
    return ids


def trace_us(md, fn, rep=REP, n=NREP):
    def body():
        for _ in range(rep):
            fn()

    body()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    body()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / rep * 1e6)
    ttnn.release_trace(md, tid)
    return statistics.median(ts)


md = open_mesh()
rows, cols = tuple(md.shape)
res = {
    "T": T,
    "env": {k: v for k, v in os.environ.items() if k.startswith(("MOE_", "DSV41_", "HY4_", "MP_"))},
    "pts": [],
}
try:
    from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer

    W = load_moe_layer(1)
    blk = Hy4MoEBlock(md, W, batch_per_device=T)
    blk.warmup()
    dec, cfg = blk.decode, blk.decode_config
    B = T * rows
    x = torch.randn(B, HID).to(torch.bfloat16)
    tt_tok = ttnn.from_torch(
        x.reshape(B, 1, 1, HID),
        device=md,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
    )
    for n in NLIST:
        ids = routing_for(n, B)
        w = torch.full((B, 8), 1.0 / 8)
        tt_i = ttnn.from_torch(
            ids.reshape(B, 1, 1, 8).to(torch.int32),
            device=md,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
        )
        tt_s = ttnn.from_torch(
            w.reshape(B, 1, 1, 8).to(torch.bfloat16),
            device=md,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
        )

        def disp():
            (x_t, d_x), (i_t, d_i), (s_t, d_s) = dec._format_dispatch_inputs(tt_tok, tt_i, tt_s)
            sp, oi, osc = ttnn.experimental.all_to_all_dispatch_metadata(
                x_t,
                i_t,
                s_t,
                dec.expert_state.tt_expert_mapping,
                **cfg.dispatch.model_dump(),
                output_tensors=dec.buffers.tt_dispatch_output_tensors,
                cross_device_semaphore=dec.buffers.dispatch_global_semaphore,
            )
            for t, d in ((x_t, d_x), (i_t, d_i), (s_t, d_s)):
                if d:
                    ttnn.deallocate(t)
            return sp, oi, osc

        def disp_comp():
            sp, oi, osc = disp()
            _, _, _, l1_out, _, comb = ttnn.experimental.moe_compute(
                sp,
                oi,
                osc,
                dec.expert_state.tt_expert_mapping,
                dec.expert_state.tt_w0_w1,
                dec.expert_state.tt_w2,
                layer_id=0,
                **moe_compute_kwargs(cfg),
                optional_output_tensor=dec.buffers.tt_combine_output,
                optional_cross_device_semaphore=dec.buffers.combine_global_semaphore,
            )
            ttnn.deallocate(l1_out)

        a = trace_us(md, disp)
        b = trace_us(md, disp_comp)
        res["pts"].append({"n": n, "dispatch_us": a, "disp_compute_us": b, "compute_us": b - a})
        p(f"T={T} busiest-chip experts n={n}: dispatch(+fmt) {a:.1f} us, compute {b-a:.1f} us (total {b:.1f})")
        ttnn.deallocate(tt_i)
        ttnn.deallocate(tt_s)
finally:
    close_mesh(md)
    json.dump(res, open(OUT, "w"), indent=1)
