# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""moe-perf: per-op table of the Hy4 MoE decode layer (routed + shared fold) by cumulative trace diff, on REAL layer-1 activations (real routing).

The layer is a list of stages (shared partial, router sub-steps, dispatch, moe_compute, tail, fold add, reduce_scatter); trace of stages[:k] x REP replayed N times
(median); per-op cost = difference of consecutive prefixes. Also the full-layer number (all stages) and the per-chip skew is not visible (wall clock of the mesh trace).
Env: HY4_T=1|2|4 (tokens per device; 1 = B=4), HY4_LAYER=1, MP_REP=10, MP_N=15, plus the lever env of tt/hy4_moe_fast.py when present."""
import json
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, close_mesh, open_mesh, p
from models.demos.blackhole.hy4_preview.tt import hy4_dense as D
from models.demos.blackhole.hy4_preview.tt.hy4_moe import Hy4MoEBlock
from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer
from models.demos.blackhole.hy4_preview.tt.hy4_router import EPS

LAYER = int(os.environ.get("HY4_LAYER", "1"))
T = int(os.environ.get("HY4_T", "1"))
REP = int(os.environ.get("MP_REP", "10"))
NREP = int(os.environ.get("MP_N", "15"))
HID = 6144
OUT = os.environ.get("OUT", "/tmp/moeperf.json")


def stages_for(blk, shared, tt_gate, tt_tok):
    """list of (name, fn(st)) ; st = dict of live tensors; each fn adds its outputs to st."""
    dec, cfg, gate = blk.decode, blk.decode_config, blk.gate
    from models.common.moe.tt_moe_decode import moe_compute_kwargs
    from models.demos.blackhole.deepseek_v41_flash.tt.moe_tail import make_col_tensor, moe_tail
    from models.demos.blackhole.hy4_preview.tt.hy4_router_select import router_select

    L1 = ttnn.L1_MEMORY_CONFIG
    if not hasattr(dec, "_col"):
        dec._col = make_col_tensor(dec._mesh)

    def s_shared(st):
        st["part"] = shared.forward(tt_gate)

    def s_logits(st):
        st["logits"] = ttnn.matmul(
            tt_gate,
            gate._w,
            compute_kernel_config=gate.ckc,
            dtype=ttnn.float32,
            core_grid=ttnn.CoreGrid(y=1, x=8),
            memory_config=L1,
        )

    def s_sigmoid(st):
        st["score"] = ttnn.sigmoid(st.pop("logits"), memory_config=L1)

    def s_rank(st):
        st["rank"] = ttnn.add(st["score"], gate._bias, memory_config=L1)

    def s_select(st):
        st["w"], st["idx"] = router_select(st["rank"], st["score"], gate.k, EPS, gate.scale)
        ttnn.deallocate(st.pop("rank"))
        ttnn.deallocate(st.pop("score"))

    def s_fmt(st):
        (x_t, d_x), (i_t, d_i), (s_t, d_s) = dec._format_dispatch_inputs(tt_tok, st["idx"], st["w"])
        st["fmt"] = (x_t, d_x, i_t, d_i, s_t, d_s)

    def s_dispatch(st):
        x_t, d_x, i_t, d_i, s_t, d_s = st["fmt"]
        st["sparse"], st["o_idx"], st["o_sc"] = ttnn.experimental.all_to_all_dispatch_metadata(
            x_t,
            i_t,
            s_t,
            dec.expert_state.tt_expert_mapping,
            **cfg.dispatch.model_dump(),
            output_tensors=dec.buffers.tt_dispatch_output_tensors,
            cross_device_semaphore=dec.buffers.dispatch_global_semaphore,
        )
        if d_x:
            ttnn.deallocate(x_t)
        if d_s:
            ttnn.deallocate(s_t)

    def s_compute(st):
        _, _, _, l1_out, _, st["combine"] = ttnn.experimental.moe_compute(
            st["sparse"],
            st["o_idx"],
            st["o_sc"],
            dec.expert_state.tt_expert_mapping,
            dec.expert_state.tt_w0_w1,
            dec.expert_state.tt_w2,
            layer_id=0,
            **moe_compute_kwargs(cfg),
            optional_output_tensor=dec.buffers.tt_combine_output,
            optional_cross_device_semaphore=dec.buffers.combine_global_semaphore,
        )
        ttnn.deallocate(l1_out)

    def s_tail(st):
        st["red"] = moe_tail(st["combine"], st["w"], st["idx"], dec.expert_state.tt_expert_mapping, dec._col)

    def s_add(st):
        r2 = ttnn.add(st["red"], st["part"], dtype=ttnn.bfloat16)
        ttnn.deallocate(st["red"])
        st["red"] = r2

    def s_rs(st):
        st["out"] = ttnn.reduce_scatter(st["red"], **cfg.reduce_scatter.model_dump())

    return [
        ("shared expert partial", s_shared),
        ("router matmul fp32", s_logits),
        ("router sigmoid", s_sigmoid),
        ("router rank add", s_rank),
        ("router_select (top8+norm)", s_select),
        ("dispatch input format", s_fmt),
        ("all_to_all_dispatch_metadata", s_dispatch),
        ("moe_compute", s_compute),
        ("moe_tail (tilize+reduce)", s_tail),
        ("fold add (shared)", s_add),
        ("reduce_scatter (cols)", s_rs),
    ]


protect = set()


def free_all(st):
    seen = set()

    def rec(v):
        if isinstance(v, ttnn.Tensor):
            if id(v) not in seen and id(v) not in protect:
                seen.add(id(v))
                try:
                    ttnn.deallocate(v)
                except Exception:
                    pass
        elif isinstance(v, (tuple, list)):
            for e in v:
                rec(e)

    # dispatch buffers / combine buffers are persistent module buffers: never freed
    protect.update(id(t) for t in blk_persistent)
    for k, v in list(st.items()):
        if k in ("sparse", "o_idx", "o_sc", "combine"):
            continue
        rec(v)
    st.clear()


def trace_us(md, stages, k, rep=REP, n=NREP):
    def body():
        for _ in range(rep):
            st = {}
            for name, fn in stages[:k]:
                fn(st)
            free_all(st)

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
    return statistics.median(ts), min(ts)


acts = torch.load(ACTS + "acts_l0_l1.pt")
X = torch.cat([h.reshape(-1, HID) for h, y, r in acts["moe"]])
md = open_mesh()
rows, cols = tuple(md.shape)
res = {
    "T": T,
    "layer": LAYER,
    "env": {k: v for k, v in os.environ.items() if k.startswith(("MOE_", "DSV41_", "HY4_", "MP_"))},
}
try:
    t0 = time.time()
    W = load_moe_layer(LAYER)
    p(f"load_moe_layer({LAYER}) {time.time()-t0:.1f}s warm_cache={W['w0'] is None}")
    shared_w = [W[k][256] for k in ("shared_w0", "shared_w1", "shared_w2")]
    blk = Hy4MoEBlock(md, W, batch_per_device=T)
    blk.warmup()
    shared = D.make_shared(md, *shared_w, decode=True)
    B = T * rows
    idx = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(100 + T))[:B]
    x = X[idx].to(torch.bfloat16)
    tt_tok = ttnn.from_torch(
        x.reshape(B, 1, 1, HID),
        device=md,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
    )
    tt_gate = ttnn.from_torch(
        x.reshape(1, 1, B, HID),
        device=md,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols)),
    )
    dec = blk.decode
    blk_persistent = list(dec.buffers.tt_dispatch_output_tensors) + [dec.buffers.tt_combine_output]
    protect.update([id(tt_gate), id(tt_tok)])
    stages = stages_for(blk, shared, tt_gate, tt_tok)
    prev, table = 0.0, []
    for k in range(0, len(stages) + 1):
        if k == 0:

            def empty():
                return None

            # baseline of an empty trace (launch overhead): skip, treat as 0
            continue
        med, mn = trace_us(md, stages, k)
        table.append({"op": stages[k - 1][0], "cum_us": med, "cum_min_us": mn, "step_us": med - prev})
        p(f"T={T} prefix {k:2d} {stages[k-1][0]:34s} cum {med:8.1f} us (min {mn:8.1f})  step {med-prev:7.1f}")
        prev = med
    res["table"] = table
    # production flow end-to-end through the block (router inside + fold), for cross-check
    shared_fwd = lambda: blk.forward(tt_gate, tt_tok, fold=shared.forward(tt_gate))
    from models.demos.blackhole.hy4_preview.tests.dev_common import traced_us

    res["block_forward_us"] = traced_us(md, shared_fwd)
    p(f"T={T} blk.forward(fold) traced {res['block_forward_us']:.1f} us ; stage-sum total {prev:.1f} us")
finally:
    close_mesh(md)
    json.dump(res, open(OUT, "w"), indent=1)
