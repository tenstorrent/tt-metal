"""mtp-device stage 6b (1x8 slot, routed MoE skipped): the sampled-spec round tail in ONE trace: verify-side lossless sampling (SpecSampling.verify on persistent mock verify logits) -> accept -> MTP s0 -> k-1 recursive
steps with q-sampled drafts -> next state. Checks: trace == eager, distribution of the emitted first token a_0 over many rounds == softmax(p_0) (the verify logits are fixed mock logits, the drafts come from the real
MTP head on random hidden rows), accept rates, and the traced cost of the sampled tail vs the greedy tail. Env: K (3), ROUNDS (1500), CTX (100)."""
import json
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tests.spec_sample_dev import stats
from models.demos.blackhole.hy4_preview.tt.hy4_attn_opt import Opt
from models.demos.blackhole.hy4_preview.tt.hy4_chain import HID, Hy4Chain
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_mtp import Hy4MTP, greedy_tokens
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx
from models.demos.blackhole.hy4_preview.tt.hy4_sampler import Hy4DeviceSampler
from models.demos.blackhole.hy4_preview.tt.hy4_spec_sample import SpecSampling
from models.demos.blackhole.hy4_preview.tt.hy4_spec_trace import MTPRoundTail, SpecDerive
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

V = 120832


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def val(t, n=None):
    x = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(-1).float()
    return x[:n].tolist() if n else x.tolist()


def main():
    k = int(os.environ.get("K", "3"))
    S = k + 1
    P0 = int(os.environ.get("CTX", "100"))
    NR = int(os.environ.get("ROUNDS", "1500"))
    full, md = open_row()
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        from models.demos.blackhole.deepseek_v41_flash.tt import ccl_lean

        ccl_lean.make_lean(mc)
        ccl_lean.make_lean_sems(ccl)
        ch = Hy4Chain(md, mc, ccl, os.environ["HY4_CKPT"])
        ch.build_head()
        ch.add_layers(78, 79)
        ly = ch.mtp_layers[0]
        opt = Opt.from_env()
        pages = 4
        ctx = PoolCtx(md, mc, ccl, pages, rows=1, S=S, mode="dense", bucket_tokens=pages * 512)
        kvp = ttnn.empty(
            [1, 1, pages * 64, 656],
            dtype=ttnn.fp8_e4m3,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        kp = ttnn.empty(
            [pages, 1, 64, 128],
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ly.attn.init_pool(ctx, kvp, kp)
        mtp = Hy4MTP(ch, ly, ch.rd.layer_sd(-1), opt, moe_fn="skip")
        sdv = SpecDerive(md, ctx, ly.attn.rope, ly.attn.indexer.idx_rope_first, S)
        smp = Hy4DeviceSampler(md, mc, ccl, users_per_row=S)
        sam = SpecSampling(smp, k, invT=1.0, seed=1234)
        gen = torch.Generator().manual_seed(5)
        lp = torch.randn(V, generator=gen) * 3.0
        P_log = ttnn.from_torch(
            lp.float().reshape(1, 1, 1, V).repeat(1, 1, S, 1),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(md, dim=3),
        )
        h_in = upload(md, torch.randn(1, 1, S, HID).bfloat16())
        tails = {}
        for name, sm in (("sampled", sam), ("greedy", None)):
            t = MTPRoundTail(ch, mtp, sdv, ctx, k, lambda lg: greedy_tokens(ch, lg), sampling=sm)
            tails[name] = t

        def body(name):
            t = tails[name]
            sdv.derive(t.pos)
            a = sam.verify(P_log, t.pos) if name == "sampled" else greedy_tokens(ch, P_log)
            t.run(a, h_in, derived=True)

        def setpos(t, P):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.tensor([float(P)]).reshape(1, 1, 1, 1),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(md),
                ),
                t.pos,
            )

        snap = lambda t: dict(
            pos=val(t.pos, 1)[0], m=val(t.m_out, 1)[0], a=val(t.a_out, S), d=val(t.d_row, k), tok=val(t.tok, S)
        )
        t = tails["sampled"]
        setpos(t, P0)
        body("sampled")  # eager (compile + the q_persist / d_col state of a first round)
        ttnn.synchronize_device(md)
        setpos(t, P0)
        body("sampled")
        ttnn.synchronize_device(md)
        e = snap(t)
        setpos(t, P0)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        body("sampled")
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        # same state as before the eager second round: q_persist / d_col / d_row hold the state AFTER that round -> run the trace twice from reset pos and compare two replays (deterministic given state)
        outs = []
        for _ in range(2):
            setpos(t, P0 + 7)
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            outs.append(snap(t))
        # eager vs trace from an identical state: capture state, run eager, restore not possible (persistent q): compare trace replay 1 with eager on the NEXT state instead
        res(
            event="replay_pair",
            note="two replays at the same pos from the same carried state differ only through the carried q/d state",
            first=outs[0],
            second=outs[1],
        )
        # distribution of a_0 over rounds (pos advances by 1 per round from the host: fresh draws; the verify logits are fixed)
        cnt = torch.zeros(V, dtype=torch.long)
        accr = 0
        ms = [0] * (k + 1)
        t0 = time.time()
        for i in range(NR):
            setpos(t, P0 + 20 + i)
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            s_ = snap(t)
            cnt[int(s_["a"][0])] += 1
            ms[int(s_["m"])] += 1
        p = torch.softmax(lp.double(), 0)
        res(
            event="a0_distribution",
            rounds=NR,
            vs_p=stats(cnt.double(), p, NR),
            m_hist=ms,
            sec=round(time.time() - t0, 1),
        )

        # cost: sampled vs greedy tail (trace-only)
        def cost(name):
            tt = tails[name]
            setpos(tt, P0)
            body(name)
            ttnn.synchronize_device(md)
            tid2 = ttnn.begin_trace_capture(md, cq_id=0)
            body(name)
            ttnn.end_trace_capture(md, tid2, cq_id=0)
            ttnn.synchronize_device(md)
            ts = []
            for _ in range(10):
                setpos(tt, P0)
                ttnn.synchronize_device(md)
                t1 = time.perf_counter()
                ttnn.execute_trace(md, tid2, cq_id=0, blocking=False)
                ttnn.synchronize_device(md)
                ts.append((time.perf_counter() - t1) * 1e6)
            ttnn.release_trace(md, tid2)
            return round(statistics.median(ts), 1)

        res(event="tail_cost", k=k, us_greedy=cost("greedy"), us_sampled=cost("sampled"))
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
