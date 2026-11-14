"""mtp-device S3b: the draft side of a round in ONE trace on a 1x8 slot (routed MoE skipped: moe_fn='skip'; the dataflow, not the numbers): accept -> MTP s0 over S = k+1 rows -> one-hot row m -> k-1 recursive
MTP steps with device-side token / hidden / position feedback -> next-round inputs. Checks: (1) trace replay == eager run (pos, m, a_out, drafts, next tokens), (2) pos advances by m + 1 for forced m = 0..k,
(3) next verify tokens = [a_m, d_1..d_k], (4) 3 consecutive replays with new inputs. Reports the traced cost of the tail (MTP passes + heads + samplers + accept). Env: K (3), CTX (position P0), MODE dense|topk."""
import json
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tests.mtp_bench_dev import synth_fill
from models.demos.blackhole.hy4_preview.tt.hy4_attn_opt import Opt
from models.demos.blackhole.hy4_preview.tt.hy4_chain import HID, Hy4Chain
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_mtp import Hy4MTP, greedy_tokens
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx
from models.demos.blackhole.hy4_preview.tt.hy4_spec_trace import MTPRoundTail, SpecDerive
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def val(t, n=None):
    x = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(-1).float()
    return x[:n].tolist() if n else x.tolist()


def main():
    k = int(os.environ.get("K", "3"))
    S = k + 1
    P0 = int(os.environ.get("CTX", "100"))
    mode = os.environ.get("MODE", "dense")
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
        pages = P0 // 512 + 2
        bucket = min(((P0 + 16) // 256 + 1) * 256, pages * 512)
        ctx = PoolCtx(md, mc, ccl, pages, rows=1, S=S, mode=mode, bucket_tokens=bucket)
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
        gen = torch.Generator().manual_seed(1)
        synth_fill(md, ctx, kvp, kp, (P0 // 512) * 512, gen)
        mtp = Hy4MTP(ch, ly, ch.rd.layer_sd(-1), opt, moe_fn="skip")
        sdv = SpecDerive(md, ctx, ly.attn.rope, ly.attn.indexer.idx_rope_first, S)
        tail = MTPRoundTail(ch, mtp, sdv, ctx, k, lambda lg: greedy_tokens(ch, lg))
        a_tok = upload(md, torch.zeros(S, 1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        h_in = upload(md, torch.randn(1, 1, S, HID).bfloat16())

        def set_state(pos, drafts):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.tensor([float(pos)]).reshape(1, 1, 1, 1),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(md),
                ),
                tail.pos,
            )
            d = torch.full((1, 1, 1, 16), -1.0)
            d[0, 0, 0, :k] = torch.tensor(drafts, dtype=torch.float32)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    d, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(md)
                ),
                tail.d_row,
            )

        def set_a(vals):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.tensor(vals, dtype=torch.int32).reshape(S, 1),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(md),
                ),
                a_tok,
            )

        def snap():
            return dict(
                pos=val(tail.pos, 1)[0],
                m=val(tail.m_out, 1)[0],
                a=val(tail.a_out, S),
                d=val(tail.d_row, k),
                tok=val(tail.tok, S),
            )

        drafts0 = [11, 22, 33][:k]
        # eager
        for m_force in range(k + 1):
            av = [drafts0[j] if j < m_force else 7000 + j for j in range(k)] + [900]
            set_state(P0, drafts0)
            set_a(av)
            tail.run(a_tok, h_in)
            ttnn.synchronize_device(md)
            e = snap()
            ok = e["m"] == m_force and e["pos"] == P0 + m_force + 1 and e["tok"][0] == av[m_force]
            res(event="eager", m_force=m_force, snap=e, ok=ok)
        # trace
        set_state(P0, drafts0)
        set_a(drafts0 + [900])
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        tail.run(a_tok, h_in)
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        for m_force in range(k + 1):
            av = [drafts0[j] if j < m_force else 7000 + j for j in range(k)] + [900]
            set_state(P0, drafts0)
            set_a(av)
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            t_ = snap()
            ok = t_["m"] == m_force and t_["pos"] == P0 + m_force + 1 and t_["tok"][0] == av[m_force]
            # eager reference of the same inputs
            set_state(P0, drafts0)
            set_a(av)
            tail.run(a_tok, h_in)
            ttnn.synchronize_device(md)
            e = snap()
            res(event="trace_vs_eager", m_force=m_force, trace=t_, eager=e, same=(t_ == e), ok=ok)
        # chained rounds: replay 3x without resetting the state (pos / drafts / tok carry over on device); the test only forces the accept vector from the carried drafts
        set_state(P0, drafts0)
        pos_prev = P0
        for r in range(3):
            d_now = val(tail.d_row, k)
            av = [int(d_now[j]) if j < (r % (k + 1)) else 8000 + j for j in range(k)] + [901 + r]
            set_a(av)
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            s = snap()
            res(
                event="chain_round",
                r=r,
                forced_m=r % (k + 1),
                snap=s,
                ok=(s["m"] == r % (k + 1) and s["pos"] == pos_prev + s["m"] + 1),
            )
            pos_prev = s["pos"]
        # timing of the tail
        ts = []
        for _ in range(10):
            set_state(P0, drafts0)
            set_a(drafts0 + [900])
            ttnn.synchronize_device(md)
            t0 = time.perf_counter()
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            ts.append((time.perf_counter() - t0) * 1e6)
        res(
            event="tail_cost",
            k=k,
            S=S,
            ctx=P0,
            mode=mode,
            us_median=round(statistics.median(ts), 1),
            us_min=round(min(ts), 1),
            note="MTP s0 + (k-1) recursive steps at S rows + k lm_head + greedy + accept, WITHOUT routed experts",
        )
        ttnn.release_trace(md, tid)
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
