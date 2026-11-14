"""device-sampling: N-layer model tests of the device-resident decode loop (4x8 mesh, real weights layers 0..NL-1, real prefill, pool decoder dense mode).
Stages (env STAGES, comma list): host (host-loop reference tokens), greedy (T1: device greedy == host argmax), admit (T4: finish + re-admission), sample (T3 distribution, S and F), same (T2 slot independence), perf.
Env: NL (3), STEPS (100), NSAMP (3000)."""
import json
import os
import statistics
import time

import torch

os.environ["HY4_HEAD_FP32"] = "1"
import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt import hy4_pool_decode as _pd
from models.demos.blackhole.hy4_preview.tt.hy4_ccl_safe import hy4_fabric_selftest
from models.demos.blackhole.hy4_preview.tt.hy4_chain import CHUNK, Hy4Chain
from models.demos.blackhole.hy4_preview.tt.hy4_device_loop import DeviceLoop
from models.demos.blackhole.hy4_preview.tt.hy4_pool_model import Hy4PoolDecoder
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

CKPT, OUT = os.environ.get("HY4_CKPT", "/ckpt"), os.environ.get("OUT", "/out")
NL, STEPS, NSAMP = (
    int(os.environ.get("NL", "3")),
    int(os.environ.get("STEPS", "100")),
    int(os.environ.get("NSAMP", "3000")),
)
STAGES = os.environ.get("STAGES", "host,greedy,admit,sample,same,perf").split(",")
PAD = 120002
FIRST = [1000, 2000, 3000, 4000]


def res(**kw):
    print("RES " + json.dumps(kw, default=str), flush=True)


def prefill(ch, dec, prompts):
    lens = [len(p) for p in prompts]
    ids = torch.full((4, CHUNK), PAD, dtype=torch.long)
    for r in range(4):
        ids[r, : lens[r]] = torch.tensor(prompts[r])
    ch.reset()
    x4, _ = ch.prefill(ids)
    ttnn.synchronize_device(ch.md)
    ttnn.deallocate(x4)
    dec.handoff(CHUNK)
    ch.reset()
    return lens


MARG = []


def gof_z(ex, counts, n, min_exp=10.0):
    """chi-square goodness of fit of the draw counts against the exact law ex, bins pooled (descending probability) until the expected count >= min_exp; returns (z, dof) with z = (chi2 - dof) / sqrt(2 dof): ~N(0,1) for an exact sampler."""
    order = ex.argsort(descending=True)
    p, c = ex[order].double(), counts[order].double()
    bins_p, bins_c, ap, ac = [], [], 0.0, 0.0
    for pi, ci in zip(p.tolist(), c.tolist()):
        ap += pi
        ac += ci
        if ap * n >= min_exp:
            bins_p.append(ap)
            bins_c.append(ac)
            ap = ac = 0.0
        if pi * n < 1e-3 and ap == 0.0:
            break
    ap = max(0.0, 1.0 - sum(bins_p))
    ac = n - sum(bins_c)
    if ap * n >= min_exp:
        bins_p.append(ap)
        bins_c.append(ac)
    elif bins_p:
        bins_p[-1] += ap
        bins_c[-1] += ac
    bp, bc = torch.tensor(bins_p), torch.tensor(bins_c)
    chi2 = float(((bc - bp * n) ** 2 / (bp * n)).sum())
    dof = len(bins_p) - 1
    if dof < 1:
        return None, dof
    return round((chi2 - dof) / (2 * dof) ** 0.5, 2), dof


MARG = []


def host_loop(dec, lens, steps, first=FIRST, sched=None):
    toks, pos = list(first), list(lens)
    out = []
    for s in range(steps):
        dec.set_inputs(toks, pos)
        o = dec.run()
        am, lgs = dec.read_argmax(o)
        tv_ = lgs.topk(2, dim=-1).values
        MARG.append((tv_[:, 0] - tv_[:, 1]).tolist())
        out.append(am)
        toks, pos = am, [p + 1 for p in pos]
    return out


def main():
    prompts = torch.load(f"{OUT}/decode_prompts.pt")
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "700e6"))))
    try:
        links = hy4_fabric_selftest(md)
        _pd._POOL_AG_LINKS = links.pool_ag
        res(event="fabric_selftest", num_links=links.num_links, pool_ag=links.pool_ag)
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=links.num_links, topology=ttnn.Topology.Ring)
        ch = Hy4Chain(md, mc, ccl, ckpt=CKPT)
        ch.add_layers(0, NL, log=lambda s: None)
        ch.build_head()
        dec = Hy4PoolDecoder(ch, 8, mode="dense")
        lens = prefill(ch, dec, prompts)
        res(event="built", nl=NL, lens=lens)
        ref = None
        if "host" in STAGES or "greedy" in STAGES or "admit" in STAGES or "perf" in STAGES:
            dec.set_inputs(FIRST, lens)
            dec.capture()
            ref = host_loop(dec, lens, STEPS)
            marg = [list(m) for m in MARG]
            from models.demos.blackhole.hy4_preview.tt import hy4_pool_model as _pm

            _orig_rt = _pm.rope_tables

            def rope64(positions, dim=64, theta=1e7, interleave=True):
                pos = torch.as_tensor(positions, dtype=torch.float64)
                inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
                ang = torch.outer(pos, inv)
                c, s_ = ang.cos().float(), ang.sin().float()
                if interleave:
                    return c.repeat_interleave(2, dim=-1), s_.repeat_interleave(2, dim=-1)
                return torch.cat([c, c], -1), torch.cat([s_, s_], -1)

            _pm.rope_tables = rope64
            MARG.clear()
            ref64 = host_loop(dec, lens, STEPS)
            _pm.rope_tables = _orig_rt
            dd = [i for i, (a, b) in enumerate(zip(ref, ref64)) if a != b]
            res(event="host_fp32rope_vs_host_fp64rope", steps=STEPS, differing_steps=len(dd), first=dd[:1])
            res(event="host_ref", steps=STEPS, first_rows=[r[:6] for r in zip(*ref)])
            if "perf" in STAGES:
                pos = list(lens)
                tl, ts_, tr_, tk_ = [], [], [], []
                toks = list(FIRST)
                for s in range(14):
                    t0 = time.perf_counter()
                    dec.set_inputs(toks, pos)
                    t1 = time.perf_counter()
                    o = dec.run()
                    t2 = time.perf_counter()
                    toks, _ = dec.read_argmax(o)
                    t3 = time.perf_counter()
                    pos = [p + 1 for p in pos]
                    ts_.append(t1 - t0), tr_.append(t2 - t1), tk_.append(t3 - t2), tl.append(t3 - t0)
                m = lambda v: round(statistics.median(v[3:]) * 1e3, 2)
                res(event="host_loop_ms", total=m(tl), set_inputs=m(ts_), run=m(tr_), readback_argmax=m(tk_))
            dec.release()
        dl = DeviceLoop(dec)
        variants = ["G"]
        if {"sample", "same"} & set(STAGES):
            variants += os.environ.get("VARS", "S,Fp,F").split(",")
        t0 = time.time()
        dl.capture(variants)
        res(event="captured_fused", variants=variants, sec=round(time.time() - t0, 1))
        if "sample" in STAGES:
            dl.capture_split([v for v in variants])
            res(event="captured_split", parity=dl.parity)
        dl.set_sampling([0.0] * 4, [0] * 4, [1.0] * 4, [1, 2, 3, 4])

        def run_dev(steps, variant="G", tokens=FIRST, pos=None, on_step=None):
            dl.load_inputs(tokens, pos if pos is not None else lens)
            hs = []
            t0 = time.perf_counter()
            for s in range(steps):
                if on_step:
                    on_step(s)
                dl.submit(variant)
                hs.append(dl.read_async())
            host_s = time.perf_counter() - t0
            ttnn.synchronize_device(md)
            return [dl.tokens(h) for h in hs], host_s, time.perf_counter() - t0

        if "bisect" in STAGES:
            dl.load_inputs(FIRST, lens)
            ttnn.synchronize_device(md)
            res(event="bisect", at="inputs loaded")
            ttnn.execute_trace(md, dl.trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            res(event="bisect", at="step trace replayed")
            ttnn.execute_trace(md, dl.stid["G"], cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            res(event="bisect", at="G trace replayed", tokens=dl.tokens(dl.read_async()))
            ttnn.execute_trace(md, dl.trace, cq_id=0, blocking=False)
            ttnn.execute_trace(md, dl.stid["G"], cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            res(event="bisect", at="step+G back to back", tokens=dl.tokens(dl.read_async()))
        if "dbg" in STAGES:
            dl.load_inputs(FIRST, lens)
            ttnn.execute_trace(md, dl.trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            lg = (
                ttnn.to_torch(dl.out, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=(4, 8)))
                .reshape(4, -1)
                .float()
            )
            top = lg.topk(2, dim=-1)
            res(
                event="dbg_step0",
                logits_dtype=str(dl.out.dtype),
                argmax=lg.argmax(-1).tolist(),
                top2=top.values.tolist(),
                host_ref0=ref[0],
            )
            o = dl.greedy_v2(dl.out)
            ttnn.synchronize_device(md)
            res(
                event="dbg_eager_v2",
                tokens=[int(ttnn.to_torch(ttnn.get_device_tensors(o)[r * 8]).reshape(-1)[0]) for r in range(4)],
            )
            rm = ttnn.to_layout(dl.out, ttnn.ROW_MAJOR_LAYOUT)
            ix = ttnn.argmax(rm, dim=-1, keepdim=True)
            res(
                event="dbg_local_argmax",
                idx=[int(ttnn.to_torch(ttnn.get_device_tensors(ix)[r * 8 + 7]).reshape(-1)[0]) for r in range(4)],
                shape=list(dl.out.shape),
                layout=str(dl.out.layout),
                mem=str(dl.out.memory_config()),
            )
            lg_ = dl.out
            rm_ = ttnn.to_layout(lg_, ttnn.ROW_MAJOR_LAYOUT)
            idx_ = ttnn.argmax(rm_, dim=-1, keepdim=True)
            mx_ = ttnn.gather(rm_, -1, index=idx_)
            idxf_ = ttnn.typecast(idx_, ttnn.float32)
            pair_ = ttnn.concat([mx_, idxf_], dim=-1)
            g_ = ch.mc.allgather(pair_, ch.ccl, axis=1, dim=2)
            ttnn.synchronize_device(md)
            f = lambda t, r, c: ttnn.to_torch(ttnn.get_device_tensors(t)[r * 8 + c]).float().reshape(-1).tolist()
            res(
                event="dbg_inner",
                idx=str(idx_.dtype),
                idx_vals=f(idx_, 0, 7),
                mx=f(mx_, 0, 7),
                idxf=f(idxf_, 0, 7),
                pair=f(pair_, 0, 7),
                gathered_row0_dev0=f(g_, 0, 0),
                gathered_row0_dev7=f(g_, 0, 7),
                shape=list(g_.shape),
            )
            t_ = ttnn.transpose(ttnn.to_layout(g_, ttnn.TILE_LAYOUT), 2, 3)
            mxa = ttnn.slice(t_, [0, 0, 0, 0], [1, 1, 1, 8])
            ixa = ttnn.slice(t_, [0, 0, 1, 0], [1, 1, 2, 8])
            be = ttnn.argmax(ttnn.to_layout(mxa, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True)
            bf = ttnn.typecast(ttnn.to_layout(be, ttnn.TILE_LAYOUT), ttnn.float32)
            cid = dl._col_ids()
            eqq = ttnn.eq(bf, cid)
            mul = ttnn.multiply(eqq, ixa)
            sm = ttnn.sum(mul, dim=-1, keepdim=True)
            ttnn.synchronize_device(md)
            res(
                event="dbg_post",
                t=f(t_, 0, 0)[:40],
                mxa=f(mxa, 0, 0),
                ixa=f(ixa, 0, 0),
                be=f(be, 0, 0),
                bf=f(bf, 0, 0),
                cid=f(cid, 0, 0),
                eq=f(eqq, 0, 0),
                mul=f(mul, 0, 0),
                sum=f(sm, 0, 0),
                tshape=list(t_.shape),
            )
            ttnn.execute_trace(md, dl.stid["G"], cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            res(
                event="dbg_G",
                tokens=dl.tokens(dl.read_async()),
                pos_f=[float(ttnn.to_torch(ttnn.get_device_tensors(dl.pos_f)[r * 8]).reshape(-1)[0]) for r in range(4)],
                lens=lens,
            )
        if "greedy" in STAGES:
            got, _, _ = run_dev(STEPS)
            got2, _, _ = run_dev(STEPS)
            got3, _, _ = run_dev(STEPS)
            res(event="T1_pipelined_repeatable", run1_eq_run2=got == got2, run1_eq_run3=got == got3)
            got_sync = []
            dl.load_inputs(FIRST, lens)
            for st in range(STEPS):
                dl.submit("G")
                got_sync.append(dl.tokens(dl.read_async()))
            res(
                event="T1_sync_vs_pipelined",
                sync_equals_pipelined=got_sync == got,
                sync_vs_host_mismatching_steps=sum(1 for a, b in zip(got_sync, ref) if a != b),
                first_diff_sync=next((i for i, (a, b) in enumerate(zip(got_sync, ref)) if a != b), None),
            )
            dd = [
                (i, r, got[i][r], ref[i][r]) for i in range(min(STEPS, 14)) for r in range(4) if got[i][r] != ref[i][r]
            ]
            res(
                event="T1_detail",
                first_pipelined_mismatches=dd[:6],
                prev_tokens_if_stale=[(i, r, got[i - 1][r]) for i, r, a, b in dd[:3] if i > 0],
            )
            for nm, rr in (("host_fp32rope", ref), ("host_fp64rope", ref64)):
                bad = sum(1 for a, b in zip(got, rr) if a != b)
                res(
                    event="T1_greedy_equal_host",
                    vs=nm,
                    steps=STEPS,
                    rows=4,
                    mismatching_steps=bad,
                    first_diff=next((i for i, (a, b) in enumerate(zip(got, rr)) if a != b), None),
                )
        if "teacher" in STAGES:
            pos = list(lens)
            bad = []
            for st in range(STEPS):
                tin = list(FIRST) if st == 0 else ref[st - 1]
                dl.load_inputs(tin, [p + st for p in lens])
                dl.submit("G")
                tk = dl.tokens(dl.read_async())
                for r in range(4):
                    if tk[r] != ref[st][r]:
                        bad.append((st, r, round(marg[st][r], 4)))
            res(
                event="T1b_teacher_forced",
                steps=STEPS,
                rows=4,
                disagreements=len(bad),
                of=STEPS * 4,
                step_row_hostmargin=bad,
            )
        if "admit" in STAGES:
            # finish row 2 at step 20 (inactive), re-admit it at step 30 (full reload, row 2 restarted from its prompt), compare against host streams
            dl.load_inputs(FIRST, lens)
            toks_hist, hs = [], []
            cur_tok, cur_pos = list(FIRST), list(lens)
            for s in range(50):
                if s == 20:
                    dl.load_inputs(cur_tok, cur_pos, active=[1, 1, 0, 1])
                if s == 30:
                    cur_tok[2], cur_pos[2] = FIRST[2], lens[2]
                    dl.load_inputs(cur_tok, cur_pos, active=[1, 1, 1, 1])
                dl.submit("G")
                tk = dl.tokens(dl.read_async())
                toks_hist.append(tk)
                cur_tok = tk
                cur_pos = [p + (0 if (r == 2 and 20 <= s < 30) else 1) for r, p in enumerate(cur_pos)]
            ok_rows = {
                r: all(toks_hist[s][r] == ref[s][r] for s in range(50) if not (r == 2 and 20 <= s < 50))
                for r in (0, 1, 3)
            }
            row2_first = all(toks_hist[s][2] == ref[s][2] for s in range(20))
            row2_readmit = all(toks_hist[30 + i][2] == ref[i][2] for i in range(20))
            res(
                event="T4_finish_admit",
                other_rows_equal_host=ok_rows,
                row2_before_finish_equal=row2_first,
                row2_after_readmit_equals_fresh_stream=row2_readmit,
            )
        if "sample" in STAGES:
            logits = None
            ttnn.execute_trace(md, dl.trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            lg = (
                ttnn.to_torch(dl.out, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=(4, 8)))
                .reshape(4, -1)
                .float()
            )
            for name, (T, k, p, var) in {
                "S_T0.9_k20": (0.9, 20, 1.0, "S"),
                "S_T1.0_k5": (1.0, 5, 1.0, "S"),
                "Fp_T0.9_default": (0.9, 0, 1.0, "Fp"),
                "F_T0.9_k50_p0.9": (0.9, 50, 0.9, "F"),
                "F_T0.9_p0.95": (0.9, 0, 0.95, "F"),
            }.items():
                if var not in dl.stid:
                    continue
                v = dl.set_sampling([T] * 4, [k] * 4, [p] * 4, [7, 7, 7, 7])
                assert v == var, (v, var)
                # replay the sampler alone NSAMP times: pos advances every replay (draw = f(seed, pos)); the logits stay frozen
                dl.load_inputs([5] * 4, [100] * 4)
                NSAMP_ = int(os.environ.get("NS_" + var, NSAMP))
                got = []
                t0 = time.perf_counter()
                res(event="sample_loop_start", var=var, case=name)
                for i in range(NSAMP_):
                    ttnn.execute_trace(md, dl.stid[var], cq_id=0, blocking=False)
                    got.append(dl.read_async())
                    if os.environ.get("SYNC_EVERY") and i % int(os.environ["SYNC_EVERY"]) == 0:
                        ttnn.synchronize_device(md)
                        res(event="progress", i=i)
                ttnn.synchronize_device(md)
                dt = (time.perf_counter() - t0) / NSAMP_ * 1e3
                toks = torch.tensor([dl.tokens(h) for h in got])  # [N,4]
                from models.demos.blackhole.hy4_preview.tests.samp_cpu_tv import exact_p, shared_p, tv

                for r in range(4):
                    ex = exact_p(lg[r], T, k, p)
                    cnt = torch.bincount(toks[:, r], minlength=ex.numel()).double()
                    emp = cnt / NSAMP_
                    g = torch.Generator().manual_seed(9 + r)
                    ref_s = torch.multinomial(ex, NSAMP_, replacement=True, generator=g)
                    ref_cnt = torch.bincount(ref_s, minlength=ex.numel()).double()
                    noise = tv(ex, ref_cnt / NSAMP_)
                    z_dev, dof = gof_z(ex, cnt, NSAMP_)
                    z_ref, _ = gof_z(ex, ref_cnt, NSAMP_)
                    tv_law_vs_exact = round(tv(ex, shared_p(lg[r], T, k, p)), 4) if var == "S" else None
                    ent = float(-(ex * (ex + 1e-30).log()).sum())
                    sh = tv(shared_p(lg[r], T, k, p), emp) if var == "S" else None
                    res(
                        event="T3_dist",
                        variant=var,
                        case=name,
                        row=r,
                        n=NSAMP_,
                        gof_z_device=z_dev,
                        gof_z_exact_multinomial=z_ref,
                        gof_dof=dof,
                        tv_emulated_shared_law_vs_exact=tv_law_vs_exact,
                        entropy=round(ent, 3),
                        tv_vs_exact=round(tv(ex, emp), 4),
                        noise_tv=round(noise, 4),
                        tv_vs_shared_law=sh,
                        distinct=int(len(set(toks[:, r].tolist()))),
                        ms_per_replay_incl_read=round(dt, 3),
                    )
        if "same" in STAGES:
            # identical prompts in the 4 rows, same seed everywhere: sampled tokens must be equal across rows (slot independence of the whole chain incl. the sampler); different seeds differ; repeat is exact
            dl.release()
            lens2 = prefill(ch, dec, [prompts[0]] * 4)
            dl.capture(variants)
            for var, (T, k, p) in {"S": (0.9, 20, 1.0), "Fp": (0.9, 0, 1.0)}.items():
                if var not in dl.ftid:
                    continue
                for seeds, label in (([11, 11, 11, 11], "same_seed"), ([11, 12, 13, 14], "diff_seeds")):
                    dl.set_sampling([T] * 4, [k] * 4, [p] * 4, seeds)
                    a, _, _ = run_dev(24, var, tokens=[7] * 4, pos=lens2)
                    b, _, _ = run_dev(24, var, tokens=[7] * 4, pos=lens2)
                    eq_rows = sum(1 for s in a if len(set(s)) == 1)
                    res(
                        event="T2_slot_independence",
                        variant=var,
                        seeds=label,
                        steps_with_all_rows_equal=f"{eq_rows}/24",
                        repeat_identical=a == b,
                        stream_row0=[s[0] for s in a[:8]],
                    )
        if "perf" in STAGES:
            for var in [v for v in variants if v in dl.ftid]:
                dl.set_sampling([0.0 if var == "G" else 0.9] * 4, [0] * 4, [1.0] * 4, [1, 2, 3, 4])
                dl.load_inputs(FIRST, lens)
                n = 30
                ttnn.synchronize_device(md)
                # (a) pipelined: no sync between steps, 4 ids read per step (host does enqueue only)
                t0 = time.perf_counter()
                hs = []
                for _ in range(n):
                    dl.submit(var)
                    hs.append(dl.read_async())
                tq = time.perf_counter() - t0
                ttnn.synchronize_device(md)
                tp = time.perf_counter() - t0
                # (b) synchronous per step (submit + wait for the ids)
                ds = []
                for _ in range(10):
                    t1 = time.perf_counter()
                    dl.submit(var)
                    dl.tokens(dl.read_async())
                    ds.append(time.perf_counter() - t1)

                # (c) trace-only
                def tr(tid, n=20):
                    ttnn.synchronize_device(md)
                    t1 = time.perf_counter()
                    for _ in range(n):
                        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(md)
                    return (time.perf_counter() - t1) / n * 1e3

                res(
                    event="device_loop_ms",
                    variant=var,
                    pipelined_ms_per_step=round(tp / n * 1e3, 2),
                    host_enqueue_ms_per_step=round(tq / n * 1e3, 3),
                    sync_step_ms=round(statistics.median(ds[2:]) * 1e3, 2),
                    fused_trace_ms=round(tr(dl.ftid[var]), 2),
                )
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
