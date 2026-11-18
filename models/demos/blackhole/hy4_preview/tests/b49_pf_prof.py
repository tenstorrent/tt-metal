"""B49 prefill chunk profile on a 1x8 slot (MESH_DEVICE=1x8): Hy4Chain layers 0..NL-1 (L0 dense+full idx, L1 MoE+full idx, L2 MoE+shared idx), eager chain prefill of K chunks of 512 tokens (one user),
(A) wall of the last chunk with no syncs inside (host dispatch + device) and with a sync only at the end, (B) per-section times with a device sync after every section (device-bound view), per layer type.
Env: NL, K (chunks, default 6), REPS, ROPE etc. Prints RES lines."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt import hy4_chain as HC
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

CKPT = os.environ.get("HY4_CKPT", "/ckpt")
NL = int(os.environ.get("NL", "3"))
K = int(os.environ.get("K", "6"))
REPS = int(os.environ.get("REPS", "3"))


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    mesh4 = os.environ.get("MESH4") == "1"  # whole 4x8 mesh (4 users, TP32 MoE prefill on the real weights)
    if mesh4:
        from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
        from models.demos.gpt_oss.config import mesh_4x8

        full, md = None, open_mesh(trace=500_000_000)
    else:
        full, md = open_row()
    try:
        mc, ccl = (mesh_4x8() if mesh4 else mesh_1x8()), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        ch = HC.Hy4Chain(md, mc, ccl, ckpt=CKPT, kv_fmt=os.environ.get("HY4_KV_FMT", "bf16"))
        t0 = time.time()
        ch.add_layers(0, NL, log=lambda s: print(s, flush=True))
        res(event="built", sec=round(time.time() - t0, 1))
        rnd = torch.Generator().manual_seed(3)
        ids = [torch.randint(0, 100000, (1, HC.CHUNK), generator=rnd) for _ in range(K)]
        if mesh4:
            ch.build_head()
            ids4 = [torch.randint(1000, 100000, (4, HC.CHUNK), generator=rnd) for _ in range(K)]
            if (
                os.environ.get("REAL") == "1"
            ):  # real text tokens (4 different slices of a long text) instead of random ids
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
                txt = (
                    open("/home/container_app_user/tt-metal/README.md").read()
                    + open("/home/container_app_user/tt-metal/METALIUM_GUIDE.md").read()
                )
                allids = tok(txt * 4, add_special_tokens=False)["input_ids"]
                need = 4 * K * HC.CHUNK
                assert len(allids) >= need, (len(allids), need)
                for r_ in range(4):
                    seg = torch.tensor(allids[r_ * 3000 : r_ * 3000 + K * HC.CHUNK]).reshape(K, HC.CHUNK)
                    for c_ in range(K):
                        ids4[c_][r_] = seg[c_]
            xs_dev = [ch.embed_ids(ids4[c]) for c in range(K)]
            xs = None
        else:
            emb = ch.rd.get(
                "model.embed_tokens.weight"
            )  # [V, D]: realistic stream inputs (the embedding rows, 4 identical streams like Hy4Chain.embed_ids)
            xs = [emb[ids[c][0]].float().reshape(1, 1, HC.CHUNK, 6144).repeat(1, 4, 1, 1) for c in range(K)]
            del emb
        mk_x = (lambda c: ttnn.clone(xs_dev[c])) if mesh4 else (lambda c: upload(md, xs[c], ttnn.float32))
        sync = lambda: ttnn.synchronize_device(md)

        # prefill K-1 chunks to build context (also warms the program cache), then time the last one K times (re-running the same start: append=True grows, so use fresh ch.reset each rep)
        def run_chunks(n_chunks, timed_last=False, secs=None):
            for ly in ch.layers:
                ly.attn.reset()
            ibl = {}
            tl = None
            for c in range(n_chunks):
                last = c == n_chunks - 1
                if last and timed_last:
                    sync()
                    tl = time.perf_counter()
                x4 = mk_x(c)
                for ly in ch.layers:
                    x4 = ly.prefill_chunk(x4, c * HC.CHUNK, ibl)
                if last and timed_last:
                    t_disp = time.perf_counter() - tl
                    sync()
                    tl = (t_disp, time.perf_counter() - tl)
                else:
                    sync() if c == n_chunks - 1 else None
                ttnn.deallocate(x4)
            for t in ibl.values():
                ttnn.deallocate(t)
            return tl

        if (
            os.environ.get("CMP") == "1"
        ):  # fused iHC (HY4_B49_IHC=fused at build) vs the stock path (pf instances switched off), same layers / weights / inputs

            def last_outs(fused):
                outs = []
                for ly in ch.layers:
                    ly.attn.reset()
                saved = [(ly.ihcpf_a, ly.ihcpf_m) for ly in ch.layers]
                os.environ["HY4_B49_COLLAPSE"] = "1" if fused else os.environ.get("CMP_STOCK_COLLAPSE", "0")
                if not fused:
                    for ly in ch.layers:
                        ly.ihcpf_a = ly.ihcpf_m = None
                ibl = {}
                for c in range(K):
                    x4 = mk_x(c)
                    for ly in ch.layers:
                        x4 = ly.prefill_chunk(x4, c * HC.CHUNK, ibl)
                        if c == 0 and os.environ.get("DBGNAN") == "1":
                            xt = ttnn.to_torch(ttnn.get_device_tensors(x4)[0]).float()
                            res(
                                event="dbg_layer",
                                fused=fused,
                                layer=ly.L,
                                finite=bool(torch.isfinite(xt).all()),
                                absmax=float(xt[torch.isfinite(xt)].abs().max()),
                            )
                    sync()
                    outs.append(ttnn.to_torch(ttnn.get_device_tensors(x4)[0]).float().reshape(4, HC.CHUNK, 6144))
                    ttnn.deallocate(x4)
                for t in ibl.values():
                    ttnn.deallocate(t)
                for ly, (a_, m_) in zip(ch.layers, saved):
                    ly.ihcpf_a, ly.ihcpf_m = a_, m_
                return outs

            def pcc(a, b):
                a, b = a.flatten().double(), b.flatten().double()
                a, b = a - a.mean(), b - b.mean()
                return float((a * b).sum() / (a.norm() * b.norm()))

            o_s0 = last_outs(False)  # cold pass (compiles)
            o_s = last_outs(False)
            o_s2 = last_outs(False)
            o_f = last_outs(True)
            o_f2 = last_outs(True)
            for c in range(K):
                fin = lambda t: bool(torch.isfinite(t).all())
                res(
                    event="cmp_chunk",
                    c=c,
                    finite=[fin(o_s0[c]), fin(o_s[c]), fin(o_s2[c]), fin(o_f[c]), fin(o_f2[c])],
                    pcc_fused_vs_stock=round(pcc(o_f[c], o_s[c]), 8),
                    pcc_stock_repeat=round(pcc(o_s2[c], o_s[c]), 8),
                    maxabs=float((o_f[c] - o_s[c]).abs().max()),
                    refabsmax=float(o_s[c].abs().max()),
                    stock_repeat_equal=bool(torch.equal(o_s[c], o_s2[c])),
                    fused_repeat_equal=bool(torch.equal(o_f[c], o_f2[c])),
                )
        if (
            os.environ.get("AB") == "1"
        ):  # eager chunk wall: stock iHC / collapse-only variant / fused iHC, same layers (HY4_B49_IHC=fused at build)
            saved = [(ly.ihcpf_a, ly.ihcpf_m) for ly in ch.layers]
            for name, pf_on, coll, ropec in (
                ("stock", False, "0", "0"),
                ("fused", True, "1", "0"),
                ("fused+ropecache", True, "1", "1"),
            ):
                os.environ["HY4_B49_COLLAPSE"] = coll
                os.environ["HY4_B49_ROPE_CACHE"] = ropec
                for ly, (a_, m_) in zip(ch.layers, saved):
                    ly.ihcpf_a, ly.ihcpf_m = (a_, m_) if pf_on else (None, None)
                run_chunks(K)
                for r in range(REPS):
                    d, w = run_chunks(K, True)
                    res(
                        event="ab_chunk",
                        variant=name,
                        rep=r,
                        ctx=K * HC.CHUNK,
                        layers=NL,
                        host_dispatch_ms=round(d * 1e3, 1),
                        wall_ms=round(w * 1e3, 1),
                        per_layer_ms=round(w * 1e3 / NL, 2),
                    )
            for ly, (a_, m_) in zip(ch.layers, saved):
                ly.ihcpf_a, ly.ihcpf_m = a_, m_
            os.environ["HY4_B49_COLLAPSE"] = "1"
        if os.environ.get("CPROF") == "1":  # cProfile of the last chunk (host python + pybind time by function)
            import cProfile
            import io
            import pstats

            run_chunks(K)
            pr = cProfile.Profile()
            for ly in ch.layers:
                ly.attn.reset()
            ibl = {}
            for c in range(K - 1):
                x4 = mk_x(c)
                for ly in ch.layers:
                    x4 = ly.prefill_chunk(x4, c * HC.CHUNK, ibl)
                ttnn.deallocate(x4)
            sync()
            x4 = mk_x(K - 1)
            pr.enable()
            for ly in ch.layers:
                x4 = ly.prefill_chunk(x4, (K - 1) * HC.CHUNK, ibl)
            pr.disable()
            sync()
            sio = io.StringIO()
            pstats.Stats(pr, stream=sio).sort_stats("tottime").print_stats(28)
            for line in sio.getvalue().splitlines():
                print("CPROF " + line[:200], flush=True)
        if (
            os.environ.get("OPLOG") == "1"
        ):  # ordered ttnn call log of one layer-chunk of each type with the calling model line
            import sys
            import types

            logs = {}
            cur = {"layer": None}
            wrapped = []

            def wrapl(owner, nm, obj):
                def w(*a, **k):
                    f = sys._getframe(1)
                    where = f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}"
                    t0 = time.perf_counter()
                    try:
                        return obj(*a, **k)
                    finally:
                        logs.setdefault(cur["layer"], []).append((nm, where, round((time.perf_counter() - t0) * 1e6)))

                setattr(owner, nm, w)
                wrapped.append((owner, nm, obj))

            for owner in (ttnn, getattr(ttnn, "experimental", None), getattr(ttnn, "transformer", None)):
                if owner is None:
                    continue
                for nm in dir(owner):
                    if nm.startswith("_") or nm in (
                        "synchronize_device",
                        "get_device_tensors",
                        "from_torch",
                        "to_torch",
                    ):
                        continue
                    try:
                        obj = getattr(owner, nm)
                    except Exception:
                        continue
                    if isinstance(obj, (type, types.ModuleType)) or not callable(obj):
                        continue
                    wrapl(owner, nm, obj)
            orig = HC.Hy4ChainLayer.prefill_chunk

            def counted(self, x4, start, ids_by_layer, taps=None):
                cur["layer"] = f"L{self.L}"
                logs[cur["layer"]] = []
                return orig(self, x4, start, ids_by_layer, taps)

            HC.Hy4ChainLayer.prefill_chunk = counted
            run_chunks(K)
            sync()
            for lay in ("L1", "L2"):
                for i, (nm, wh, us) in enumerate(logs.get(lay, [])):
                    print(f"OPLOG {lay} {i:3d} {nm:34s} {wh:34s} {us}", flush=True)
            for owner, nm, obj in wrapped:
                setattr(owner, nm, obj)
            HC.Hy4ChainLayer.prefill_chunk = orig
        if (
            os.environ.get("MOECMP") == "1"
        ):  # MoE block with / without the fused rs-exact tail (HY4_B49_RSTAIL), all 32 devices
            ly = ch.layers[1]
            blkp = ly.blk_pf
            assert blkp is not None and blkp.rowbase is not None, "needs HY4_B49_RSTAIL=1 and HY4_TP32PF_RS_EXACT=1"
            x4 = mk_x(0)
            pre, post = ly.ihc_m.pre_post(x4)
            raw = HC.Hy4IHCGen.collapse(x4, pre)
            h = ly._norm(raw, ly.w_post)
            outs = {}
            for mode in ("stock", "fused", "stock", "fused"):
                saved = blkp.rowbase
                if mode == "stock":
                    blkp.rowbase = None
                y = blkp.forward(h)
                blkp.rowbase = saved
                ttnn.synchronize_device(md)
                outs.setdefault(mode, []).append([ttnn.to_torch(t) for t in ttnn.get_device_tensors(y)])
                ttnn.deallocate(y)

            def cmp(a, b):
                ne = tot = mx = 0
                for ta, tb in zip(a, b):
                    d = (ta.view(torch.int16).int() - tb.view(torch.int16).int()).abs()
                    ne += int((d != 0).sum())
                    tot += d.numel()
                    mx = max(mx, int(d.max()))
                return ne, tot, mx

            res(
                event="moecmp",
                stock_repeat=cmp(outs["stock"][0], outs["stock"][1]),
                fused_repeat=cmp(outs["fused"][0], outs["fused"][1]),
                fused_vs_stock=cmp(outs["fused"][0], outs["stock"][0]),
            )
        if os.environ.get(
            "ENVAB"
        ):  # ENVAB='[{"name":"a","env":{"K":"V"}}, ...]': eager chunk wall per env variant (flags are read at call time), 2 passes, REPS reps each
            variants = json.loads(os.environ["ENVAB"])
            keys = sorted({k for v in variants for k in v["env"]})
            base = {k: os.environ.get(k) for k in keys}
            for rnd_ in range(2):
                for v in variants:
                    for k in keys:
                        if k in v["env"]:
                            os.environ[k] = v["env"][k]
                        elif base[k] is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = base[k]
                    run_chunks(K)
                    ws = []
                    for r in range(REPS):
                        d, w = run_chunks(K, True)
                        ws.append(round(w * 1e3 / NL, 2))
                    res(event="envab", round=rnd_, variant=v["name"], per_layer_ms=ws)
            for k in keys:
                if base[k] is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = base[k]
        if (
            os.environ.get("OPCOUNT") == "1"
        ):  # host op counts / host time per ttnn callable for one layer of each type (fused path as configured)
            import types

            counts = {}
            cur = {"layer": None}
            skip = {
                "deallocate",
                "synchronize_device",
                "from_torch",
                "to_torch",
                "get_device_tensors",
                "allocate_tensor_on_device",
            }
            wrapped = []

            def wrap(owner, nm, obj):
                def w(*a, **k):
                    t0 = time.perf_counter()
                    try:
                        return obj(*a, **k)
                    finally:
                        e = counts.setdefault(cur["layer"], {}).setdefault(nm, [0, 0.0])
                        e[0] += 1
                        e[1] += time.perf_counter() - t0

                w.__wrapped__ = obj
                setattr(owner, nm, w)
                wrapped.append((owner, nm, obj))

            for owner in (ttnn, getattr(ttnn, "experimental", None), getattr(ttnn, "transformer", None)):
                if owner is None:
                    continue
                for nm in dir(owner):
                    if nm.startswith("_") or nm in skip:
                        continue
                    try:
                        obj = getattr(owner, nm)
                    except Exception:
                        continue
                    if isinstance(obj, (type, types.ModuleType)) or not callable(obj):
                        continue
                    if owner is not ttnn:
                        nm_full = nm
                    wrap(owner, nm, obj)
            orig = HC.Hy4ChainLayer.prefill_chunk

            def counted(self, x4, start, ids_by_layer, taps=None):
                cur["layer"] = f"L{self.L}"
                return orig(self, x4, start, ids_by_layer, taps)

            HC.Hy4ChainLayer.prefill_chunk = counted
            run_chunks(K)
            counts.clear()
            run_chunks(K)
            sync()
            for lay, d in counts.items():
                tot = sum(v[0] for v in d.values())
                tms = sum(v[1] for v in d.values()) * 1e3
                top = sorted(d.items(), key=lambda kv: -kv[1][1])[:14]
                res(
                    event="opcount",
                    layer=lay,
                    total_calls=tot,
                    total_host_ms=round(tms, 2),
                    top=[(k, v[0], round(v[1] * 1e3, 2)) for k, v in top],
                )
            for owner, nm, obj in wrapped:
                setattr(owner, nm, obj)
            HC.Hy4ChainLayer.prefill_chunk = orig
        run_chunks(K)  # warm
        from models.demos.blackhole.hy4_preview.tt import hy4_ihc_pf as _pfm

        _pfm.HOST_T.clear()
        for r in range(REPS):
            d, w = run_chunks(K, True)
            res(
                event="ihcpf_host_ms_per_call",
                t={
                    k: [round(v[1] / max(v[0], 1) * 1e3, 3), round(v[2] / max(v[0], 1) * 1e3, 3)]
                    for k, v in _pfm.HOST_T.items()
                },
                calls={k: v[0] for k, v in _pfm.HOST_T.items()},
            )
            _pfm.HOST_T.clear()
            res(
                event="chunk_eager",
                rep=r,
                ctx=K * HC.CHUNK,
                layers=NL,
                host_dispatch_ms=round(d * 1e3, 1),
                wall_ms=round(w * 1e3, 1),
                per_layer_ms=round(w * 1e3 / NL, 2),
            )
        # (B) per-section device-synced times
        acc, disp = {}, {}

        def T(name, fn):
            sync()
            t = time.perf_counter()
            out = fn()
            td = time.perf_counter()
            sync()
            acc.setdefault(name, []).append((time.perf_counter() - t) * 1e3)
            disp.setdefault(name, []).append((td - t) * 1e3)
            return out

        L = HC.Hy4ChainLayer

        def timed_prefill_chunk(self, x4, start, ids_by_layer, taps=None):
            tag = f"L{self.L}"
            if self.ihcpf_a is not None:  # fused iHC variant (same flow as _prefill_chunk_pf)
                bf = lambda t: t if t.dtype == ttnn.bfloat16 else ttnn.typecast(t, ttnn.bfloat16)
                h, st = T(tag + ".ihcpf_a.pre", lambda: self.ihcpf_a.pre(x4))
                src = self.spec.indexer_source
                ids_in = None if self.spec.indexer == HC.FULL else ids_by_layer[src]
                a, ids = T(tag + ".attn", lambda: self.attn.prefill_chunk(h, start, ids_in=ids_in))
                if self.spec.indexer == HC.FULL:
                    ids_by_layer[self.L] = ids
                ab = T(tag + ".bf16_a", lambda: bf(a))
                x4a = T(tag + ".ihcpf_a.expand", lambda: self.ihcpf_a.expand(x4, ab, st))
                h, st = T(tag + ".ihcpf_m.pre", lambda: self.ihcpf_m.pre(x4a))
                y = T(tag + ".mlp", lambda: self._mlp_prefill(h))
                yb = T(tag + ".bf16_y", lambda: bf(y))
                return T(tag + ".ihcpf_m.expand", lambda: self.ihcpf_m.expand(x4a, yb, st))
            tag = f"L{self.L}"
            pre, post = T(tag + ".ihc_a.pre_post", lambda: self.ihc_a.pre_post(x4))
            raw = T(tag + ".ihc_a.collapse", lambda: self.ihc_a.collapse(x4, pre))
            h = T(tag + ".norm_a", lambda: self._norm(raw, self.w_in))
            src = self.spec.indexer_source
            ids_in = None if self.spec.indexer == HC.FULL else ids_by_layer[src]
            a, ids = T(tag + ".attn", lambda: self.attn.prefill_chunk(h, start, ids_in=ids_in))
            if self.spec.indexer == HC.FULL:
                ids_by_layer[self.L] = ids
            x4 = T(tag + ".ihc_a.expand", lambda: self.ihc_a.expand(x4, a, post))
            pre, post = T(tag + ".ihc_m.pre_post", lambda: self.ihc_m.pre_post(x4))
            raw = T(tag + ".ihc_m.collapse", lambda: self.ihc_m.collapse(x4, pre))
            h = T(tag + ".norm_m", lambda: self._norm(raw, self.w_post))
            y = T(tag + ".mlp", lambda: self._mlp_prefill(h))
            x4 = T(tag + ".ihc_m.expand", lambda: self.ihc_m.expand(x4, y, post))
            return x4

        L.prefill_chunk = timed_prefill_chunk
        if mesh4:  # MoE block sub-stages
            from models.demos.blackhole.hy4_preview.tt import hy4_tp32_prefill as TP

            for cls, nm in ((TP.Tp32PlanBuilder, "plan_build"), (TP.Tp32PrefillExperts, "experts")):
                f0 = cls.__call__
                cls.__call__ = (lambda f0, nm: (lambda self, *x, **k: T(f"moe.{nm}", lambda: f0(self, *x, **k))))(
                    f0, nm
                )
            for ly in ch.layers:
                b_ = getattr(ly, "blk_pf", None)
                if b_ is not None:
                    for mname in ("route", "_pack_route", "_ag_rows", "_rs"):
                        f0 = getattr(b_, mname)
                        setattr(
                            b_, mname, (lambda f0, nm: (lambda *x, **k: T(f"moe.{nm}", lambda: f0(*x, **k))))(f0, mname)
                        )
                    f0 = ly.shared.forward
                    ly.shared.forward = (lambda f0: (lambda *x, **k: T("moe.shared_fwd", lambda: f0(*x, **k))))(f0)
        # attention sub-stages (device-synced): wrap the methods of every layer's attention
        for ly in ch.layers:
            a_ = ly.attn
            for mname in ("_sparse_prefill", "_finish", "_append_cache"):
                f0 = getattr(a_, mname)
                setattr(
                    a_,
                    mname,
                    (lambda f0, nm, tag: (lambda *x, **k: T(f"{tag}.attn.{nm}", lambda: f0(*x, **k))))(
                        f0, mname, f"L{ly.L}"
                    ),
                )
            if a_.has_indexer:
                f0 = a_.indexer.select
                a_.indexer.select = (
                    lambda f0, tag: (lambda *x, **k: T(f"{tag}.attn.indexer.select", lambda: f0(*x, **k)))
                )(f0, f"L{ly.L}")
        run_chunks(K)
        acc.clear()
        disp.clear()
        run_chunks(K)
        tot = {}
        for k, v in acc.items():
            lay = k.split(".")[0]
            tot[lay] = tot.get(lay, 0) + v[-1]
        res(event="dispatch_ms", sections={k: round(v[-1], 3) for k, v in disp.items()})
        res(
            event="sections_ms",
            sections={k: round(v[-1], 3) for k, v in acc.items()},
            per_layer_total_ms={k: round(v, 2) for k, v in tot.items()},
        )
    finally:
        if mesh4:
            close_mesh(md)
        else:
            close_row(full, md)


if __name__ == "__main__":
    main()
