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
    full, md = open_row()
    try:
        mc, ccl = mesh_1x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        ch = HC.Hy4Chain(md, mc, ccl, ckpt=CKPT, kv_fmt=os.environ.get("HY4_KV_FMT", "bf16"))
        t0 = time.time()
        ch.add_layers(0, NL, log=lambda s: print(s, flush=True))
        res(event="built", sec=round(time.time() - t0, 1))
        rnd = torch.Generator().manual_seed(3)
        ids = [torch.randint(0, 100000, (1, HC.CHUNK), generator=rnd) for _ in range(K)]
        emb = ch.rd.get(
            "model.embed_tokens.weight"
        )  # [V, D]: realistic stream inputs (the embedding rows, 4 identical streams like Hy4Chain.embed_ids)
        xs = [emb[ids[c][0]].float().reshape(1, 1, HC.CHUNK, 6144).repeat(1, 4, 1, 1) for c in range(K)]
        del emb
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
                x4 = upload(md, xs[c], ttnn.float32)
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
                    x4 = upload(md, xs[c], ttnn.float32)
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
        run_chunks(K)  # warm
        for r in range(REPS):
            d, w = run_chunks(K, True)
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
        close_row(full, md)


if __name__ == "__main__":
    main()
