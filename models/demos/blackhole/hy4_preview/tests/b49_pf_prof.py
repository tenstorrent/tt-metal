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
        xs = [(torch.randn(1, 4, HC.CHUNK, 6144, generator=rnd) * 0.5) for _ in range(K)]
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
