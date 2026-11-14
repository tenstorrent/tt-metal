"""pf-integrate S2: WHOLE-MESH (4x8, all four slots) 78-layer prefill + decode, 4 users (one per mesh row).
MODE=pf  : Hy4PoolPrefill (direct-to-pool, HY4_POOL_PF=1, no hand-off)      MODE=rep : integ replicated-cache chain prefill + dec.handoff (HY4_POOL_PF=0; prompts <= ~36k tokens)
Env: SRC (needle .pt of gen_needle_prompts.py), LEN (truncate every prompt to LEN tokens; 0 = full), NSTEPS (greedy decode steps, 64), TAGNAME, PF_ROWS (512|1024), MEMS (mem sampler interval s, 0 off),
     MAXCHUNKS (debug: stop after that many chunks), REPEAT (a second identical prefill + decode of the same prompts: determinism), OUT.
Writes $OUT/pf-integrate/res_<TAG>.pt (first tokens, last-token hidden [4,6144] and logits argmax, decode tokens, per-chunk seconds) and prints RES json lines."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt.hy4_chain import Hy4Chain, dram_view
from models.demos.blackhole.hy4_preview.tt.hy4_pool_model import Hy4PoolDecoder
from models.demos.blackhole.hy4_preview.tt.hy4_pool_prefill_model import Hy4PoolPrefill, decode_bucket
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

OUT = os.environ.get("OUT", "/out")
MODE = os.environ.get("MODE", "pf")
PF = int(os.environ.get("PF_ROWS", "512"))
NSTEPS = int(os.environ.get("NSTEPS", "64"))
TAG = os.environ.get("TAGNAME", MODE)
PAD = 120002


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def decode(dec, first, lens, nsteps):
    toks = [[t] for t in first]
    pos = list(lens)
    dec.set_inputs([t[-1] for t in toks], pos)
    dec.capture()
    for _ in range(nsteps):
        dec.set_inputs([t[-1] for t in toks], pos)
        nxt, _ = dec.read_argmax(dec.run())
        for r in range(4):
            toks[r].append(nxt[r])
            pos[r] += 1
    dec.release()
    return toks


def rep_prefill(ch, dec, md, prompts):
    L = max(len(p) for p in prompts)
    nch = -(-L // PF)
    first, hid = [None] * 4, [None] * 4
    t0 = time.time()
    for c in range(nch):
        ids = torch.full((4, PF), PAD, dtype=torch.long)
        for r in range(4):
            seg = prompts[r][c * PF : (c + 1) * PF]
            if seg:
                ids[r, : len(seg)] = torch.tensor(seg)
        x4, _ = ch.prefill(ids, start=c * PF)
        hn, logits = ch.head(x4)
        lg = (
            ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=tuple(md.shape)))
            .reshape(4, PF, -1)
            .float()
        )
        devs = ttnn.get_device_tensors(hn)
        for r in range(4):
            if c * PF <= len(prompts[r]) - 1 < (c + 1) * PF:
                last = len(prompts[r]) - 1 - c * PF
                first[r] = int(lg[r, last].argmax())
                hid[r] = ttnn.to_torch(devs[r * 8]).float().reshape(PF, -1)[last]
        ttnn.synchronize_device(md)
        ttnn.deallocate(x4)
        ttnn.deallocate(hn)
        ttnn.deallocate(logits)
        if c % 8 == 0 or c == nch - 1:
            res(event="prefill_chunk", mode="rep", c=c, of=nch, sec=round(time.time() - t0, 1))
    dec.handoff(nch * PF)
    ttnn.synchronize_device(md)
    ch.reset()
    return first, hid, [len(p) for p in prompts], time.time() - t0


def main():
    assert os.environ.get("HY4_POOL") == "1"
    assert (os.environ.get("HY4_POOL_PF") == "1") == (MODE == "pf")
    srcs = [
        torch.load(x) for x in os.environ["SRC"].split(",")
    ]  # several prompt files in one boot (needles 64k then 256k)
    d = srcs[0]
    LENS = [
        int(x) for x in os.environ.get("LENS", os.environ.get("LEN", "0")).split(",")
    ]  # several lengths in one boot (gates)
    fulls = [[list(p) for p in x["ids"]] for x in srcs]
    L = max(max(len(p[:n]) if n else len(p) for p in f) for f in fulls for n in LENS)
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "700e6"))))
    try:
        from models.demos.blackhole.hy4_preview.tt import hy4_pool_decode as _pd
        from models.demos.blackhole.hy4_preview.tt.hy4_ccl_safe import hy4_fabric_selftest

        links = hy4_fabric_selftest(md)
        _pd._POOL_AG_LINKS = links.pool_ag
        import models.demos.blackhole.hy4_preview.tt.hy4_pf_attention as _pa

        _pa.LINKS = links.pool_ag
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=links.num_links, topology=ttnn.Topology.Ring)
        ch = Hy4Chain(md, mc, ccl, ckpt=os.environ.get("HY4_CKPT", "/ckpt"))
        ch.add_layers(0, 78, log=lambda s: None)
        ch.add_layers(78, 79, log=lambda s: None)
        ch.build_head()
        pages = (-(-L // PF)) * PF // 512 + 2
        dec = Hy4PoolDecoder(ch, pages, mode="topk")
        dec.ctx.set_bucket(decode_bucket(L, pages), "topk")
        res(event="built", mode=MODE, L=L, pages=pages, bucket=dec.ctx.bucket_tokens, **dram_view(md))
        out = dict(L=L, mode=MODE, keys=d.get("keys"), depths=d.get("depths"))
        pp = None
        if MODE == "pf":
            pp = Hy4PoolPrefill(
                dec,
                pf_rows=PF,
                log=lambda **kw: res(**kw) if (kw["c"] % 8 == 0 or kw["c"] == kw["of"] - 1 or kw["c"] < 6) else None,
                mem_sample_s=float(os.environ.get("MEMS", "0")),
            )
        maxch = int(os.environ.get("MAXCHUNKS", "0"))
        for si, rep, LEN in [
            (s_, r_, n_)
            for s_ in range(len(srcs))
            for n_ in LENS
            for r_ in range(1 + int(os.environ.get("REPEAT", "0")))
        ]:
            d, full = srcs[si], fulls[si]
            prompts = [p[:LEN] if LEN else p for p in full]
            if MODE == "pf":
                if maxch:
                    prompts = [p[: maxch * PF] for p in prompts]
                r = pp.prefill(prompts, keep_hidden=True)
                first, hid, lens, sec = r.first, r.hidden, r.lens, r.total_sec
                out[f"chunk_sec_s{si}"] = r.chunk_sec
                if pp.mem:
                    out["mem_peak_alloc_per_bank"] = pp.mem_peak_per_bank
                    out["mem_min_free_per_bank"] = pp.mem_min_free_per_bank
                    res(
                        event="mem",
                        peak_alloc_per_bank=pp.mem_peak_per_bank,
                        min_free_per_bank=pp.mem_min_free_per_bank,
                    )
            else:
                first, hid, lens, sec = rep_prefill(ch, dec, md, prompts)
            res(
                event="prefill_done",
                src=si,
                rep=rep,
                LEN=LEN,
                sec=round(sec, 1),
                first=first,
                lens=lens,
                **dram_view(md),
            )
            if maxch:
                return
            dec.ctx.set_bucket(decode_bucket(max(lens), pages), "topk")
            toks = decode(dec, first, lens, NSTEPS)
            res(event="generated", src=si, rep=rep, LEN=LEN, toks=toks, keys=d.get("keys"), depths=d.get("depths"))
            (
                out[f"first{LEN}_{rep}_s{si}"],
                out[f"hid{LEN}_{rep}_s{si}"],
                out[f"toks{LEN}_{rep}_s{si}"],
                out[f"sec{LEN}_{rep}_s{si}"],
            ) = (
                first,
                torch.stack(hid),
                toks,
                sec,
            )
        os.makedirs(f"{OUT}/pf-integrate", exist_ok=True)
        torch.save(out, f"{OUT}/pf-integrate/res_{TAG}.pt")
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
