"""NAN49: poison-fill + determinism bisect of Hy4Attention.prefill_chunk (layer 0, 1x8 slot, NL=1, HY4_NO_MOE=1).
For each POISON mode (list, env MODES, default 'none,nan,zero,big'): fill all free DRAM (and L1 if L1=1) with the pattern, free it, run chunks 0..CH-1 of the SAME inputs,
tap every attention intermediate (device 0 copy), compare each tap against the first mode's run: first tap that is non-finite or differs = the uninitialised read."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt import hy4_chain as HC
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

CH = int(os.environ.get("CH", "4"))
MODES = os.environ.get("MODES", "none,nan,zero,big").split(",")
REPS = int(os.environ.get("REPS", "2"))
ORDER = ["qa", "lat", "k_pe", "idx_score", "ids", "qfull", "o_lat", "o", "gate", "og", "part"]
NOFIN_OK = ("idx_score",)


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def poison(md, mode, mem):
    val = {"nan": float("nan"), "zero": 0.0, "big": 3.0e38 if False else 6.0e4}[mode]
    held = []
    for mb in (64, 16, 4, 1):
        rows = mb * 1024 * 1024 // (6144 * 2)
        while True:
            try:
                t = ttnn.full(
                    [1, 1, rows, 6144], val, device=md, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=mem
                )
            except Exception:
                break
            held.append(t)
            if len(held) > 4000:
                break
    n = len(held)
    ttnn.synchronize_device(md)
    for t in reversed(held):
        ttnn.deallocate(t)
    return n


def main():
    full, md = open_row()
    try:
        mc, ccl = mesh_1x8(), CCLManager(
            md,
            num_links=int(os.environ.get("LINKS", "2")),
            topology=getattr(ttnn.Topology, os.environ.get("TOPO", "Ring")),
        )
        ch = HC.Hy4Chain(md, mc, ccl, ckpt="/ckpt", kv_fmt=os.environ.get("HY4_KV_FMT", "bf16"))
        ch.add_layers(0, 1, log=lambda s: None)
        emb = ch.rd.get("model.embed_tokens.weight")
        g = torch.Generator().manual_seed(3)
        idsl = [torch.randint(1000, 100000, (HC.CHUNK,), generator=g) for _ in range(CH)]
        xs = [emb[i].float().reshape(1, 1, HC.CHUNK, 6144).repeat(1, 4, 1, 1) for i in idsl]
        ly = ch.layers[0]
        ref = {}
        for mode in MODES:
            if mode != "none":
                nd = poison(md, mode, ttnn.DRAM_MEMORY_CONFIG)
                nl = poison(md, mode, ttnn.L1_MEMORY_CONFIG) if os.environ.get("L1", "1") == "1" else 0
                res(event="poison", mode=mode, dram_tensors=nd, l1_tensors=nl)
            for rep in range(REPS):
                ly.attn.reset()
                ly.attn.taps = {}
                ibl = {}
                for c in range(CH):
                    ly.attn.taps.clear()
                    if os.environ.get("RESETSEM") == "1":
                        ttnn.synchronize_device(md)
                        ccl.reset_global_semaphores()
                    if os.environ.get("PINGFIX") == "1":
                        ccl.rs_ping_pong_idx = ccl.ag_ping_pong_idx = ccl.barrier_idx = 0
                    for nm in os.environ.get("FIXIDX", "").split(","):
                        if nm:
                            setattr(ccl, nm, 0)
                    res(event="idx", chunk=c, rs=ccl.rs_ping_pong_idx, ag=ccl.ag_ping_pong_idx, bar=ccl.barrier_idx)
                    if os.environ.get("SYNCALL") == "1":
                        ttnn.synchronize_device(md)
                    x = upload(md, xs[c], ttnn.float32)
                    x4 = ly.prefill_chunk(x, c * HC.CHUNK, ibl)
                    ttnn.synchronize_device(md)
                    rec = {"mode": mode, "rep": rep, "chunk": c}
                    first_bad = None
                    for k in ORDER:
                        if k not in ly.attn.taps:
                            continue
                        t = ly.attn.taps[k]
                        h = ttnn.to_torch(ttnn.get_device_tensors(t)[0])
                        hf = h.float()
                        fin = bool(torch.isfinite(hf).all())
                        if k == "part":
                            rec["part_bad_devs"] = [
                                d
                                for d, td in enumerate(ttnn.get_device_tensors(t))
                                if not bool(torch.isfinite(ttnn.to_torch(td).float()).all())
                            ]
                        if k in NOFIN_OK:
                            fin = True
                        key = (c, k)
                        if key not in ref:
                            ref[key] = h.clone()
                        same = bool(torch.equal(ref[key], h))
                        rec[k] = ("ok" if fin else "NONFINITE") + ("" if same else "/DIFF")
                        if first_bad is None and (not fin or not same):
                            first_bad = k
                    rec["out_bad_devs"] = [
                        d
                        for d, td in enumerate(ttnn.get_device_tensors(x4))
                        if not bool(torch.isfinite(ttnn.to_torch(td).float()).all())
                    ]
                    rec["out_finite"] = not rec["out_bad_devs"]
                    rec["first_bad"] = first_bad
                    res(**rec)
                    ttnn.deallocate(x4)
                ly.attn.taps = None
                ly.attn.taps = {}
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
