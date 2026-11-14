"""Hy4 head on one 1x8 slot row: variants x rows. Env: VARIANTS="bf16:fp32,bfp8:fp32,..." (head weight dtype : logits dtype [:fidelity]), TS="1,2,4,8,16", NROWS (real rows for accuracy), FAB=1d|1d_ring, STAGES=acc,lat,smp,emb.
Prints JSON lines `RES {...}`."""

import json
import os
import sys
import time

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/container_app_user/tt-metal")
sys.path.insert(0, os.path.join(HERE, "..", "tt"))
from hy4_sampler import Hy4DeviceSampler, sampling_rows  # noqa

from hy4_head import DIM, VOCAB, Hy4DeviceEmbedding, Hy4DeviceHead, head_bytes_per_chip  # noqa
from models.demos.gpt_oss.config import mesh_4x8  # noqa
from models.demos.gpt_oss.tt.ccl import CCLManager  # noqa

LMH = os.environ.get("LMH", "/lmh")
FAB = os.environ.get("FAB", "1d")
TS = [int(x) for x in os.environ.get("TS", "1,2,4,8,16").split(",")]
STAGES = os.environ.get("STAGES", "acc,lat").split(",")
NROWS = int(os.environ.get("NROWS", "3385"))
VARIANTS = [v.split(":") for v in os.environ.get("VARIANTS", "bf16:fp32").split(",")]
ITERS = int(os.environ.get("ITERS", "50"))


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


ttnn.set_fabric_config({"1d": ttnn.FabricConfig.FABRIC_1D, "1d_ring": ttnn.FabricConfig.FABRIC_1D_RING}[FAB])
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
topo = ttnn.Topology.Ring if FAB == "1d_ring" else ttnn.Topology.Linear
mc, ccl = mesh_4x8(), CCLManager(md, num_links=int(os.environ.get("LINKS", "1")), topology=topo)
res(event="open", devices=[d for d in md.get_device_ids()], fab=FAB)

W = torch.load(f"{LMH}/head_weights.pt")
R = torch.load(f"{LMH}/" + os.environ.get("REF", "head_ref.pt"))
X = R["X"][:NROWS]  # [N,4,D] bf16
top1, marg = R["top1"][:NROWS], R["marg"][:NROWS]
sub = [i for i in R["sub"] if i < NROWS]
sub_logits = R["sub_logits"][: len(sub)]


def streams_tt(rows):
    return ttnn.from_torch(
        rows.float().reshape(-1, 1, 4, DIM),
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


def timed_trace(fn, iters=ITERS):
    fn()  # compile
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    for _ in range(3):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    t = time.perf_counter()
    for _ in range(iters):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    us = (time.perf_counter() - t) / iters * 1e6
    ttnn.release_trace(md, tid)
    return us


for v in VARIANTS:
    hd, od = v[0], v[1]
    fid = v[2] if len(v) > 2 else "HiFi2"
    acc32 = not (len(v) > 3 and v[3] == "nofp32")
    name = f"{hd}:{od}:{fid}" + ("" if acc32 else ":bf16acc")
    t0 = time.time()
    head = Hy4DeviceHead(md, mc, ccl, W, head_dtype=hd, out_dtype=od, fidelity=fid, fp32_acc=acc32, packer_l1=acc32)
    res(event="built", variant=name, s=round(time.time() - t0, 1), bytes_per_chip_head=head_bytes_per_chip(hd))
    if "acc" in STAGES:
        T = max(TS)
        toks, logit_rows = [], {}
        for i in range(0, X.shape[0], T):
            ch = X[i : i + T]
            if ch.shape[0] < T:  # pad by repeating the last row
                ch = torch.cat([ch, ch[-1:].repeat(T - ch.shape[0], 1, 1)])
            lg = head.forward(streams_tt(ch))
            full = head.gather_logits(lg)  # [T, V]
            n = min(T, X.shape[0] - i)
            toks.append(full[:n].argmax(-1))
            for j in range(n):
                if i + j in sub:
                    logit_rows[i + j] = full[j]
        tk = torch.cat(toks)
        agree = (tk == top1).float()
        ref_max = None
        pc = [pcc(logit_rows[i], sub_logits[k]) for k, i in enumerate(sub) if i in logit_rows]
        mad = max(float((logit_rows[i] - sub_logits[k]).abs().max()) for k, i in enumerate(sub) if i in logit_rows)
        # regret: reference logit of the reference top-1 minus the reference logit of the device top-1 (needs ref logits: only for the subset)
        reg = []
        for k, i in enumerate(sub):
            if i in logit_rows:
                reg.append(float(sub_logits[k].max() - sub_logits[k][int(tk[i])]))
        res(
            event="acc",
            variant=name,
            rows=int(X.shape[0]),
            top1_agree=float(agree.mean()),
            top1_agree_margin_ge_0p05=float(agree[marg >= 0.05].mean()),
            top1_agree_margin_ge_0p2=float(agree[marg >= 0.2].mean()),
            n_flip=int((1 - agree).sum()),
            pcc_min=min(pc),
            pcc_mean=sum(pc) / len(pc),
            max_abs_err=mad,
            regret_mean=sum(reg) / len(reg),
            regret_max=max(reg),
        )
    if (
        "hacc" in STAGES
    ):  # lm_head variant alone: the CPU reference hidden H (bf16) goes straight into the matmul (no hc_head / norm on device)
        Hh = R["H"][:NROWS]
        T = max(TS)
        toks, logit_rows = [], {}
        for i in range(0, Hh.shape[0], T):
            ch = Hh[i : i + T]
            if ch.shape[0] < T:
                ch = torch.cat([ch, ch[-1:].repeat(T - ch.shape[0], 1)])
            ht = ttnn.from_torch(
                ch.reshape(1, 1, T, DIM).float() if hd == "fp32" else ch.reshape(1, 1, T, DIM),
                device=md,
                dtype=ttnn.float32 if hd == "fp32" else ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            )
            full = head.gather_logits(head.matmul_only(ht))
            n = min(T, Hh.shape[0] - i)
            toks.append(full[:n].argmax(-1))
            for j in range(n):
                if i + j in sub:
                    logit_rows[i + j] = full[j]
        tk = torch.cat(toks)
        agree = (tk == top1).float()
        pc = [pcc(logit_rows[i], sub_logits[k]) for k, i in enumerate(sub) if i in logit_rows]
        mad = max(float((logit_rows[i] - sub_logits[k]).abs().max()) for k, i in enumerate(sub) if i in logit_rows)
        reg = [float(sub_logits[k].max() - sub_logits[k][int(tk[i])]) for k, i in enumerate(sub) if i in logit_rows]
        res(
            event="hacc",
            variant=name,
            rows=int(Hh.shape[0]),
            top1_agree=float(agree.mean()),
            top1_agree_margin_ge_0p01=float(agree[marg >= 0.01].mean()),
            top1_agree_margin_ge_0p05=float(agree[marg >= 0.05].mean()),
            n_flip=int((1 - agree).sum()),
            n_flip_margin_ge_0p01=int((1 - agree)[marg >= 0.01].sum()),
            flip_margins=[round(float(m), 4) for m in marg[agree == 0][:20]],
            pcc_min=min(pc),
            pcc_mean=sum(pc) / len(pc),
            max_abs_err=mad,
            regret_mean=sum(reg) / len(reg),
            regret_max=max(reg),
        )
    if "nacc" in STAGES:  # final-layer: device RMSNorm of the hc_head output + lm_head (no streams are dumped)
        HCr = R["HC"][:NROWS]
        T = max(TS)
        toks = []
        for i in range(0, HCr.shape[0], T):
            ch = HCr[i : i + T]
            n = ch.shape[0]
            if n < T:
                ch = torch.cat([ch, ch[-1:].repeat(T - n, 1)])
            y = ttnn.from_torch(
                ch.reshape(1, 1, T, DIM).float(),
                device=md,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            )
            ms2 = ttnn.mean(ttnn.multiply(y, y), dim=-1, keepdim=True)
            h = ttnn.typecast(
                ttnn.multiply(ttnn.multiply(y, ttnn.rsqrt(ttnn.add(ms2, 1e-5))), head.norm_w), ttnn.bfloat16
            )
            toks.append(head.gather_logits(head.matmul_only(h))[:n].argmax(-1))
        tk = torch.cat(toks)
        agree = (tk == top1).float()
        res(
            event="nacc",
            variant=name,
            rows=int(HCr.shape[0]),
            top1_agree=float(agree.mean()),
            n_flip=int((1 - agree).sum()),
            flip_margins=[round(float(m), 4) for m in marg[agree == 0][:20]],
        )
    if "lat" in STAGES:
        for T in TS:
            s = streams_tt(X[:T])
            h = head.hid(s)
            us_full = timed_trace(lambda: head.forward(s))
            us_hid = timed_trace(lambda: head.hid(s))
            us_mm = timed_trace(lambda: head.matmul_only(h))
            lg = head.forward(s)
            us_greedy = timed_trace(lambda: head.sample_global(lg))
            res(
                event="lat",
                variant=name,
                T=T,
                forward_us=round(us_full, 1),
                hidden_us=round(us_hid, 1),
                matmul_us=round(us_mm, 1),
                greedy_us=round(us_greedy, 1),
            )
    del head
    ttnn.synchronize_device(md)
ttnn.close_mesh_device(md)
res(event="done")
