# SPDX-License-Identifier: Apache-2.0
"""Mesh-level (4x8) chunked prefill + decode test of MiMo attention (SWA and global layer) vs torch fp32 HF-semantics reference.
Run: ATTN_KINDS=swa,ga [ATTN_TRACE=1] [ATTN_REAL=1] python test_attention_mesh.py ; results: JSON lines `ATTN_RESULT ...` (PCC gate 0.999).
Env: ATTN_CH chunk (default 1024), ATTN_NCH max chunks per user, ATTN_U users prefilled per row (<=32), ATTN_ND decode steps, ATTN_RAGGED=1 (user u gets
NCH-u%NCH chunks), ATTN_MANY=1 (32 users, chunk 256, lengths 256*(1+u%4)), ATTN_TRACE=1 (prefill + decode run through captured traces and replays),
ATTN_REAL=1 (real layer 1 (swa) / 5 (ga) weights from MIMO_CKPT, real text tokens through real embedding + input_layernorm), ATTN_WDTYPE=bf8|bf16, ATTN_PERF=1.
4 mesh rows = 4 independent user groups with different data; columns = TP8."""
import glob
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_ref import pcc, ref_attention  # noqa: E402
from harness import Traced, close_mesh, dev_rows, dram, gather_all, host_rows, open_mesh, rec  # noqa: E402

from models.demos.blackhole.mimo_v26.tt.attention import (  # noqa: E402
    BLOCK,
    DQK,
    DV,
    GA,
    HIDDEN,
    NQ,
    SWA,
    MiMoAttention,
    convert_weights,
)

E = os.environ.get
MANY = E("ATTN_MANY", "0") == "1"
CH = 256 if MANY else int(E("ATTN_CH", "1024"))
NCH = int(E("ATTN_NCH", "3"))
U = 32 if MANY else int(E("ATTN_U", "2"))
ND = int(E("ATTN_ND", "3"))
SINKM = float(E("ATTN_SINKM", "3"))
RAGGED = E("ATTN_RAGGED", "0") == "1"
TRACE = E("ATTN_TRACE", "0") == "1"
REAL = E("ATTN_REAL", "0") == "1"
WD = {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[E("ATTN_WDTYPE", "bf8")]
LENS = [CH * (1 + u % 4) if MANY else (CH * max(1, NCH - u % NCH) if RAGGED else CH * NCH) for u in range(U)]
PMAX = max(LENS)
MAXPOS = ((PMAX + ND + 255) // 256) * 256 + 256
ROWS = 4
REAL_LAYER = {"swa": 1, "ga": 5}


def random_weights(cfg, g):
    q_w = torch.randn(NQ * DQK, HIDDEN, generator=g) / 64
    k_w = torch.randn(cfg.nkv * DQK, HIDDEN, generator=g) / 64
    v_w = torch.randn(cfg.nkv * DV, HIDDEN, generator=g) / 64
    o_w = torch.randn(HIDDEN, NQ * DV, generator=g) / 90
    sink = ((torch.rand(NQ, generator=g) * 0.58 + 0.56) * SINKM).bfloat16() if cfg.sink else None
    return q_w, k_w, v_w, o_w, sink


def real_inputs(kind, nseq, T):
    """real layer weights + normed real-token embeddings: x [nseq, T, 4096] bf16"""
    from ref_loader_copy import Ckpt, dequant_qkv

    L = REAL_LAYER[kind]
    ck = Ckpt()
    p = f"model.layers.{L}."
    qkv = dequant_qkv(L, ck.get(p + "self_attn.qkv_proj.weight"), ck.get(p + "self_attn.qkv_proj.weight_scale_inv"))
    nq, nk = NQ * DQK, (SWA if kind == "swa" else GA).nkv * DQK
    q_w, k_w, v_w = qkv[:nq], qkv[nq : nq + nk], qkv[nq + nk :]
    o_w = ck.get(p + "self_attn.o_proj.weight").float()
    sink = ck.get(p + "self_attn.attention_sink_bias") if ck.has(p + "self_attn.attention_sink_bias") else None
    ln = ck.get(p + "input_layernorm.weight").float()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.environ.get("MIMO_CKPT", "/mnt/tt-data/MiMo-V2.6-Flash-RL"))
    files = sorted(
        glob.glob("/mnt/tt-data/ssinghal/mimo/wt/attn/docs/**/*.rst", recursive=True)
        + glob.glob("/mnt/tt-data/ssinghal/mimo/wt/attn/tech_reports/**/*.md", recursive=True)
    )
    text = ""
    for f in files:
        text += open(f, errors="ignore").read() + "\n\n"
        if len(text) > 12 * nseq * T:
            break
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= nseq * T, (len(ids), nseq * T)
    ids = torch.tensor(ids[: nseq * T]).reshape(nseq, T)
    emb = ck.get("model.embed_tokens.weight")
    h = emb[ids].float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * ln
    rec(
        "real_inputs",
        kind=kind,
        layer=L,
        tokens=nseq * T,
        sink_range=None if sink is None else [float(sink.min()), float(sink.max())],
        q_absmean=float(q_w.abs().mean()),
        k_absmean=float(k_w.abs().mean()),
        v_absmean=float(v_w.abs().mean()),
        o_absmean=float(o_w.abs().mean()),
        x_std=float(h.std()),
    )
    return (q_w, k_w, v_w, o_w, sink), h.bfloat16()


def main():
    kinds = E("ATTN_KINDS", "swa,ga").split(",")
    mesh = open_mesh()
    try:
        g = torch.Generator().manual_seed(0)
        for kind in kinds:
            cfg = SWA if kind == "swa" else GA
            T = PMAX + ND
            if REAL:
                (q_w, k_w, v_w, o_w, sink), xr = real_inputs(kind, ROWS * U, T)
                x = xr.reshape(ROWS, U, T, HIDDEN)
            else:
                q_w, k_w, v_w, o_w, sink = random_weights(cfg, g)
                x = torch.randn(ROWS, U, T, HIDDEN, generator=g).bfloat16()
            hw = convert_weights(cfg, q_w, k_w, v_w, o_w, sink)
            ring = CH // BLOCK + 8 if cfg.window else 0
            nb = ring if cfg.window else MAXPOS // BLOCK
            perm = torch.randperm(32 * nb, generator=g).reshape(32, nb).to(torch.int32)
            t0 = time.time()
            m = MiMoAttention(mesh, cfg, hw, perm, MAXPOS, ring_blocks=ring, wdtype=WD)
            rec(
                "build",
                kind=kind,
                secs=round(time.time() - t0, 1),
                ring=ring or None,
                trace=TRACE,
                real=REAL,
                users=U,
                lens=LENS[:8],
                dram=dram(mesh),
            )
            t0 = time.time()
            ref = torch.zeros(ROWS, U, T, HIDDEN)
            for r in range(ROWS):
                for u in range(U):
                    n = LENS[u] + ND
                    ref[r, u, :n] = ref_attention(x[r, u, :n], q_w, k_w, v_w, o_w, sink, cfg.nkv, cfg.theta, cfg.window)
            rec("ref_done", kind=kind, secs=round(time.time() - t0, 1))
            # ---- chunked prefill, user by user
            io = m.prefill_io(CH)
            xdev = dev_rows(mesh, torch.zeros(ROWS, 1, CH, HIDDEN).bfloat16())
            pre = torch.zeros(ROWS, U, PMAX, HIDDEN)
            colspread = 0.0
            tr = None
            replay_same = None
            t0 = time.time()
            for u in range(U):
                for c in range(LENS[u] // CH):
                    xin = x[:, u, c * CH : (c + 1) * CH].reshape(ROWS, 1, CH, HIDDEN)
                    ttnn.copy_host_to_device_tensor(host_rows(mesh, xin), xdev)
                    m.prefill_set(io, c * CH, u)
                    if TRACE:
                        if tr is None:
                            tr = Traced(mesh, lambda: m.prefill_chunk(xdev, io))
                        o = tr.replay()
                    else:
                        o = m.prefill_chunk(xdev, io)
                    h = gather_all(mesh, o)  # [rows, 8, CH, 4096]
                    colspread = max(colspread, float((h - h[:, :1]).abs().max()))
                    pre[:, u, c * CH : (c + 1) * CH] = h[:, 0]
                    if TRACE:
                        last = (u, c)
                    else:
                        ttnn.deallocate(o)
            if TRACE:  # bit-identical across replays: replay the very last chunk twice more (cache fill is idempotent)
                same = True
                for _ in range(2):
                    h2 = gather_all(mesh, tr.replay())
                    same &= bool(torch.equal(h2[:, 0], pre[:, last[0], last[1] * CH : (last[1] + 1) * CH]))
                rec("prefill_replay_bit_identical", kind=kind, ok=same)
            rec("prefill_time", kind=kind, secs=round(time.time() - t0, 1))
            res = {
                f"r{r}u{u}": round(pcc(pre[r, u, : LENS[u]], ref[r, u, : LENS[u]]), 5)
                for r in range(ROWS)
                for u in range(U)
            }
            per_chunk = {}
            for c in range(max(LENS) // CH):
                v = [
                    pcc(pre[r, u, c * CH : (c + 1) * CH], ref[r, u, c * CH : (c + 1) * CH])
                    for r in range(ROWS)
                    for u in range(U)
                    if (c + 1) * CH <= LENS[u]
                ]
                per_chunk[f"c{c}"] = round(min(v), 5)
            rec(
                "prefill_pcc",
                kind=kind,
                min=min(res.values()),
                min_per_row={r: min(v for k, v in res.items() if k.startswith(f"r{r}")) for r in range(ROWS)},
                min_per_chunk=per_chunk,
                col_spread_absmax=colspread,
                trace=TRACE,
                real=REAL,
                users=U,
            )
            # ---- decode: 32 slots (slots >= U hold empty caches)
            xd = torch.randn(ROWS, 32, ND, HIDDEN, generator=g).bfloat16()
            for u in range(U):
                xd[:, u] = x[:, u, LENS[u] : LENS[u] + ND]
            dio = m.decode_io()
            dx = dev_rows(mesh, torch.zeros(ROWS, 1, 32, HIDDEN).bfloat16())
            dtr = None
            dres = []
            for step in range(ND):
                pos = [(LENS[u] if u < U else PMAX) + step for u in range(32)]
                m.decode_set(dio, pos)
                ttnn.copy_host_to_device_tensor(host_rows(mesh, xd[:, :, step].reshape(ROWS, 1, 32, HIDDEN)), dx)
                if TRACE:
                    if dtr is None:
                        dtr = Traced(mesh, lambda: m.decode(dx, dio["cur"], dio["idx"]))
                    o = dtr.replay()
                else:
                    o = m.decode(dx, dio["cur"], dio["idx"])
                h = gather_all(mesh, o)
                cs = float((h - h[:, :1]).abs().max())
                dd = {
                    f"r{r}u{u}": round(pcc(h[r, 0, u], ref[r, u, LENS[u] + step]), 5)
                    for r in range(ROWS)
                    for u in range(U)
                }
                dres.append(
                    dict(
                        step=step,
                        min=min(dd.values()),
                        col_spread=cs,
                        per_user_min_per_row={
                            r: min(v for k, v in dd.items() if k.startswith(f"r{r}")) for r in range(ROWS)
                        },
                    )
                )
                if step == ND - 1:  # determinism: same step again (cache write is idempotent)
                    same = True
                    for _ in range(3):
                        o2 = dtr.replay() if TRACE else m.decode(dx, dio["cur"], dio["idx"])
                        same &= bool(torch.equal(gather_all(mesh, o2), h))
                        if not TRACE:
                            ttnn.deallocate(o2)
                    rec("decode_replay_bit_identical" if TRACE else "decode_determinism_3x", kind=kind, ok=same)
                if not TRACE:
                    ttnn.deallocate(o)
            rec("decode_pcc", kind=kind, min=min(d["min"] for d in dres), steps=dres, trace=TRACE, real=REAL, users=U)
            if E("ATTN_PERF", "0") == "1":
                t0 = time.perf_counter()
                for _ in range(20):
                    (dtr.replay(block=False) if TRACE else ttnn.deallocate(m.decode(dx, dio["cur"], dio["idx"])))
                ttnn.synchronize_device(mesh)
                rec("perf", kind=kind, decode_ms=round((time.perf_counter() - t0) / 20 * 1e3, 3), trace=TRACE)
            if TRACE:
                tr.release()
                dtr.release()
            rec("dram_after", kind=kind, dram=dram(mesh))
            del m
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
