# SPDX-License-Identifier: Apache-2.0
"""Real-weight, real-activation gate for MiMo attention on the 4x8 mesh vs track-0 dumps (teacher forced).
Layer 1 (SWA) and 5 (GA): prefill S=128 B=16 (4 users per mesh row): device attn_out vs dump attn_out with the dump attn_in as input, KV cache contents vs dump KV,
and 3 decode steps (dump dec_in -> input_layernorm on host -> device decode) vs track-0 oracle `attention()` (the dump stores only the whole-layer dec_out).
Env: MIMO_CKPT, ATTN_DUMP (default /mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16), ATTN_LAYERS=1,5, ATTN_WDTYPE."""
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_layer_copy as RL  # noqa: E402
from attn_ref import pcc  # noqa: E402
from harness import close_mesh, dev_rows, dram, gather_all, host_rows, open_mesh, rec  # noqa: E402
from ref_loader_copy import Ckpt, dequant_qkv, is_swa  # noqa: E402

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
DUMP = E("ATTN_DUMP", "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16")
WD = {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[E("ATTN_WDTYPE", "bf8")]
ROWS, UPR = 4, 4  # 16 users = 4 rows x 4 users
S = 128


def main():
    layers = [int(v) for v in E("ATTN_LAYERS", "1,5").split(",")]
    ck = Ckpt()
    mesh = open_mesh()
    try:
        g = torch.Generator().manual_seed(0)
        for L in layers:
            cfg = SWA if is_swa(L) else GA
            p = f"model.layers.{L}."
            qkv = dequant_qkv(
                L, ck.get(p + "self_attn.qkv_proj.weight"), ck.get(p + "self_attn.qkv_proj.weight_scale_inv")
            )
            nq, nk = NQ * DQK, cfg.nkv * DQK
            q_w, k_w, v_w = qkv[:nq], qkv[nq : nq + nk], qkv[nq + nk :]
            o_w = ck.get(p + "self_attn.o_proj.weight").float()
            sink = ck.get(p + "self_attn.attention_sink_bias") if ck.has(p + "self_attn.attention_sink_bias") else None
            ln = ck.get(p + "input_layernorm.weight")
            sd = {"self_attn.qkv_proj.weight": qkv.bfloat16(), "self_attn.o_proj.weight": o_w.bfloat16()}
            if sink is not None:
                sd["self_attn.attention_sink_bias"] = sink
            d = torch.load(f"{DUMP}/layer_{L}.pt")
            d32 = torch.load(f"{DUMP}-f32/layer_{L}.pt")
            hw = convert_weights(cfg, q_w, k_w, v_w, o_w, sink)
            ring = S // BLOCK + 8 if cfg.window else 0
            maxpos = 256 + 128
            nb = ring if cfg.window else maxpos // BLOCK
            perm = torch.randperm(32 * nb, generator=g).reshape(32, nb).to(torch.int32)
            m = MiMoAttention(mesh, cfg, hw, perm, maxpos, ring_blocks=ring, wdtype=WD)
            io = m.prefill_io(S)
            xdev = dev_rows(mesh, torch.zeros(ROWS, 1, S, HIDDEN).bfloat16())
            ain, aout = d["prefill"]["attn_in"], d["prefill"]["attn_out"]
            aout32 = d32["prefill"]["attn_out"]
            out = torch.zeros(ROWS * UPR, S, HIDDEN)
            for ul in range(UPR):
                xin = torch.stack([ain[r * UPR + ul] for r in range(ROWS)]).reshape(ROWS, 1, S, HIDDEN)
                ttnn.copy_host_to_device_tensor(host_rows(mesh, xin), xdev)
                m.prefill_set(io, 0, ul)
                o = m.prefill_chunk(xdev, io)
                h = gather_all(mesh, o)  # [rows, 8, S, 4096]
                assert float((h - h[:, :1]).abs().max()) == 0.0
                for r in range(ROWS):
                    out[r * UPR + ul] = h[r, 0, 0] if h.dim() == 4 and h.shape[2] == 1 else h[r, 0]
                ttnn.deallocate(o)
            per_user = [pcc(out[b], aout[b]) for b in range(16)]
            per_user32 = [pcc(out[b], aout32[b]) for b in range(16)]
            floor = [pcc(aout[b], aout32[b]) for b in range(16)]
            rec(
                "dump_prefill_attn_out",
                layer=L,
                kind=cfg.kind,
                pcc_vs_bf16_dump_min=round(min(per_user), 5),
                pcc_vs_f32_dump_min=round(min(per_user32), 5),
                bf16_dump_vs_f32_dump_min=round(min(floor), 5),
                per_row_min=[round(min(per_user[r * UPR : (r + 1) * UPR]), 5) for r in range(ROWS)],
            )
            # ---- KV contents
            kd = ttnn.get_device_tensors(m.k_cache)
            vd = ttnn.get_device_tensors(m.v_cache)
            kk, vv, pad = [], [], []
            ptab = m.page_table
            for b in range(16):
                r, ul = divmod(b, UPR)
                for c in range(8):
                    kh = c * cfg.nkv // 8
                    kcache = ttnn.to_torch(kd[r * 8 + c]).float()
                    vcache = ttnn.to_torch(vd[r * 8 + c]).float()
                    blocks = [int(ptab[ul][i % ring]) if ring else int(ptab[ul][i]) for i in range(S // BLOCK)]
                    kdev = torch.cat([kcache[bl, 0] for bl in blocks], 0)  # [S,192]
                    vdev = torch.cat([vcache[bl, 0] for bl in blocks], 0)
                    if cfg.window:
                        kref, vref = (
                            d["state"]["k_ring"][b, kh].float(),
                            d["state"]["v_ring"][b, kh].float(),
                        )  # slot = position (S == window)
                    else:
                        kref, vref = d["state"]["k"][b, kh].float(), d["state"]["v"][b, kh].float()
                    kk.append(pcc(kdev, kref))
                    vv.append(pcc(vdev[:, :DV], vref))
                    pad.append(float(vdev[:, DV:].abs().max()))
            rec(
                "dump_prefill_kv",
                layer=L,
                kind=cfg.kind,
                k_pcc_min=round(min(kk), 5),
                v_pcc_min=round(min(vv), 5),
                v_pad_absmax=max(pad),
            )
            # ---- decode, 3 teacher-forced steps. reference = track-0 attention() with the dump KV as past
            dio = m.decode_io()
            dx = dev_rows(mesh, torch.zeros(ROWS, 1, 32, HIDDEN).bfloat16())
            nsteps = len(d["decode"])
            past = {}
            for b in range(16):
                if cfg.window:
                    past[b] = (d["state"]["k_ring"][b : b + 1].clone(), d["state"]["v_ring"][b : b + 1].clone())
                else:
                    past[b] = (d["state"]["k"][b : b + 1].clone(), d["state"]["v"][b : b + 1].clone())
            res = []
            for step in range(nsteps):
                dec_in = d["decode"][step]["dec_in"]  # [16,1,4096] layer input hidden
                xn = RL.rms_norm(dec_in, ln)  # bf16 normed
                xin = torch.zeros(ROWS, 32, HIDDEN).bfloat16()
                for b in range(16):
                    r, ul = divmod(b, UPR)
                    xin[r, ul] = xn[b, 0]
                m.decode_set(dio, [S + step] * 32)
                ttnn.copy_host_to_device_tensor(host_rows(mesh, xin.reshape(ROWS, 1, 32, HIDDEN)), dx)
                o = m.decode(dx, dio["cur"], dio["idx"])
                h = gather_all(mesh, o)
                cs = float((h - h[:, :1]).abs().max())
                pcs = []
                for b in range(16):
                    r, ul = divmod(b, UPR)
                    ref_o, kn, vn = RL.attention(xn[b : b + 1], sd, L, kv_past=past[b], q0=S + step)
                    past[b] = (torch.cat([past[b][0], kn], 2), torch.cat([past[b][1], vn], 2))
                    pcs.append(pcc(h[r, 0, ul], ref_o[0, 0]))
                res.append(dict(step=step, pcc_min=round(min(pcs), 5), col_spread=cs))
                ttnn.deallocate(o)
            rec(
                "dump_decode_attn_out",
                layer=L,
                kind=cfg.kind,
                steps=res,
                note="reference = track-0 attention() with dump KV past; dump has no per-step attn_out",
            )
            rec("dram", layer=L, dram=dram(mesh))
            del m
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
