# SPDX-License-Identifier: Apache-2.0
"""L0..L(n-1) decoder-layer chain on the 4x8 mesh vs the track-0 dumps (B=16, S=128, 4 users per mesh row; bf16 dump + fp32 dump = noise floor).

Phase A (teacher forced, CHAIN_TF=1): per layer, dump h_in -> device layer (real bfp8 weights, device router, bf16-activation experts) -> attn_in, attn_out,
  ffn_in, ffn_out, h_out PCC vs bf16 dump and vs fp32 dump, with the bf16-dump-vs-fp32-dump floor of the same field; MoE routing expert-SET match vs dump;
  ffn_out PCC restricted to tokens whose routed set equals the dump set; then (CHAIN_DECODE=1) 3 teacher-forced decode steps (dump dec_in) vs dump dec_out.
Phase B (free running, CHAIN_FREE=1): device embedding of the dump tokens -> all layers chained on device, per-layer h_out PCC vs bf16 / fp32 dump + floor.
Env: MIMO_CKPT, CHAIN_LAYERS (6), CHAIN_DUMP, CHAIN_OUT (jsonl), MIMO_EXPERT_OP (mm), MIMO_ROUTER_MODE (exact).
"""
import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.attn_ref import pcc
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel

E = os.environ.get
DUMP = E("CHAIN_DUMP", "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16")
NL = int(E("CHAIN_LAYERS", "6"))
OUT = E("CHAIN_OUT", "/tmp/chain.jsonl")
ROWS, UPR, S = 4, 4, 128
H = 4096


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("CHAIN_RESULT " + s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")


def row_major(t):  # [16,S,H] (batch = r*UPR+u) -> [ROWS, UPR*S, H]
    return t.reshape(ROWS, UPR * S, -1)


def per_user(a, b):
    return [pcc(a[i], b[i]) for i in range(a.shape[0])]


def rows_users(h):  # [ROWS, UPR*S, H] -> [16,S,H]
    return h.reshape(ROWS * UPR, S, -1)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "trace_region_size": 64 * 1024 * 1024,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_chain_dump(mesh_device):
    md = mesh_device
    dumps = [torch.load(f"{DUMP}/layer_{L}.pt") for L in range(NL)]
    d32 = [torch.load(f"{DUMP}-f32/layer_{L}.pt") for L in range(NL)]
    toks = torch.load(f"{DUMP}/tokens.pt")["prefill_tokens"]  # [16,128]
    model = MiMoModel(
        md, n_layers=NL, max_pos=384, users=32, ring_blocks=S // 64 + 8, log=lambda s: print(s, flush=True)
    )
    rec("env", layers=NL, expert_op=os.environ.get("MIMO_EXPERT_OP", "mm"), cache=model.wc.stats)

    if E("CHAIN_TF", "1") == "1":
        for L in range(NL):
            lyr = model.layers[L]
            d, f = dumps[L]["prefill"], d32[L]["prefill"]
            h = model.upload_rows(row_major(d["h_in"]))
            taps = {}
            h2 = lyr.prefill(h, S, UPR, taps=taps)
            res = {"layer": L}
            dev = {k: rows_users(model.rows_to_host(taps[k])) for k in ("attn_in", "attn_out", "ffn_in", "ffn_out")}
            dev["h_out"] = rows_users(model.rows_to_host(h2))
            for k in ("attn_in", "attn_out", "ffn_in", "ffn_out", "h_out"):
                pb, pf, fl = (
                    per_user(dev[k], d[k].float()),
                    per_user(dev[k], f[k].float()),
                    per_user(d[k].float(), f[k].float()),
                )
                res[k] = dict(
                    vs_bf16=round(min(pb), 5),
                    vs_f32=round(min(pf), 5),
                    floor=round(min(fl), 5),
                    mean_vs_bf16=round(sum(pb) / 16, 5),
                )
            if L > 0:
                idx_dev = torch.stack(
                    [
                        ttnn.to_torch(ttnn.get_device_tensors(lyr.last_route[1])[r * 8]).reshape(UPR * S, 8).long()
                        for r in range(ROWS)
                    ]
                ).reshape(-1, 8)
                idx_ref = d["routing_idx"].reshape(-1, 8)
                same = (idx_dev.sort(-1).values == idx_ref.sort(-1).values).all(-1)
                gap = d["route_gap"].reshape(-1)
                flips = ~same
                res["routing"] = dict(
                    set_match=round(float(same.float().mean()), 5),
                    flips=int(flips.sum()),
                    flips_gap_lt_0p01=int((flips & (gap < 0.01)).sum()),
                    tokens=int(same.numel()),
                )
                fo, fr = dev["ffn_out"].reshape(-1, H), d["ffn_out"].float().reshape(-1, H)
                res["ffn_out_pcc_set_matched_tokens"] = round(pcc(fo[same], fr[same]), 5)
            rec("teacher_forced_prefill", **res)
            # FFN-only teacher forcing: dump ffn_in -> device router + experts (isolates the FFN from upstream attention noise)
            xin = model.upload_rows(row_major(d["ffn_in"]))
            fo = lyr.ffn(xin)
            fdev = rows_users(model.rows_to_host(fo))
            r2 = {"layer": L}
            fl = per_user(d["ffn_out"].float(), f["ffn_out"].float())
            r2["ffn_out_vs_bf16"] = round(min(per_user(fdev, d["ffn_out"].float())), 5)
            r2["ffn_out_vs_f32"] = round(min(per_user(fdev, f["ffn_out"].float())), 5)
            r2["floor"] = round(min(fl), 5)
            if L > 0:
                idx_dev = torch.stack(
                    [
                        ttnn.to_torch(ttnn.get_device_tensors(lyr.last_route[1])[r * 8]).reshape(UPR * S, 8).long()
                        for r in range(ROWS)
                    ]
                ).reshape(-1, 8)
                same = (idx_dev.sort(-1).values == d["routing_idx"].reshape(-1, 8).sort(-1).values).all(-1)
                gap = d["route_gap"].reshape(-1)
                r2["routing_on_dump_ffn_in"] = dict(
                    set_match=round(float(same.float().mean()), 5),
                    flips=int((~same).sum()),
                    flips_gap_lt_0p01=int(((~same) & (gap < 0.01)).sum()),
                )
            rec("teacher_forced_ffn_only", **r2)
            ttnn.deallocate(fo)
            ttnn.deallocate(xin)
            for k in taps:
                ttnn.deallocate(taps[k])
            ttnn.deallocate(h2)
            ttnn.deallocate(h)

            if E("CHAIN_DECODE", "0") == "1" and len(dumps[L]["decode"]):
                steps = []
                dio = lyr.attn.decode_io()
                for s_ in range(len(dumps[L]["decode"])):
                    di = dumps[L]["decode"][s_]
                    x = torch.zeros(ROWS, 32, H)
                    for b in range(16):
                        x[b // UPR, b % UPR] = di["dec_in"][b, 0].float()
                    lyr.attn.decode_set(dio, [S + s_] * 32)
                    dt = model.upload_rows(x)
                    o = lyr.decode(dt, dio["cur"], dio["idx"])
                    oh = model.rows_to_host(o)
                    pcs = [pcc(oh[b // UPR, b % UPR], di["dec_out"][b, 0].float()) for b in range(16)]
                    steps.append(dict(step=s_, pcc_min=round(min(pcs), 5), pcc_mean=round(sum(pcs) / 16, 5)))
                    ttnn.deallocate(o)
                    ttnn.deallocate(dt)
                rec("teacher_forced_decode", layer=L, steps=steps)

    if E("CHAIN_FREE", "1") == "1":
        h = model.embed_tokens(toks.reshape(ROWS, UPR * S))
        emb = rows_users(model.rows_to_host(h))
        ref_emb = dumps[0]["prefill"]["h_in"].float()
        rec("embedding", pcc_min=round(min(per_user(emb, ref_emb)), 5))
        curve = []
        for L in range(NL):
            lyr = model.layers[L]
            h2 = lyr.prefill(h, S, UPR)
            ttnn.deallocate(h)
            h = h2
            ho = rows_users(model.rows_to_host(h))
            d, f = dumps[L]["prefill"]["h_out"].float(), d32[L]["prefill"]["h_out"].float()
            pb, pf, fl = per_user(ho, d), per_user(ho, f), per_user(d, f)
            curve.append(
                dict(
                    layer=L,
                    vs_bf16=round(min(pb), 5),
                    vs_f32=round(min(pf), 5),
                    floor=round(min(fl), 5),
                    mean_vs_f32=round(sum(pf) / 16, 5),
                )
            )
            rec("free_chain_layer", **curve[-1])
        rec("free_chain_curve", curve=curve)
