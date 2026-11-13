# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""attn-bfp4 task, CPU part: weight round trip bf16 -> bfp8_b / bfp4_b (host pack by ttnn, TILE layout, in the DEVICE layout of
tt/hy4_attn_weights.py) for layers 0..3, plus activation-level output error on the layer-attn golden activations
(/mnt/tt-data/ssinghal/hy4-logs/layer-attn/golden/L{0,1,2}.pt) and on the layer-moe activations.

Run (slot 1 of host .44, ttnn only used for the host-side pack, no device open):
  eval $(bin/slot_env.sh 1 10.82.97.44)
  PYTHONPATH=<wt>:/mnt/tt-data/ssinghal/wt/hy4_ref python tools/abf4_cpu.py OUT.json [table|block]
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

GOLD = "/mnt/tt-data/ssinghal/hy4-logs/layer-attn/golden"
MOEA = "/mnt/tt-data/ssinghal/hy4-logs/program/layer-moe/acts"
ATTN_DUMP = "/mnt/tt-data/ssinghal/hy4-logs/program/layer-attn/dump_L1_chunk0.pt"
FMTS = ("bfp8", "bfp4")


def pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def rel(a, b):
    return float((a.double() - b.double()).norm() / (b.double().norm() + 1e-30))


def rt(t, fmt):
    """Round trip of a device-layout tensor [..., K, N] (fp32/bf16) through the ttnn host packer; returns fp32 same shape."""
    import ttnn

    dt = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "bf16": ttnn.bfloat16}[fmt]
    x = t.to(torch.bfloat16)
    sh = x.shape
    x4 = x.reshape(-1, *sh[-2:])[None] if x.dim() != 4 else x
    q = ttnn.to_torch(ttnn.from_torch(x4, dtype=dt, layout=ttnn.TILE_LAYOUT)).float()
    return q.reshape(sh)


# device layout <-> hf layout, per attention/indexer tensor (mirrors tt/hy4_attn_weights.py)
def dev_attn(cfg, sd):
    from models.demos.blackhole.hy4_preview.tt.hy4_attn_weights import build_attn_host_weights

    class C:
        pass

    c = C()
    for k in (
        "num_attention_heads qk_nope_head_dim qk_rope_head_dim v_head_dim kv_lora_rank hidden_size q_lora_rank index_n_heads index_head_dim"
    ).split():
        setattr(c, k, getattr(cfg, k))
    return build_attn_host_weights(c, sd)


def hf_from_dev(name, t, cfg):
    if name in ("q_a", "q_b", "kv_a", "gate", "o_proj", "idx_wq_b", "idx_wk"):
        return t.reshape(t.shape[-2], t.shape[-1]).T.contiguous()
    raise KeyError(name)


HF = {
    "q_a": "self_attn.q_a_proj.weight",
    "q_b": "self_attn.q_b_proj.weight",
    "kv_a": "self_attn.kv_a_proj_with_mqa.weight",
    "gate": "self_attn.linear_gate.weight",
    "o_proj": "self_attn.o_proj.weight",
    "idx_wq_b": "self_attn.indexer.wq_b.weight",
    "idx_wk": "self_attn.indexer.wk.weight",
}


def quant_sd(cfg, sd, names, fmt, dev):
    """Copy of sd with the given tensor names replaced by their fmt round trip (dequantised fp32, hf layout)."""
    out = dict(sd)
    for n in names:
        if n in HF:
            out[HF[n]] = hf_from_dev(n, rt(dev[n].tensor, fmt), cfg)
        elif n == "kv_b":
            H, nope, vd, kvl = cfg.n_heads, cfg.nope, cfg.v_dim, cfg.kv_lora
            uk = rt(dev["w_uk"].tensor, fmt)  # [1,H,192,512]
            uv = rt(dev["w_uv_t"].tensor, fmt)  # [1,H,512,256]
            kvb = torch.cat([uk[0], uv[0].transpose(1, 2)], 1)  # [H,448,512]
            out["self_attn.kv_b_proj.weight"] = kvb.reshape(H * (nope + vd), kvl).contiguous()
        elif n == "idx_wp":
            ni, di = cfg.idx_heads, cfg.idx_dim
            q = rt(dev["idx_wp"].tensor, fmt).reshape(cfg.hidden, ni)
            out["self_attn.indexer.weights_proj.weight"] = (q / ((ni**-0.5) * (di**-0.5))).T.contiguous()
        else:
            raise KeyError(n)
    return out


def main():
    outp = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "table"
    torch.set_num_threads(int(os.environ.get("HY4_THREADS", "16")))
    import models.demos.blackhole.hy4_preview as _p

    _p.__path__.append("/mnt/tt-data/ssinghal/wt/hy4_ref/models/demos/blackhole/hy4_preview")
    from models.demos.blackhole.hy4_preview.reference import oracle as O
    from models.demos.blackhole.hy4_preview.reference.oracle import Conv, Hy4Oracle

    o = Hy4Oracle()
    cfg = o.ck.cfg
    from models.demos.blackhole.hy4_preview.config import Hy4Config

    hcfg = Hy4Config.from_checkpoint("/mnt/tt-data/Hy4-Preview")
    conv = Conv()
    res = json.load(open(outp)) if os.path.exists(outp) else {}

    def save():
        json.dump(res, open(outp, "w"), indent=1)

    t0 = time.time()
    NT = int(os.environ.get("ABF4_TOK", "1024"))
    LAYERS = [int(x) for x in os.environ.get("ABF4_LAYERS", "0,1,2,3").split(",")]
    GD = os.environ.get("ABF4_GOLD", GOLD)
    for L in LAYERS:
        sd = o.ck.layer_state(L)
        dev = dev_attn(hcfg, sd)
        names = ["q_a", "q_b", "kv_a", "w_uk", "w_uv_t", "gate", "o_proj"] + (
            ["idx_wq_b", "idx_wk", "idx_wp"] if "self_attn.indexer.wq_b.weight" in sd else []
        )
        if mode == "table":
            # ---- weight-only round trip, device layout
            for n in names:
                w = dev[n].tensor
                for f in FMTS:
                    q = rt(w, f)
                    res[f"L{L}.{n}.{f}.w"] = [pcc(q, w), rel(q, w)]
                print(
                    f"L{L} {n} {tuple(w.shape)} "
                    + " ".join(
                        f"{f}: pcc {res[f'L{L}.{n}.{f}.w'][0]:.5f} rel {res[f'L{L}.{n}.{f}.w'][1]:.4f}" for f in FMTS
                    ),
                    f"{time.time()-t0:.0f}s",
                    flush=True,
                )
            mlp = (
                [
                    ("mlp.gate_proj.weight", "dense_gate"),
                    ("mlp.up_proj.weight", "dense_up"),
                    ("mlp.down_proj.weight", "dense_down"),
                ]
                if L == 0
                else [
                    ("mlp.shared_experts.gate_proj.weight", "sh_gate"),
                    ("mlp.shared_experts.up_proj.weight", "sh_up"),
                    ("mlp.shared_experts.down_proj.weight", "sh_down"),
                ]
            )
            for k, n in mlp:
                w = sd[k].T.contiguous()
                for f in FMTS:
                    q = rt(w, f)
                    res[f"L{L}.{n}.{f}.w"] = [pcc(q, w), rel(q, w)]
                print(
                    f"L{L} {n} {tuple(w.shape)} "
                    + " ".join(
                        f"{f}: pcc {res[f'L{L}.{n}.{f}.w'][0]:.5f} rel {res[f'L{L}.{n}.{f}.w'][1]:.4f}" for f in FMTS
                    ),
                    f"{time.time()-t0:.0f}s",
                    flush=True,
                )
            save()
            # ---- projection output error on real activations (layers 0..2 attention, mlp from layer-moe acts)
            if L <= 2:
                G = torch.load(f"{GOLD}/L{L}.pt", map_location="cpu")
                h, qa, o_, gate_ = G["h"][:NT], G["qa"][:NT], G["o"][:NT], G["gate"][:NT]
                q_nope = G["q"][:NT, :, : cfg.nope]  # [T,64,192]
                ins = {
                    "q_a": (h, "q_a"),
                    "q_b": (qa, "q_b"),
                    "kv_a": (h, "kv_a"),
                    "gate": (h, "gate"),
                    "o_proj": (o_ * gate_, "o_proj"),
                    "idx_wq_b": (qa, "idx_wq_b"),
                    "idx_wk": (h, "idx_wk"),
                    "idx_wp": (h, "idx_wp"),
                }
                for n, (x, _) in ins.items():
                    if n not in names:
                        continue
                    w = dev[n].tensor.reshape(dev[n].tensor.shape[-2], -1)
                    y0 = x @ w
                    for f in FMTS:
                        y = x @ rt(dev[n].tensor, f).reshape(w.shape)
                        res[f"L{L}.{n}.{f}.y"] = [pcc(y, y0), rel(y, y0)]
                uk = dev["w_uk"].tensor[0]  # [64,192,512]
                y0 = torch.einsum("thd,hdk->thk", q_nope, uk)
                for f in FMTS:
                    y = torch.einsum("thd,hdk->thk", q_nope, rt(dev["w_uk"].tensor, f)[0])
                    res[f"L{L}.w_uk.{f}.y"] = [pcc(y, y0), rel(y, y0)]
                if L == 1:
                    ol = torch.load(ATTN_DUMP, map_location="cpu")["o_lat_c0"][0]  # [8,512,512] heads 0..7
                    uv = dev["w_uv_t"].tensor[0, :8]
                    y0 = torch.einsum("hsk,hkv->hsv", ol, uv)
                    for f in FMTS:
                        y = torch.einsum("hsk,hkv->hsv", ol, rt(dev["w_uv_t"].tensor, f)[0, :8])
                        res[f"L{L}.w_uv_t.{f}.y"] = [pcc(y, y0), rel(y, y0)]
                print(f"L{L} activations done {time.time()-t0:.0f}s", flush=True)
                del G
            # shared / dense MLP outputs
            A = None
            if L == 0:
                A = torch.load(f"{MOEA}/acts_l0_l1.pt", map_location="cpu")["dense"]
                x = torch.cat([a[0][0] for a in A])[:NT]
                for f in FMTS:
                    s2 = dict(sd)
                    for k in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
                        s2[k] = rt(sd[k].T.contiguous(), f).T.contiguous()
                    y0 = O.dense_mlp(x, sd, cfg, conv)
                    y = O.dense_mlp(x, s2, cfg, conv)
                    res[f"L0.dense_mlp.{f}.y"] = [pcc(y, y0), rel(y, y0)]
            if L == 1:
                A = torch.load(f"{MOEA}/acts_l0_l1.pt", map_location="cpu")["moe"]
                x = torch.cat([a[0][0] for a in A])[:NT]
                for f in FMTS:
                    s2 = dict(sd)
                    for k in (
                        "mlp.shared_experts.gate_proj.weight",
                        "mlp.shared_experts.up_proj.weight",
                        "mlp.shared_experts.down_proj.weight",
                    ):
                        s2[k] = rt(sd[k].T.contiguous(), f).T.contiguous()
                    y0 = O.shared_expert(x, sd, cfg, conv)
                    y = O.shared_expert(x, s2, cfg, conv)
                    res[f"L1.shared_mlp.{f}.y"] = [pcc(y, y0), rel(y, y0)]
            save()
        else:
            # ---- full attention block (oracle attention with dequantised weights), layers 0..2, S=3072 with the oracle's own indexer
            G = torch.load(f"{GD}/L{L}.pt", map_location="cpu")
            h = G["h"][None]
            S = h.shape[1]
            cos_i, sin_i = O.rope_cos_sin(S, cfg.rope, cfg.rope_theta)
            full = "self_attn.indexer.wq_b.weight" in sd
            ids0 = G["idx"][None] if G["idx"] is not None else None
            if ids0 is None and S > cfg.idx_topk:
                raise SystemExit("need idx")
            qa0 = O.rms_norm(
                F.linear(h, sd["self_attn.q_a_proj.weight"]), sd["self_attn.q_a_layernorm.weight"], cfg.norm_eps
            )
            attn_names = ["q_a", "q_b", "kv_a", "kv_b", "gate", "o_proj"]
            idx_names = ["idx_wq_b", "idx_wk", "idx_wp"] if (full and ids0 is not None) else []
            if L == 2:
                ids_src = ids0  # shared layer: indices come from layer 1
            y0 = G["out"][None]
            if os.environ.get("ABF4_REF_CHECK"):
                yy = O.attn_block(h, h, sd, cfg, conv, ids0)
                print("oracle self-check vs golden out pcc", pcc(yy, y0), flush=True)
            variants = {
                "bfp8_all": ("bfp8", attn_names + idx_names),
                "bfp4_all": ("bfp4", attn_names + idx_names),
                "bfp4_attn_idx8": ("bfp4", attn_names),
            }
            variants["bfp4_gate_qa_qb"] = ("bfp4", ["gate", "q_a", "q_b"])
            variants["bfp4_gate_qa_qb_kva"] = ("bfp4", ["gate", "q_a", "q_b", "kv_a"])
            variants["bfp4_all_but_oproj_kvb"] = ("bfp4", ["gate", "q_a", "q_b", "kv_a"] + idx_names)
            for n in attn_names + idx_names:
                variants[f"bfp4_only_{n}"] = ("bfp4", [n])
            for vn, (f, nm) in variants.items():
                if res.get(f"L{L}.block.{vn}"):
                    continue
                s2 = quant_sd(cfg, sd, nm, f, dev)
                if full and any(x.startswith("idx") for x in nm):
                    ids = O.indexer_topk(h, qa0, s2, cfg, conv, cos_i, sin_i)
                    ov = float(
                        sum(
                            len(set(a.tolist()) & set(b.tolist())) / max(1, len(set(b.tolist()) - {-1}))
                            for a, b in zip(ids[0, 2048::64], ids0[0, 2048::64])
                        )
                        / len(ids[0, 2048::64])
                    )
                else:
                    ids, ov = ids0, 1.0
                y = O.attn_block(h, h, s2, cfg, conv, ids)
                res[f"L{L}.block.{vn}"] = [pcc(y, y0), rel(y, y0), ov]
                print(
                    f"L{L} block {vn}: pcc {res[f'L{L}.block.{vn}'][0]:.5f} rel {res[f'L{L}.block.{vn}'][1]:.4f} idx-overlap {ov:.4f} {time.time()-t0:.0f}s",
                    flush=True,
                )
                save()
    save()


if __name__ == "__main__":
    main()
