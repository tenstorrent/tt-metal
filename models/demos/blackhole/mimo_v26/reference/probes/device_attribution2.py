"""Second attribution pass. (a2) MoE with forced DEVICE routing on the DEVICE ffn_in using bfp8-quantised expert weights (device layout: W.T tiles, 16-groups along
the output dim; fp4+e8m0 values are NOT exact in bfp8 because scales differ per row inside a group) with bf16 activations vs device ffn_out.
(b2) attention variants on the DEVICE attn_in (bf16 act, bfp8 qkv/o) vs device attn_out: baseline, sink variants (x/sqrt(192), x sqrt(192), none), window 127/129,
V scale 1.0, rope theta x2 etc. - a variant that matches the device much better than the baseline points at the cause; plus the per-position / per-user error profile."""
import argparse
import json
import os
import time

os.environ.setdefault("MIMO_CKPT", "/mnt/tt-data/MiMo-V2.6-Flash-RL")
import torch
import torch.nn.functional as F

from models.demos.blackhole.mimo_v26.reference import loader as Ld
from models.demos.blackhole.mimo_v26.reference import ref_layer as R
from models.demos.blackhole.mimo_v26.reference.bfp import quant_weight_bfp8
from models.demos.blackhole.mimo_v26.reference.probes.device_attribution import moe_forced, rel
from models.demos.blackhole.mimo_v26.reference.ref_prefill_dump import parse_layers

torch.set_num_threads(32)


def attention_var(x, sd, i, sink_mul=1.0, window=128, v_scale=0.707, theta_mul=1.0, sink=True, interleaved=False):
    swa = R.is_swa(i)
    nkv = 8 if swa else 4
    B, S, _ = x.shape
    qkv = x @ sd["self_attn.qkv_proj.weight"].T
    q, k, v = qkv.split([64 * 192, nkv * 192, nkv * 128], -1)
    q = q.view(B, S, 64, 192).transpose(1, 2)
    k = k.view(B, S, nkv, 192).transpose(1, 2)
    v = v.view(B, S, nkv, 128).transpose(1, 2) * v_scale
    cos, sin = R.rope_tables(S, (1e4 if swa else 1e7) * theta_mul)
    if interleaved:

        def rot(t):
            r = t[..., :64]
            re, im = r[..., 0::2], r[..., 1::2]
            c, s_ = cos[:, :32].to(t.dtype), sin[:, :32].to(t.dtype)
            o = torch.stack([re * c - im * s_, re * s_ + im * c], -1).flatten(-2)
            return torch.cat([o, t[..., 64:]], -1)

        q, k = rot(q), rot(k)
    else:
        q, k = R.apply_rope_partial(q, cos, sin), R.apply_rope_partial(k, cos, sin)
    rep = 64 // nkv
    ke, ve = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    s = (q @ ke.transpose(2, 3)).float() * 192**-0.5
    ii = torch.arange(S)[:, None]
    jj = torch.arange(S)[None]
    m = jj <= ii
    if swa:
        m &= jj > ii - window
    s = s.masked_fill(~m, float("-inf"))
    use_sink = swa and sink
    if use_sink:
        s = torch.cat(
            [s, (sd["self_attn.attention_sink_bias"].float() * sink_mul).view(1, 64, 1, 1).expand(B, 64, S, 1)], -1
        )
    p = F.softmax(s - s.max(-1, keepdim=True).values, -1)
    if use_sink:
        p = p[..., :-1]
    o = (p.to(x.dtype) @ ve).transpose(1, 2).reshape(B, S, 64 * 128)
    return o @ sd["self_attn.o_proj.weight"].T


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dev")
    ap.add_argument("out")
    ap.add_argument("--layers", default="16,41,6-8,17,47")
    ap.add_argument("--skip-variants", action="store_true")
    a = ap.parse_args()
    ck = Ld.Ckpt()
    done = {json.loads(l)["L"] for l in open(a.out)} if os.path.exists(a.out) else set()
    layers = []
    for part in a.layers.split(","):
        layers += parse_layers(part)
    for L in layers:
        if L in done:
            continue
        t0 = time.time()
        d = torch.load(f"{a.dev}/layer_{L}.pt")["prefill"]
        sd32 = Ld.load_layer(L, torch.float32, ck)
        sd16 = {
            k: (v.to(torch.bfloat16) if v.is_floating_point() and "e_score" not in k else v) for k, v in sd32.items()
        }
        row = {"L": L, "type": "swa" if R.is_swa(L) else "global"}
        fin = d["ffn_in"].reshape(-1, 4096)
        di, dw = d["routing_idx"], d["routing_w"].float()
        dev_f = d["ffn_out"].reshape(-1, 4096)
        f32 = moe_forced(fin.float(), sd32, di, dw)
        f16 = moe_forced(fin, sd16, di, dw)
        sdq = dict(sd16)
        for k in list(sd16):
            if "experts." in k:
                sdq[k] = quant_weight_bfp8(sd32[k]).to(torch.bfloat16)
        fq = moe_forced(fin, sdq, di, dw)
        row["moe_forced"] = {
            "floor_bf16_vs_fp32": rel(f16, f32),
            "dev_vs_fp32": rel(dev_f, f32),
            "dev_vs_bf16": rel(dev_f, f16),
            "dev_vs_bfp8exp_bf16act": rel(dev_f, fq),
            "bfp8exp_bf16act_vs_fp32": rel(fq, f32),
            "ratio_dev_fp32_over_floor": rel(dev_f, f32) / rel(f16, f32),
            "ratio_dev_bfp8exp_over_floor": rel(dev_f, fq) / rel(f16, f32),
        }
        del sdq, f16, fq
        ain = d["attn_in"]
        dev = d["attn_out"]
        sdw = dict(sd16)
        for k in ("self_attn.qkv_proj.weight", "self_attn.o_proj.weight"):
            sdw[k] = quant_weight_bfp8(sd32[k]).to(torch.bfloat16)
        base = attention_var(ain, sdw, L)
        row["attn_variants_rel_dev_vs"] = {"baseline_bfp8w": rel(dev, base)}
        if not a.skip_variants:
            vs = {
                "v_scale_1.0": dict(v_scale=1.0),
                "theta_x2": dict(theta_mul=2.0),
                "theta_div2": dict(theta_mul=0.5),
                "rope_interleaved": dict(interleaved=True),
            }
            if R.is_swa(L):
                vs.update(
                    {
                        "sink_div_sqrt192": dict(sink_mul=192**-0.5),
                        "sink_x_sqrt192": dict(sink_mul=192**0.5),
                        "no_sink": dict(sink=False),
                        "window_127": dict(window=127),
                        "window_129": dict(window=129),
                    }
                )
            for n, kw in vs.items():
                row["attn_variants_rel_dev_vs"][n] = rel(dev, attention_var(ain, sdw, L, **kw))
        err = (dev.float() - base.float()).norm(dim=-1) / base.float().norm(dim=-1)  # [B,S]
        row["attn_err_profile"] = {
            "by_pos_bucket_median": [err[:, i : i + 16].median().item() for i in range(0, 128, 16)],
            "max_tok": err.max().item(),
            "median": err.median().item(),
            "frac_tokens_gt_3x_median": (err > 3 * err.median()).float().mean().item(),
            "by_user_median": [err[b].median().item() for b in range(err.shape[0])],
        }
        row["secs"] = round(time.time() - t0)
        open(a.out, "a").write(json.dumps(row) + "\n")
        m = row["moe_forced"]
        print(
            L,
            row["type"],
            "MOE forced dev/fp32 %.4f dev/bfp8exp %.4f floor %.4f  ratio %.1f->%.1f"
            % (
                m["dev_vs_fp32"],
                m["dev_vs_bfp8exp_bf16act"],
                m["floor_bf16_vs_fp32"],
                m["ratio_dev_fp32_over_floor"],
                m["ratio_dev_bfp8exp_over_floor"],
            ),
            "| attn variants",
            {k: round(v, 4) for k, v in row["attn_variants_rel_dev_vs"].items()},
            flush=True,
        )
        del sd32, sd16, sdw


if __name__ == "__main__":
    main()
