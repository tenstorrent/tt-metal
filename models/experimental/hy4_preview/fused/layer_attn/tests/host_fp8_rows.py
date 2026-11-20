# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host only (no device is opened): why attention over the packed FP8 rows comes out 4 to 5 % below attention over
the unquantised rows, layers 0..5 of the checkpoint on the golden prompt (PLAN.md section 12a).

The pack is emulated in torch (BF16 input, 4 power-of-two scales per row, e4m3 with the packer's truncation toward
zero, BF16 RoPE part); the emulation reproduces the gains test_layer.py measured on the device's stored rows to four
digits (DEVICE_CTX, DEVICE_ROW below). Run:
    python models/experimental/hy4_preview/fused/layer_attn/tests/host_fp8_rows.py
"""
import statistics, sys
import torch
from models.experimental.hy4_preview.fused.sparse_mla.tests.test_attend import _checkpoint_reference
from models.experimental.hy4_preview.reference import hf_harness

POS = list(range(120, 128))
DEVICE_CTX = [
    0.9584,
    0.9561,
    0.9587,
    0.9524,
    0.9470,
    0.9548,
]  # mk8a_layer_acc_fp32: the reference on the device's stored rows
DEVICE_ROW = [0.9998, 0.9998, 1.0000, 0.9987, 0.9964, 0.9999]


def e4m3(v, mode):
    a = v.abs().double()
    e = torch.floor(torch.log2(a.clamp(min=2.0**-6)))
    step = torch.where(a < 2.0**-6, torch.full_like(a, 2.0**-9), 2.0 ** (e - 3))
    q = (torch.floor(a / step) if mode == "trunc" else torch.round(a / step)) * step
    return (torch.sign(v).double() * q.clamp(max=448.0)).float()


def store(rows, mode="trunc", bf16=True):
    """[n, 576] (k_pass 512 | k_rope 64) as the packed storage holds it: the BF16 input of the cast, 4 power-of-two
    scales per row, e4m3 (packer truncating toward zero, or to nearest), BF16 RoPE."""
    p = rows[:, :512]
    if bf16:
        p = p.to(torch.bfloat16).float()
    p = p.reshape(-1, 4, 128)
    amax = p.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    q = e4m3(p / scale, mode) * scale
    return torch.cat([q.reshape(-1, 512), rows[:, 512:].to(torch.bfloat16).float()], dim=1)


def gain(ref, dev):
    ref, dev = ref.double().flatten(), dev.double().flatten()
    return float((ref * dev).sum() / (ref * ref).sum())


def pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.double().flatten(), b.double().flatten()]))[0, 1])


def attend(ref, q_lat, score_rows, value_rows):
    logits = (q_lat.double() @ score_rows.double().t()) * ref.cfg.scaling
    sink = ref.attention.learnable_sink_param.detach().double().reshape(-1, 1)
    probs = torch.softmax(torch.cat([logits, sink], dim=-1), dim=-1)
    return probs[:, :-1] @ value_rows[:, :512].double(), probs[:, -1], torch.exp(logits - sink).sum(-1), logits


CASES = (
    ("stored scores, stored values", "ss"),
    ("stored scores, exact values", "se"),
    ("exact scores, stored values", "es"),
    ("scores FP8 part stored + RoPE exact, exact values", "pe"),
    ("scores FP8 part exact + RoPE stored, exact values", "re"),
    ("nearest-rounding storage, scores and values", "nn"),
)


def run(ref, q_lat, exact):
    """per case: ctx [64, 512], sink [64], rows' mass relative to the sink [64], logits [64, n]; plus 'ee'."""
    st, sn = store(exact, "trunc"), store(exact, "near")
    mix = lambda a, b: torch.cat([a[:, :512], b[:, 512:]], dim=1)
    pairs = dict(
        ee=(exact, exact),
        ss=(st, st),
        se=(st, exact),
        es=(exact, st),
        pe=(mix(st, exact), exact),
        re=(mix(exact, st), exact),
        nn=(sn, sn),
    )
    return {k: attend(ref, q_lat, *v) for k, v in pairs.items()}, st, sn


def summary(results):
    """results: list (users) of run() outputs -> per case the pooled numbers."""
    out = {}
    ctx_e = torch.stack([r["ee"][0] for r in results])  # [u, 64, 512]
    sink_e = torch.stack([r["ee"][1] for r in results])
    for _, k in CASES:
        ctx = torch.stack([r[k][0] for r in results])
        sink = torch.stack([r[k][1] for r in results])
        mass = torch.stack([r[k][2] / r["ee"][2] for r in results])
        dl = torch.cat([(r[k][3] - r["ee"][3]).flatten() for r in results])
        heads = [gain(ctx_e[:, h], ctx[:, h]) for h in range(64)]
        users = [gain(ctx_e[u], ctx[u]) for u in range(ctx.shape[0])]
        out[k] = dict(
            gain=gain(ctx_e, ctx),
            pcc=pcc(ctx_e, ctx),
            sink=float(sink.mean()),
            dsink=float((sink - sink_e).abs().max()),
            mass=float(mass.log().mean().exp()),
            mass_lo=float(mass.min()),
            mass_hi=float(mass.max()),
            dl_mean=float(dl.mean()),
            dl_rms=float(dl.pow(2).mean().sqrt()),
            dl_max=float(dl.abs().max()),
            h_lo=min(heads),
            h_med=statistics.median(heads),
            h_hi=max(heads),
            u_lo=min(users),
            u_hi=max(users),
        )
    return out, sink_e


def show(out, keys=None, pad="   "):
    for name, k in CASES:
        if keys and k not in keys:
            continue
        v = out[k]
        print(
            f"{pad}{name}: ctx gain {v['gain']:.4f} (heads {v['h_lo']:.4f} / {v['h_med']:.4f} / {v['h_hi']:.4f}; users {v['u_lo']:.4f} .. {v['u_hi']:.4f}) pcc {v['pcc']:.6f}; "
            f"sink share {v['sink']:.4f} (largest change in a head {v['dsink']:.4f}); rows' mass x{v['mass']:.4f} ({v['mass_lo']:.3f} .. {v['mass_hi']:.3f}); "
            f"logit error mean {v['dl_mean']:+.4f} rms {v['dl_rms']:.4f} max {v['dl_max']:.3f}"
        )


def main():
    gold = hf_harness.find_golden()
    for layer in range(6):
        ref = _checkpoint_reference(layer)
        hf = gold["layers"][layer]
        with torch.no_grad():
            toks, lats = [], []
            for p in POS:
                x_all = hf["attn_in"][: p + 1].float()
                cache = ref.prefill_cache(x_all[:p])
                token = ref.new_token(x_all[p : p + 1], p)
                cache.append(token.latent)
                toks.append(ref.latent_queries(token).float())
                lats.append(cache.latent.float())
            full = [run(ref, q, e) for q, e in zip(toks, lats)]
            out, sink_e = summary([r[0] for r in full])
            # the stored rows themselves
            p_e = torch.cat([e[:, :512] for e in lats])
            p_b = p_e.to(torch.bfloat16).float()
            p_s = torch.cat([r[1][:, :512] for r in full])
            p_n = torch.cat([r[2][:, :512] for r in full])
            p_f = torch.cat([store(e, "trunc", bf16=False)[:, :512] for e in lats])
            nz = p_e.abs() > 0
            rel = lambda s: ((s.abs() - p_e.abs()) / p_e.abs().clamp(min=1e-20))[nz]
            row_e, row_s = torch.cat(lats), torch.cat([r[1] for r in full])
            # the 4 scales: the stored magnitude against the block's largest
            blk = p_b.reshape(-1, 4, 128).abs().amax(-1)
            sc = torch.pow(2.0, torch.ceil(torch.log2(blk.clamp(min=1e-4) / 448.0)))
            norm = blk / sc
            # the FP8 part of the logit against the RoPE part, exact rows, and the slope of the stored FP8 part
            lp_e = torch.cat(
                [
                    ((q[:, :512].double() @ e[:, :512].double().t()) * ref.cfg.scaling).flatten()
                    for q, e in zip(toks, lats)
                ]
            )
            lr_e = torch.cat(
                [
                    ((q[:, 512:].double() @ e[:, 512:].double().t()) * ref.cfg.scaling).flatten()
                    for q, e in zip(toks, lats)
                ]
            )
            lp_s = torch.cat(
                [
                    ((q[:, :512].double() @ r[1][:, :512].double().t()) * ref.cfg.scaling).flatten()
                    for q, r in zip(toks, full)
                ]
            )
            print(
                f"== layer {layer}: golden prompt, users at positions {POS[0]}..{POS[-1]} ({POS[0] + 1}..{POS[-1] + 1} selected rows), 64 heads"
            )
            print(
                f"   stored FP8 part (BF16 in, truncating packer): gain {gain(p_e, p_s):.4f} pcc {pcc(p_e, p_s):.6f}; (|stored| - |exact|) / |exact|: mean {rel(p_s).mean():+.4f} "
                f"rms {rel(p_s).pow(2).mean().sqrt():.4f} min {rel(p_s).min():+.4f} max {rel(p_s).max():+.4f}; share of elements with |stored| > |exact| {float((rel(p_s) > 0).float().mean()):.4f}"
            )
            print(
                f"   the same from an FP32 input: gain {gain(p_e, p_f):.4f} mean {rel(p_f).mean():+.4f}; with a nearest-rounding packer: gain {gain(p_e, p_n):.4f} mean {rel(p_n).mean():+.4f} "
                f"rms {rel(p_n).pow(2).mean().sqrt():.4f}; BF16 alone: gain {gain(p_e, p_b):.5f}"
            )
            print(
                f"   scales: block maximum / scale (448 is the format's largest): min {norm.min():.1f} median {norm.median():.1f} max {norm.max():.1f}; "
                f"stored values that are zero {float((p_s == 0).float().mean()):.5f}, subnormal {float(((p_s != 0) & ((p_s.reshape(-1, 4, 128) / sc[..., None]).abs() < 2.0**-6).reshape(-1, 512)).float().mean()):.5f}"
            )
            print(
                f"   whole row (576): gain {gain(row_e, row_s):.4f} (device {DEVICE_ROW[layer]:.4f}) pcc {pcc(row_e, row_s):.6f}; rms per element FP8 part {p_e.pow(2).mean().sqrt():.3f}, RoPE part "
                f"{row_e[:, 512:].pow(2).mean().sqrt():.3f}; the FP8 part's share of the row's energy {float(p_e.pow(2).sum() / row_e.pow(2).sum()):.4f}"
            )
            print(
                f"   logits (exact rows): std of the FP8 part {lp_e.std():.3f}, of the RoPE part {lr_e.std():.3f}; stored FP8 part against exact: gain {gain(lp_e, lp_s):.4f}, "
                f"error mean {(lp_s - lp_e).mean():+.4f} rms {(lp_s - lp_e).pow(2).mean().sqrt():.4f}"
            )
            print(
                f"   sink share, exact rows: mean {sink_e.mean():.4f}; heads (mean over users) min {sink_e.mean(0).min():.4f} median {sink_e.mean(0).median():.4f} max {sink_e.mean(0).max():.4f}"
            )
            print(
                f"   device-measured ctx gain of the stored rows {DEVICE_CTX[layer]:.4f}; this emulation {out['ss']['gain']:.4f}"
            )
            show(out)
            # dependence on the number of selected rows: the last n rows of the golden prompt
            for n in (4, 16, 64):
                sub, _ = summary([run(ref, q, e[-n:])[0] for q, e in zip(toks, lats)])
                print(f"   last {n} rows only:")
                show(sub, keys=("ss", "se", "es"), pad="      ")
            for n in (512, 2048):
                g = torch.Generator().manual_seed(layer)
                x = torch.randn(n, ref.cfg.hidden_size, generator=g)
                res = []
                for u in range(2):
                    cache = ref.prefill_cache(x)
                    token = ref.new_token(torch.randn(1, ref.cfg.hidden_size, generator=g), n)
                    cache.append(token.latent)
                    res.append(run(ref, ref.latent_queries(token).float(), cache.latent.float())[0])
                big, _ = summary(res)
                print(f"   {n + 1} rows of random pre-normed inputs (2 users):")
                show(big, keys=("ss", "se", "es", "nn"), pad="      ")
        sys.stdout.flush()


main()
