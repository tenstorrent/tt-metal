# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""MoE-block output PCC per layer with experts at BFP4 / BFP8 vs the bf16 dequant (CPU, no ttnn).

    python -m models.demos.inkling2_mini.convert.moe_pcc --layers 8 [--out moe_pcc.json]

Uses transformers' Inkling2 router and SoftcappedSwiGLU; the expert loop mirrors ``Inkling2Experts.forward``.
Input: ``tokens`` random rows, RMS-normed and scaled by the layer's ``mlp_norm`` (the MoE input in HF). The
router is fp32 in both, so the same experts are picked and the PCC is the expert-weight error alone.
BFP round trip is emulated in torch (``bfp_roundtrip``), matched against ttnn in ``test_dequant.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from models.demos.inkling2_mini.convert.source import Checkpoint, deinterleave_gate_up, dequant_experts

P = "model.llm."
MANTISSA_BITS = {"BFP4": 3, "BFP8": 7}  # incl. the explicit leading 1, ttnn bfloat4_b / bfloat8_b


def bfp_roundtrip(w: torch.Tensor, dtype: str) -> torch.Tensor:
    """Host ttnn ``from_torch(dtype, TILE)`` -> ``to_torch`` of ``w`` [..., K, N]: one shared exponent (the max) per
    16 consecutive values along N (a tile face row), mantissa round-to-nearest-even, saturating."""
    mbits = MANTISSA_BITS[dtype]
    b = w.float().unflatten(-1, (-1, 16))
    exp = torch.frexp(b.abs().amax(-1, keepdim=True))[1] - 1  # floor(log2(max))
    step = torch.ldexp(torch.ones_like(exp, dtype=torch.float32), exp - (mbits - 1))
    q = torch.round(b.abs() / step).clamp(max=2**mbits - 1) * step
    return (q * b.sign()).flatten(-2)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


def layer_experts(ck: Checkpoint, layer: int, start: int, end: int):
    """bf16 dequant [e, 2I, K] (gate rows first) and [e, H, I], torch [out, in] per expert."""
    out = []
    for name in ("w13", "w2"):
        base = f"{P}layers.{layer}.mlp.experts.{name}_weight"
        nvfp4 = ck.has(base + "_amax")
        amax = ck.get(base + "_amax")[start:end] if nvfp4 else None
        den = ck.get(base + "_outer_scale_denominator") if nvfp4 else None
        t = dequant_experts(ck.experts(base, start, end), ck.experts(base + "_scale", start, end), amax, den)
        out.append(deinterleave_gate_up(t) if name == "w13" else t)
    return out


def moe_pcc(ck: Checkpoint, text_config, layer: int, tokens: int = 512, seed: int = 0, mixed: bool = False) -> dict:
    """MoE-block output PCC vs bf16 dequant (A) with all experts BFP4 and all BFP8. With ``mixed``, also
    gate_up/down at different dtypes, and the curve with the top-k most sensitive experts at BFP8 (rest BFP4).
    Sensitivity = squared BFP4 output error of that expert summed over the tokens routed to it (router weight
    included), so frequently picked experts weigh more."""
    from transformers.models.inkling2.modeling_inkling2 import Inkling2SoftcappedSwiGLU, Inkling2TopKRouter

    p = f"{P}layers.{layer}."
    router = Inkling2TopKRouter(text_config)
    router.weight.data = ck.get(p + "mlp.gate.weight").float()
    router.e_score_correction_bias.data = ck.get(p + "mlp.gate.bias").float()
    act = Inkling2SoftcappedSwiGLU(text_config)

    x = torch.randn(tokens, text_config.hidden_size, generator=torch.Generator().manual_seed(seed))
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * ck.get(p + "mlp_norm.weight").float()
    x = x.bfloat16()
    with torch.no_grad():
        _, weights, idx = router(x)
    # (gate_up dtype, down dtype); "A" = bf16 dequant
    variants = [("A", "A"), ("BFP4", "BFP4"), ("BFP8", "BFP8")]
    if mixed:
        variants += [("BFP8", "BFP4"), ("BFP4", "BFP8")]
    out = {v: torch.zeros(tokens, x.shape[1]) for v in variants}
    sens, delta = {}, {}  # per expert: BFP4 error energy; (tok, BFP8 - BFP4 contribution)
    for s in range(0, text_config.n_routed_experts, 8):
        gate_up, down = layer_experts(ck, layer, s, s + 8)
        for j in range(gate_up.shape[0]):
            e = s + j
            tok, k = torch.where(idx == e)
            if not tok.numel():
                sens[e] = 0.0
                continue
            # round trip in the cached [in, out] layout, back to torch [out, in]
            gu = {"A": gate_up[j], **{d: bfp_roundtrip(gate_up[j].T, d).T.bfloat16() for d in MANTISSA_BITS}}
            dn = {"A": down[j], **{d: bfp_roundtrip(down[j].T, d).T.bfloat16() for d in MANTISSA_BITS}}
            contrib = {}
            for v in variants:
                h = F.linear(act(*F.linear(x[tok], gu[v[0]]).chunk(2, dim=-1)), dn[v[1]])
                contrib[v] = (h * weights[tok, k, None]).float()
                out[v].index_add_(0, tok, contrib[v])
            if mixed:
                c4, c8 = contrib[("BFP4", "BFP4")], contrib[("BFP8", "BFP8")]
                sens[e] = (c4 - contrib[("A", "A")]).square().sum().item()
                delta[e] = (tok, c8 - c4)
    ref = out[("A", "A")]
    name = {
        ("BFP4", "BFP4"): "bfp4",
        ("BFP8", "BFP8"): "bfp8",
        ("BFP8", "BFP4"): "gu8_dn4",
        ("BFP4", "BFP8"): "gu4_dn8",
    }
    r = {"layer": layer, "tokens": tokens, **{f"moe_pcc_{name[v]}": round(pcc(out[v], ref), 6) for v in variants[1:]}}
    if mixed:
        order = sorted(sens, key=sens.get, reverse=True)
        curve = {}
        for top in (8, 16, 32, 64):
            o = out[("BFP4", "BFP4")].clone()
            for e in order[:top]:
                if e in delta:  # experts no token picked have nothing to change
                    o.index_add_(0, *delta[e])
            curve[top] = round(pcc(o, ref), 6)
        counts = torch.bincount(idx.flatten(), minlength=text_config.n_routed_experts)
        r |= {
            "topk_bfp8_pcc": curve,
            "sensitive_experts": order[:32],
            "top32_share_of_bfp4_error": round(sum(sens[e] for e in order[:32]) / sum(sens.values()), 4),
            "top32_share_of_picks": round(counts[order[:32]].sum().item() / counts.sum().item(), 4),
        }
    return r


def main(argv=None):
    from transformers import AutoConfig

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--ckpt", type=Path, default=os.environ.get("INKLING_CKPT"), required=not os.environ.get("INKLING_CKPT")
    )
    ap.add_argument("--layers", default=",".join(map(str, range(20))))
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--out", type=Path, help="JSON lines, one per layer (appended)")
    ap.add_argument("--mixed", action="store_true", help="also gate_up/down mixes and the top-k expert curve")
    args = ap.parse_args(argv)
    ck = Checkpoint(args.ckpt)
    text_config = AutoConfig.from_pretrained(args.ckpt).get_text_config()
    torch.set_grad_enabled(False)
    for layer in map(int, args.layers.split(",")):
        t0 = time.time()
        r = moe_pcc(ck, text_config, layer, args.tokens, mixed=args.mixed) | {"seconds": round(time.time() - t0, 1)}
        print(json.dumps(r), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
