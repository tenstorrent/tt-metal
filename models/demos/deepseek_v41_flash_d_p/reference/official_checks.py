# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""R.5 evidence (CPU) for official_taps.yaml's `explained` compressor entry (DS41-38).

    PYTHONPATH=$PWD python -m models.demos.deepseek_v41_flash_d_p.reference.official_checks

Runs the hook official_taps (hooks.py) over the component golden's chunk, every golden layer, and for every layer with
a compressor recomputes the step from its tapped input with model.py's internal bf16 roundings emulated (Compressor /
Indexer / Attention of inference/model.py at the bf16 default dtype):
- ratio 1: wkv in bf16 on the bf16 x (`self.norm(self.wkv(x))`), latent bf16 after the norm;
- ratio > 1: wkv / wgate fp32 on x.float(), fp32 softmax pooling, `kv.to(dtype)` (bf16) before the norm, bf16 after;
- index keys: bf16 wk GEMM on the bf16 latent, bf16 k_norm, RoPE copied back into bf16, fp4 (e8m0 per 32) on that;
- compressed kv: RoPE copied back into bf16, fp4 (e4m3 scale per 16) on that.
The official output must equal it bit for bit (exit 1 otherwise). Also printed: the official output vs the fp32 step
(the reference's ds41_ref step on the same input), the excess the tap check sees.
"""

from __future__ import annotations

import os
import sys
import time

import torch


def bf(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16)


def compressor_bf16(ref, L: int, x: torch.Tensor, start: int) -> torch.Tensor:
    """The compressor step (output [S / ratio, head_dim + index_head_dim]: compressed kv | index k) with model.py's
    bf16 points emulated; `x` is the step's input (the attention norm output), `start` the chunk's first position."""
    from models.demos.deepseek_v41_flash_d_p.reference import ds41_ref as D
    from models.demos.deepseek_v41_flash_d_p.reference import kernels as K

    cfg, w = ref.cfg, ref.w[L]
    r = cfg.ratio(L)
    rot = ref._rot(L, (start // r + torch.arange(x.shape[0] // r)) * r)
    xb = bf(x)
    norm_w, eps = w["attn.compressor.norm.weight"], cfg.eps
    if r == 1:  # Compressor.forward ratio 1: bf16 wkv on bf16 x, RMSNorm -> bf16
        lat = bf(D.rms_norm(bf(K.linear(xb, bf(w["attn.compressor.wkv.weight"]))).float(), norm_w, eps))
    else:  # fp32 wkv / wgate on x.float(), fp32 pooling, kv.to(bf16) before the norm
        kv = K.linear(xb.float(), w["attn.compressor.wkv.weight"]).view(-1, r, cfg.head_dim)
        sc = K.linear(xb.float(), w["attn.compressor.wgate.weight"]).view(-1, r, cfg.head_dim)
        lat = bf(D.rms_norm(bf((kv * sc.softmax(dim=1)).sum(dim=1)).float(), norm_w, eps))
    kk = bf(K.linear(lat, bf(w["attn.indexer.wk.weight"])))  # Indexer.wk: bf16
    kk = bf(D.rms_norm(kk.float(), w["attn.indexer.k_norm.weight"], eps))
    kk = bf(K.fp4_quant_dequant(bf(D.rope(kk.float(), rot)).float(), 32))
    ckv = bf(K.fp4_quant_dequant(bf(D.rope(lat.float(), rot)).float(), 16, e4m3_scale=True))
    return torch.cat([ckv.float(), kk.float()], -1)


def check_compressor(s, taps=None) -> dict:
    """{layer: (equal fraction official vs emulated, rel official vs fp32 step)} for every tapped compressor."""
    from models.demos.common.bringup.testing import component_checks as CC
    from models.demos.common.bringup.testing.harness import component_golden, reference_ctx

    g, c = component_golden(s)
    if taps is None:
        taps = s.hooks().official_taps(s, list(g.layers), g.tokens(), c, g.chunk)  # sources need every layer
    layers = [L for L in taps if "compressor" in taps[L]]
    ref = s.hooks().reference(s, layers=layers, dtype=torch.float32)
    start = c * g.chunk
    out = {}
    for L in layers:
        t = taps[L]["compressor"]
        x, off = t["inputs"][0].float(), t["output"].float()
        em = compressor_bf16(ref, L, x, start)
        y = ref.component(L, "compressor")(reference_ctx(ref, L, g, c, s), x.clone())
        eq = (off == em).float().mean().item() if off.shape == em.shape else 0.0
        out[L] = (eq, CC.rel(off, y.reshape(off.shape)))
        print(
            f"L{L:02d} compressor (ratio {ref.cfg.ratio(L)}, chunk {c}, {off.shape[0]} rows): official vs the "
            f"bf16-emulated step equal {eq:.6f}; official vs the fp32 step rel {out[L][1]:.4f}",
            flush=True,
        )
    return out


def main():
    from models.demos.common.bringup.core.spec import Spec

    s = Spec.load(os.environ["BRINGUP_SPEC"])
    t0 = time.time()
    res = check_compressor(s)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    bad = [L for L, (eq, _) in res.items() if eq != 1.0]
    if not res or bad:
        print(f"FAIL: not bit-equal at layers {bad}" if res else "FAIL: no compressor tapped")
        sys.exit(1)
    print(f"PASS: the official compressor equals the bf16-emulated step bit for bit at layers {list(res)}")


if __name__ == "__main__":
    main()
