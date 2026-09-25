# SPDX-License-Identifier: MIT
"""Device depth sweep: hidden/logits NRMSE vs encoder depth + gelu probe.

Diagnoses whether the device-vs-fp32-twin hidden error (5.0423e-2 at 33
layers,. gate 0.04) grows ~linearly in depth (coherent
per-layer bias, e.g. a deterministic function approximation like the TTNN
gelu 'Accurate' piecewise CDF) or ~sqrt(depth) (uncorrelated rounding).
Reuses the test_hidden_mini harness pattern: sliced canonical weights
(prefix layers), dataclasses.replace config override, exact public smoke
input, harness batch-row NRMSE metric (per-batch-row RMS error ratio, max
over rows; for B=1 equals the whole-tensor RMS ratio), one device job
for all depths, DRAM usage printed between builds (weights freed manually:
build()'s atexit hook keeps each backend object alive).

Gelu probe: captures REAL ffn1 pre-activations (layers 0/16/32) from the
fp32 twin, runs ttnn.gelu on them on device (bf16, TILE) and compares
against (a) exact erf gelu on the same bf16-valued inputs and (b) the
emulated TT piecewise-CDF formula extracted from the runtime kernel source
(tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h,
tt-metal-fd80faa3). Validates the CPU sim's gelu emulation on the real
activation distribution.
"""
from __future__ import annotations

import dataclasses
import gc
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")
from tt.esm2.config import Esm2TTConfig
from tt.esm2.loader import load_canonical_weights
from tt.esm2.reference_layers import Esm2Model

DEPTHS = (2, 4, 8, 16, 33)
PROBE_LAYERS = (0, 16, 32)

# TT gelu 'Accurate' BF16 piecewise-CDF coefficients (kernel source check).
# Regions: x<=-5.54259443 -> 0; (-5.5426,-3.125] -> x*H, H = exp(-x^2/2)*hc(x)
# snapped to the 2^-25 grid; (-3.125,2.78125] -> x*(0.5 + x*P(x^2));
# x>2.78125 -> x.  Output rounded to bf16 by the caller (dest is 16-bit).
_G_C1, _G_C3, _G_C5, _G_C7 = 3.9894227818e-01, -6.6361041488e-02, 9.7720050615e-03, -1.0717806322e-03
_G_C9, _G_C11, _G_C13 = 8.1812159812e-05, -3.8082057209e-06, 7.9821413868e-08
_H_C0, _H_C1, _H_C2, _H_C3 = 3.0369991064e-01, 9.5413386822e-02, 1.3809983619e-02, 7.5950479368e-04
_GELU_SAT = -5.54259443
_GRID = 0.375


def gelu_tt_emul(x: torch.Tensor) -> torch.Tensor:
    """Emulate TTNN gelu default 'Accurate' (BF16 piecewise CDF, fp32 math)."""
    x2 = x * x
    hc = _H_C0 + x * (_H_C1 + x * (_H_C2 + x * _H_C3))
    h = torch.exp(-0.5 * x2) * hc
    hs = (h + _GRID) - _GRID                       # round to 2^-25 grid
    res = x * hs                                   # exp tail region
    p = _G_C1 + x2 * (_G_C3 + x2 * (_G_C5 + x2 * (_G_C7 + x2 * (_G_C9 + x2 * (_G_C11 + x2 * _G_C13)))))
    core = x * (0.5 + x * p)
    res = torch.where(x > -3.125, core, res)       # core CDF region
    res = torch.where(x > 2.78125, x, res)         # identity region
    res = torch.where(x <= _GELU_SAT, torch.zeros_like(res), res)
    return res


def row_nrmse(actual, expected):
    """Harness metric (evaluate.py max_row_nrmse): per-batch-row RMS error
    ratio, MAX over rows; for B=1 max == mean == whole-tensor RMS ratio."""
    axes = tuple(range(1, actual.ndim))
    err = np.sqrt(np.mean((actual - expected) ** 2, axis=axes)) / np.maximum(
        1e-12, np.sqrt(np.mean(expected ** 2, axis=axes)))
    return float(err.max()), float(err.mean())


def slice_weights(w, depth):
    return {k: v for k, v in w.items()
            if not (k.startswith("layers.") and int(k.split(".")[1]) >= depth)}


def main():
    d = np.load("/input/inputs.npz")
    ids, am = d["short__input_ids"], d["short__attention_mask"]
    with open("/weights/config.json") as f:
        cfg0 = Esm2TTConfig.from_dict(json.load(f))
    wall = load_canonical_weights("/weights", cfg0)

    import ttnn
    from tt.esm2.ttnn_backend import TtnnEsm2

    def dram_used(dev):
        try:
            mv = ttnn.device.get_memory_view(dev, ttnn.BufferType.DRAM)
            return float(getattr(mv, "used_bytes", -1))
        except Exception:
            return -1.0

    caps: dict = {}

    def hook_for(i):
        def h(_mod, _inp, out):
            caps[i] = out.detach().reshape(-1).clone()
        return h

    def free_backend(tt):
        # build()'s atexit hook pins the object; drop device tensors manually.
        tt.layer_ops = None
        for k in ("final_w", "final_b", "lm_d_w", "lm_d_b", "lm_l_w", "lm_l_b",
                  "dec_w", "lm_bias"):
            setattr(tt, k, None)
        tt._rotary_cache.clear()
        tt._mask_cache.clear()

    rows = []
    device = ttnn.open_device(device_id=0)
    try:
        for depth in DEPTHS:
            cfg = dataclasses.replace(cfg0, num_hidden_layers=depth)
            w = slice_weights(wall, depth)
            ref = Esm2Model(cfg, weights=w).eval()
            if depth == max(DEPTHS):
                for i in PROBE_LAYERS:
                    ref.layers[i].ffn1.register_forward_hook(hook_for(i))
            with torch.no_grad():
                lr, hr = ref(torch.from_numpy(ids), torch.from_numpy(am))
            lr, hr = lr.numpy(), hr.numpy()
            tt = TtnnEsm2(cfg, w, device=device, precision="bf16").build()
            out = tt.forward(ids, am)
            hm, ha = row_nrmse(out["hidden"], hr)
            lm, la = row_nrmse(out["logits"], lr)
            rows.append((depth, hm, lm))
            print(f"depth={depth:2d} hidden max={hm:.4e} mean={ha:.4e} "
                  f"logits max={lm:.4e} mean={la:.4e} dram={dram_used(device)/1e6:.0f}MB",
                  flush=True)
            free_backend(tt)
            del tt, ref, w, out
            gc.collect()
        print("--- fits (hidden max-row NRMSE vs depth) ---", flush=True)
        dep = np.array([r[0] for r in rows], float)
        h = np.array([r[1] for r in rows])
        for name, basis in (("linear", dep), ("sqrt", np.sqrt(dep))):
            c = np.polyfit(basis, h, 1)
            pred = np.polyval(c, basis)
            r2 = 1 - float(((h - pred) ** 2).sum() / max(1e-30, ((h - h.mean()) ** 2).sum()))
            print(f"fit {name}: slope={c[0]:.3e} intercept={c[1]:.3e} R2={r2:.4f}", flush=True)
        print("SWEEP_DONE", flush=True)

        for i in PROBE_LAYERS:
            x = caps[i]
            xb = x.to(torch.bfloat16).to(torch.float32)  # values the SFPU sees
            t = ttnn.to_device(
                ttnn.from_torch(xb.to(torch.bfloat16).reshape(1, -1), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT), device)
            g = ttnn.gelu(t)
            ttnn.synchronize_device(device)
            dev = ttnn.to_torch(ttnn.to_layout(ttnn.from_device(g),
                                               ttnn.ROW_MAJOR_LAYOUT)).reshape(-1).float()
            ref_e = F.gelu(xb)          # exact erf gelu on same bf16 inputs
            emu = gelu_tt_emul(xb)

            def rms(a, b):
                return float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt())

            print(f"L{i:2d} gelu n={x.numel()}: dev_vs_erf NRMSE={rms(dev, ref_e):.3e} "
                  f"emu_vs_erf NRMSE={rms(emu, ref_e):.3e} "
                  f"max|dev-emu|={float((dev - emu).abs().max()):.3e} "
                  f"max|dev-erf|={float((dev - ref_e).abs().max()):.3e}", flush=True)
        print("PROBE_DONE", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
