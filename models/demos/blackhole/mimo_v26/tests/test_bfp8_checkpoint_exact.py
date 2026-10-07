# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase A: is bfp8 lossless for the checkpoint's fp4 (e2m1 + e8m0 per 32) experts? No device needed for the encode:
dequant a real expert (fp32, exact), round-trip through ttnn bfloat8_b (host pack/unpack) and report max abs diff and
the count of changed values. MIMO_CKPT = HF checkpoint dir (safetensors + index); skipped if unset/absent.
Note: bfp8 shares one exponent per 16 values with 7-bit mantissa; fp4 values {0,.5,1,1.5,2,3,4,6} x 2^e fit exactly only
if a 16-group's dynamic range <= 2^7 mantissa steps, so equality is expected but MEASURED here, per layer and projection.
"""
import json
import os

import pytest
import torch

import ttnn

CKPT = os.environ.get("MIMO_CKPT")
LAYERS = [int(v) for v in os.environ.get("MIMO_LAYERS", "1,20,46").split(",")]
FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dequant_fp4(w_u8, s_u8, block=32):
    """packed [N,K/2] low nibble first; scale e8m0 [N,K/block] (2^(s-127)). DSV4 dequant_fp4_weight, with e8m0 decoded."""
    b = w_u8.view(torch.uint8)
    pair = torch.stack([FP4_TABLE[(b & 0x0F).long()], FP4_TABLE[(b >> 4).long()]], dim=-1)
    n = b.size(0)
    wf = pair.reshape(n, -1)
    scale = torch.pow(2.0, s_u8.view(torch.uint8).float() - 127.0)
    return (wf.view(n, -1, block) * scale.unsqueeze(-1)).reshape(n, -1)


def _get(idx, name):
    from safetensors import safe_open

    with safe_open(os.path.join(CKPT, idx[name]), "pt") as f:
        return f.get_tensor(name)


@pytest.mark.parametrize("layer", LAYERS)
def test_bfp8_reproduces_fp4_checkpoint(layer):
    if not CKPT or not os.path.exists(os.path.join(CKPT, "model.safetensors.index.json")):
        pytest.skip("set MIMO_CKPT")
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    for e in (0, 137, 255):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
            w = dequant_fp4(_get(idx, base + ".weight"), _get(idx, base + ".weight_scale"))
            t = ttnn.from_torch(w.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
            back = ttnn.to_torch(t).float()
            # also the bf16 step (bf16 holds fp4*2^e exactly? measure)
            bf = w.to(torch.bfloat16).float()
            print(
                f"RESULT layer={layer} e={e} {proj} shape={tuple(w.shape)} bf16_maxdiff={(bf - w).abs().max():.3e} "
                f"bfp8_maxdiff={(back - w).abs().max():.3e} bfp8_changed={(back != w).sum().item()}/{w.numel()} "
                f"absmax={w.abs().max():.3e}"
            )
            assert (back - w).abs().max() == 0, "bfp8 not lossless for this tensor: log and report (precision decision)"
