# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Router matmul fidelity (tuned 2D config, fp32 dest / fp32 out): HiFi2 / HiFi3 logits vs HiFi4, and how often the
biased top-8 changes. Real gate weights and e_score_correction_bias of layer 1; x = normalized real-prompt
embeddings (a stand-in for post-attention hidden states). Signposts ``routerfid_M{M}_{fid}``."""

import pytest
import torch

import ttnn
from models.demos.mimo_v2_d_p.reference import hf
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.reference.weights import global_state, layer_state
from models.demos.mimo_v2_d_p.tt.mm_configs import router_mm_config

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None


@pytest.mark.parametrize("M", [2048])
def test_router_fidelity(device, M):
    cfg = MiMoTextConfig.from_json()
    sd = layer_state(1, cfg, experts=False)
    wg = sd["mlp.gate.weight"].float()  # [256, 4096]
    bias = sd["mlp.gate.e_score_correction_bias"].float()
    ids = hf.tokenize_prompt(M)
    x = global_state()["embed_tokens.weight"][ids].float()
    x = x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    xt = ttnn.from_torch(x[None, None], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    wt = ttnn.from_torch(wg.T.contiguous()[None, None], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    pc = router_mm_config(device, M)
    out = {}
    for fid in ("HiFi4", "HiFi3", "HiFi2"):
        ckc = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=True
        )
        for it in range(4):
            if it:
                signpost(f"routerfid_M{M}_{fid}_start")
            o = ttnn.linear(xt, wt, dtype=ttnn.float32, compute_kernel_config=ckc, program_config=pc)
            ttnn.synchronize_device(device)
            if it:
                signpost(f"routerfid_M{M}_{fid}_end")
        out[fid] = ttnn.to_torch(o).float().reshape(M, -1)
    ref = x.bfloat16().float() @ wg.T.bfloat16().float()
    sel = lambda l: (torch.sigmoid(l) + bias).topk(8, dim=-1).indices.sort(-1).values
    for fid, l in out.items():
        print(
            f"FID {fid}: max |dlogit| vs fp32 host {(l - ref).abs().max():.3e}, vs HiFi4 {(l - out['HiFi4']).abs().max():.3e}; "
            f"top-8 differs from HiFi4 on {100 * (sel(l) != sel(out['HiFi4'])).any(-1).float().mean():.2f}% of tokens, "
            f"from fp32 host on {100 * (sel(l) != sel(ref)).any(-1).float().mean():.2f}%"
        )
