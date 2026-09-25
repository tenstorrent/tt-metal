# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device) checks that the TP / EP layouts the device code builds are correct for every mesh
width, including the BH Galaxy 8x4 (TP=4, EP=32) this code targets but the QuietBox cannot run.

* attention: simulate the TP-column-sharded attention exactly as the device consumes ``attention_host_weights``
  (per-col GQA heads, Meta-permuted partial rope, V scale/pad, sink/scale) and compare to HF MiMoV2Attention.
* MoE: capacity factor / dispatch constants for 4 and 32 chips.
"""

import pytest
import torch

from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants
from models.demos.mimo_v2_d_p.reference import hf
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.reference.weights import layer_state
from models.demos.mimo_v2_d_p.tt.attention.attention import attention_host_weights, kv_heads_for_col
from models.demos.mimo_v2_d_p.tt.ffn import moe_capacity_factor
from models.demos.mimo_v2_d_p.tt.rope import meta_cos_sin


def rope_meta(x, cos, sin):
    """Meta interleaved rotation on the first cos.shape[-1] dims of x [.., S, D]."""
    r = cos.shape[-1]
    xr = x[..., :r]
    rot = torch.stack([-xr[..., 1::2], xr[..., 0::2]], -1).flatten(-2)
    return torch.cat([xr * cos + rot * sin, x[..., r:]], -1)


def simulate_tp_attention(cfg, layer_idx, host, x, tp):
    spec = cfg.layer_attn(layer_idx)
    n_q, n_kv, hd, vd = spec.n_q, spec.n_kv, spec.head_dim, spec.v_head_dim
    nq_l = n_q // tp
    S = x.shape[1]
    cos, sin = meta_cos_sin(S, spec)
    cols_per = host["wqkv"].shape[1] // tp
    mask = hf.mask(torch.arange(S), S, spec.window)[0, 0]
    out = 0
    for c in range(tp):
        kv_idx = kv_heads_for_col(c, tp, n_q, n_kv)
        nkv_l = len(kv_idx)
        qkv = x[0] @ host["wqkv"][:, c * cols_per : (c + 1) * cols_per]
        q = qkv[:, : nq_l * hd].view(S, nq_l, hd).transpose(0, 1)
        k = qkv[:, nq_l * hd : (nq_l + nkv_l) * hd].view(S, nkv_l, hd).transpose(0, 1)
        v = qkv[:, (nq_l + nkv_l) * hd :].view(S, nkv_l, hd).transpose(0, 1)[..., :vd]
        q, k = rope_meta(q, cos[0, 0], sin[0, 0]), rope_meta(k, cos[0, 0], sin[0, 0])
        o = []
        for h in range(nq_l):
            g = c * nq_l + h  # global q head
            kv_l = kv_idx.index(g // (n_q // n_kv))  # the col-local kv head this q head uses
            s = (q[h] @ k[kv_l].T) * hd**-0.5 + mask
            if host["sink"] is not None:
                s = torch.cat([s, (host["sink"][0, g, 0, 0] * hd**-0.5).expand(S, 1)], -1)
            p = torch.softmax(s, -1)[:, :S]
            o.append(p @ v[kv_l])
        out = out + torch.cat(o, -1) @ host["wo"][c * nq_l * vd : (c + 1) * nq_l * vd]
    return out[None]


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["GA", "SWA"])
@pytest.mark.parametrize("tp", [1, 2, 4, 8], ids=lambda t: f"tp{t}")
def test_tp_attention_layout(layer_idx, tp):
    cfg = MiMoTextConfig.from_json()
    sd = {k[len("self_attn.") :]: v for k, v in layer_state(layer_idx, cfg, experts=False).items() if k.startswith("self_attn.")}
    torch.manual_seed(0)
    S = 256
    x = torch.randn(1, S, cfg.hidden_size) * 0.5
    hcfg = hf.hf_config()
    _, mod = hf.hf_modules()
    spec = cfg.layer_attn(layer_idx)
    ref = mod.MiMoV2Attention(hcfg, spec.window is not None, layer_idx, projection_layout="fused_qkv")
    ref.load_state_dict({k: v.float() for k, v in sd.items()})
    with torch.no_grad():
        pos = torch.arange(S)[None]
        cos, sin = hf.rotary(spec.window is not None, hcfg)(x, pos)
        want, _ = ref.float()(x, (cos, sin), hf.mask(pos[0], S, spec.window))
        got = simulate_tp_attention(cfg, layer_idx, attention_host_weights(cfg, layer_idx, sd, tp), x, tp)
    pcc = torch.corrcoef(torch.stack([want.flatten(), got.flatten()]))[0, 1].item()
    assert pcc > 0.99999, pcc


@pytest.mark.parametrize("n_dev,expected_cap", [(4, 4), (32, 2)], ids=["2x2", "galaxy-8x4"])
def test_moe_capacity(n_dev, expected_cap):
    cfg = MiMoTextConfig.from_json()
    cap = moe_capacity_factor(cfg.num_experts_per_tok, cfg.n_routed_experts, n_dev)
    assert cap == expected_cap
    dgs = 2 if n_dev == 4 else 8
    epc, _, buf, _ = compute_constants(640, cfg.n_routed_experts, cfg.num_experts_per_tok, n_dev, dgs, cap)
    assert epc == cfg.n_routed_experts // n_dev
    # expected routed tokens per chip = dgs * seq * K / n_dev: the buffer holds >= 2x that
    assert buf >= 2 * dgs * 640 * cfg.num_experts_per_tok / n_dev
