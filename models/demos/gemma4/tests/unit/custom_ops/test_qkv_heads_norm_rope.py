# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""QkvHeadsNormRope (custom op: head split + per-head norms + RoPE straight from the fused QKV projection output)
vs the shipped chain (nlp_create_qkv_heads_decode -> QkvNormRope) and the torch golden, single chip, at the dense
per-chip head counts (31B / 12B at TP=4 / TP=8)."""

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.qkv_heads_norm_rope import QkvHeadsNormRope
from models.demos.gemma4.custom_ops.qkv_norm_rope import QkvNormRope

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
EPS = 1e-6


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _cos_sin(B, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randint(0, 4096, (32,), generator=g)
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
    emb = torch.cat([pos[:, None].float() * inv[None, :]] * 2, dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)  # [32, D] rows


@pytest.mark.parametrize(
    "B, hq, hkv, D",
    [
        (32, 8, 4, 256),
        (1, 8, 4, 256),
        (8, 8, 4, 256),
        (32, 4, 2, 256),
        (32, 2, 1, 256),
        (1, 2, 1, 256),
        (32, 8, 1, 512),
        (32, 4, 1, 512),
    ],
    ids=[
        "31b_tp4_b32",
        "31b_tp4_b1",
        "31b_tp4_b8",
        "31b_tp8_12b_tp4",
        "12b_tp8",
        "12b_tp8_b1",
        "31b_tp4_global",
        "12b_tp4_global",
    ],
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_qkv_heads_norm_rope_matches_chain(mesh_device, device_params, B, hq, hkv, D):
    torch.manual_seed(0)
    N = (hq + 2 * hkv) * D
    x = (torch.randn(1, 1, B, N) * 3).to(torch.bfloat16)
    gq = (torch.rand(D) + 0.5).to(torch.bfloat16)
    gk = (torch.rand(D) + 0.5).to(torch.bfloat16)
    cos, sin = _cos_sin(B, D)
    gold = QkvHeadsNormRope.golden(x, hq, hkv, D, gq, gk, EPS, cos, sin)
    dev_tile = lambda t: ttnn.from_torch(t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tx = ttnn.from_torch(
        x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tgq, tgk = dev_tile(gq.reshape(1, 1, 1, D)), dev_tile(gk.reshape(1, 1, 1, D))
    tcos, tsin = dev_tile(cos.reshape(1, 1, 32, D)), dev_tile(sin.reshape(1, 1, 32, D))
    # the shipped chain on the same L1 input: nlp_create_qkv_heads_decode (one user per core) then the fused
    # norm + RoPE op (the same compute kernel), so the two must agree bit for bit on the head rows
    q_s, k_s, v_s = ttnn.experimental.nlp_create_qkv_heads_decode(
        tx, num_heads=hq, num_kv_heads=hkv, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
    )
    ref = [ttnn.to_torch(t).float() for t in QkvNormRope.op(q_s, k_s, v_s, tgq, tgk, tcos, tsin, EPS)]
    outs = QkvHeadsNormRope.op(tx, hq, hkv, D, tgq, tgk, tcos, tsin, EPS)
    got = [ttnn.to_torch(t).float() for t in outs]
    for name, g, r, o, heads in zip("qkv", gold, ref, got, (hq, hkv, hkv)):
        assert list(o.shape) == [1, B, heads, D], (name, o.shape)
        assert torch.isfinite(o).all(), name
        p = _pcc(g, o)
        print(f"[B{B} hq{hq} hkv{hkv} D{D} {name}] pcc vs golden {p:.6f} max|diff| vs chain {(r - o).abs().max():.4f}")
        assert p > 0.9999, (name, p)
        assert torch.equal(o, r), f"{name}: differs from nlp_create_qkv_heads_decode + QkvNormRope"
    assert outs[0].memory_config() == q_s.memory_config(), "q layout must match nlp_create_qkv_heads_decode"
    # program-cache hit on a fresh input buffer
    tx2 = ttnn.from_torch(
        x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    got2 = [ttnn.to_torch(t).float() for t in QkvHeadsNormRope.op(tx2, hq, hkv, D, tgq, tgk, tcos, tsin, EPS)]
    for a, b in zip(got, got2):
        assert torch.equal(a, b), "cache-hit call differs"
