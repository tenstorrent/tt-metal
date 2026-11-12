# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""QkvHeadsNormRope (custom op: head split + per-head norms + RoPE straight from the fused QKV projection output)
vs the shipped chain (nlp_create_qkv_heads_decode -> QkvNormRope) and the torch golden, single chip, at the dense
per-chip head counts (31B / 12B at TP=4 / TP=8)."""

import os

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


CI = os.environ.get("CI") == "true"
SHAPES = {
    "31b_tp4_b32": (32, 8, 4, 256),
    "31b_tp4_b1": (1, 8, 4, 256),
    "31b_tp4_b8": (8, 8, 4, 256),
    "31b_tp8_12b_tp4": (32, 4, 2, 256),
    "12b_tp8": (32, 2, 1, 256),
    "12b_tp8_b1": (1, 2, 1, 256),
    "31b_tp4_global": (32, 8, 1, 512),
    "12b_tp4_global": (32, 4, 1, 512),
    "31b_global_b1": (1, 8, 1, 512),
    "12b_global_b1": (1, 4, 1, 512),
}
CI_SHAPES = ("31b_tp4_b32", "31b_tp4_b1", "12b_tp8", "31b_tp4_global", "31b_global_b1")


def _cases():
    """(B, hq, hkv, D, forced grid): the full sweep locally; under CI=true (the ~60 s budget of a one-chip unit
    test) the dense tp4 shapes at b32 / b1 / global on the device grid plus one forced-8x8 case."""
    out = []
    for name, (B, hq, hkv, D) in SHAPES.items():
        for grid, g_id in ((None, "device_grid"), ("8x8", "grid8x8")):
            in_ci = name in CI_SHAPES and (grid is None or name == "31b_tp4_b32")
            marks = [] if in_ci or not CI else [pytest.mark.skip(reason="CI subset: the dense tp4 shapes")]
            out.append(pytest.param(B, hq, hkv, D, grid, id=f"{name}-{g_id}", marks=marks))
    return out


@pytest.mark.parametrize("B, hq, hkv, D, grid", _cases())
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_qkv_heads_norm_rope_matches_chain(mesh_device, device_params, B, hq, hkv, D, grid, monkeypatch):
    """``grid8x8``: the Wormhole-sized split (GEMMA4_TEST_GRID=8x8: at B=32 one user per core with its three
    tensors in sequence instead of 96 (user, tensor) cores) on the same chip."""
    if grid is not None:
        if B not in (32, 8):
            pytest.skip("the 8x8 split only differs at B >= 22")
        monkeypatch.setenv("GEMMA4_TEST_GRID", grid)
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
    if grid is not None:  # the chain's shards sit on the device grid; move them to the capped grid's cores
        from models.demos.gemma4.custom_ops.qkv_heads_norm_rope.op import heads_sharded_memory_config

        mem = heads_sharded_memory_config(mesh_device, B, D)
        q_s, k_s, v_s = (ttnn.to_memory_config(t, mem) for t in (q_s, k_s, v_s))
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
