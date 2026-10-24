# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""custom_ops/qkv_norm_rope_prefill vs the composite prefill path (per-head rms_norm with the flatten reshapes,
rotary_embedding) and the torch golden, single chip: the 26B per-chip shapes (tp4 sliding 4 q / 2 kv heads D 256,
global 4 q / 1 kv D 512 with identity-padded partial rotary, tp1 16 / 8), 128 and 2048 positions (1 and 16 units
per core), the batched-prefill form [B, H, S/B, D], a chunk at a row-tile offset, and the KV-shared form (q only)."""

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.qkv_norm_rope_prefill import QkvNormRopePrefill
from models.demos.gemma4.tt.attention.operations import apply_per_head_norm, apply_rope

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
EPS = 1e-6


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _cos_sin(s_max, d, partial):
    """HF-style caches [1, 1, s_max, d]; ``partial`` rotary dims (or None = full): identity padding beyond."""
    pos = torch.arange(s_max).float()
    rot = partial or d
    inv = 1.0 / (10000 ** (torch.arange(0, rot, 2).float() / rot))
    f = torch.outer(pos, inv)
    emb = torch.cat([f, f], dim=-1)
    cos, sin = torch.ones(s_max, d), torch.zeros(s_max, d)
    cos[:, :rot], sin[:, :rot] = emb.cos(), emb.sin()
    return cos.reshape(1, 1, s_max, d).to(torch.bfloat16), sin.reshape(1, 1, s_max, d).to(torch.bfloat16)


@pytest.mark.parametrize(
    "B, Hq, Hkv, S, D, partial, offset, shared_kv",
    [
        (1, 4, 2, 128, 256, None, 0, False),  # 26B tp4 sliding, 128 tokens: 32 units on 32 cores
        (1, 4, 1, 128, 512, 256, 0, False),  # 26B tp4 global (kv replicated), partial rotary 256 of 512
        (1, 16, 8, 128, 256, None, 0, False),  # tp1
        (1, 4, 2, 2048, 256, None, 0, False),  # 64 row tiles: several units per core
        (4, 4, 2, 32, 256, None, 0, False),  # batched prefill [B, H, S/B, D]
        (1, 4, 2, 128, 256, None, 20, False),  # chunk at position 640
        (1, 4, 2, 128, 512, 256, 0, True),  # KV-shared layer: q only
    ],
    ids=lambda v: str(v),
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_qkv_norm_rope_prefill_matches_composite(
    mesh_device, device_params, B, Hq, Hkv, S, D, partial, offset, shared_kv
):
    torch.manual_seed(0)
    q = (torch.randn(B, Hq, S, D) * 3).to(torch.bfloat16)
    k = (torch.randn(B, Hkv, S, D) * 3).to(torch.bfloat16)
    v = (torch.randn(B, Hkv, S, D) * 3).to(torch.bfloat16)
    gq = (torch.rand(D) + 0.5).to(torch.bfloat16)
    gk = (torch.rand(D) + 0.5).to(torch.bfloat16)
    s_max = 32 * offset + S
    cos, sin = _cos_sin(s_max, D, partial)
    cos_s, sin_s = cos[:, :, 32 * offset :], sin[:, :, 32 * offset :]
    gold_q = QkvNormRopePrefill.golden(q, gq, EPS, cos_s, sin_s)
    gold_k = QkvNormRopePrefill.golden(k, gk, EPS, cos_s, sin_s)
    gold_v = QkvNormRopePrefill.golden(v, None, EPS)
    dev = lambda t: ttnn.from_torch(t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tq, tk, tv = dev(q), dev(k), dev(v)
    tgq, tgk = dev(gq.reshape(1, 1, 1, D)), dev(gk.reshape(1, 1, 1, D))
    tcos, tsin = dev(cos), dev(sin)
    # the composite (tt/attention/prefill.py): flatten -> rms_norm (HiFi4, fp32 acc) -> unflatten, rotary_embedding
    gq_rm = ttnn.from_torch(
        gq.reshape(1, 1, -1, 32), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )
    gk_rm = ttnn.from_torch(
        gk.reshape(1, 1, -1, 32), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )
    tcos_s, tsin_s = dev(cos_s), dev(sin_s)
    rq = apply_per_head_norm(tq, gq_rm, EPS, with_scale=True, fp32_accumulate=True)
    rq = apply_rope(rq, tcos_s, tsin_s)
    rk = apply_rope(apply_per_head_norm(tk, gk_rm, EPS, with_scale=True, fp32_accumulate=True), tcos_s, tsin_s)
    rv = apply_per_head_norm(tv, None, EPS, with_scale=False, fp32_accumulate=True)
    ref = [ttnn.to_torch(t).float() for t in (rq, rk, rv)]
    # the op
    oq, ok, ov = QkvNormRopePrefill.op(
        tq, None if shared_kv else tk, None if shared_kv else tv, tgq, tgk, tcos, tsin, EPS, row_tile_offset=offset
    )
    assert oq.memory_config() == tq.memory_config() and [int(s) for s in oq.shape] == [B, Hq, S, D]
    if shared_kv:
        assert ok is None and ov is None
        pairs = [("q", gold_q, ref[0], oq)]
    else:
        pairs = [("q", gold_q, ref[0], oq), ("k", gold_k, ref[1], ok), ("v", gold_v, ref[2], ov)]
    for name, g, r, o in pairs:
        got = ttnn.to_torch(o).float()
        p_g, p_r = _pcc(g, got), _pcc(r, got)
        rows = [
            round(_pcc(g[:, :, i * 32 : (i + 1) * 32], got[:, :, i * 32 : (i + 1) * 32]), 4)
            for i in range(min(S // 32, 4))
        ]
        heads = [round(_pcc(g[:, h], got[:, h]), 4) for h in range(min(g.shape[1], 4))]
        print(
            f"[{name} B{B} H{g.shape[1]} S{S} D{D} partial={partial} offset={offset}] pcc vs golden {p_g:.6f} (max|d| {(g - got).abs().max():.4f}), vs composite {p_r:.6f}; first row tiles {rows}; heads {heads}"
        )
        assert p_g > 0.9999 and p_r > 0.9999, (name, p_g, p_r)
    # program-cache hit on fresh tensors (new addresses)
    tq2, tk2, tv2 = dev(q), dev(k), dev(v)
    oq2, ok2, ov2 = QkvNormRopePrefill.op(
        tq2, None if shared_kv else tk2, None if shared_kv else tv2, tgq, tgk, tcos, tsin, EPS, row_tile_offset=offset
    )
    assert torch.equal(ttnn.to_torch(oq), ttnn.to_torch(oq2)), "cache-hit q differs"
    if not shared_kv:
        assert torch.equal(ttnn.to_torch(ov), ttnn.to_torch(ov2)), "cache-hit v differs"


@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_two_lengths_in_one_session(mesh_device, device_params):
    """1024 then 2048 then 1024 rows on the same device and core set: the second call needs more units per core than
    the cached first program carried (generic_op's structural hash ignores per-core runtime-arg counts), which
    without the layout in the cache key overrode arguments the program never had ("Index 14 is larger than runtime
    args size 14") and produced garbage (0540: the 12B prose at 2048 collapsed to top-1 0.001 after a 512 run)."""
    torch.manual_seed(0)
    B, Hq, Hkv, D = 1, 4, 2, 256
    s_max = 2048
    cos, sin = _cos_sin(s_max, D, None)
    gq_rm, gk_rm = (torch.rand(D) + 0.5).to(torch.bfloat16), (torch.rand(D) + 0.5).to(torch.bfloat16)
    dev = lambda x: ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tgq, tgk = dev(gq_rm.reshape(1, 1, 1, D)), dev(gk_rm.reshape(1, 1, 1, D))
    tcos, tsin = dev(cos), dev(sin)
    for S in (1024, 2048, 1024):
        q = (torch.randn(B, Hq, S, D) * 2).to(torch.bfloat16)
        k = (torch.randn(B, Hkv, S, D) * 2).to(torch.bfloat16)
        v = (torch.randn(B, Hkv, S, D) * 2).to(torch.bfloat16)
        gold_q = QkvNormRopePrefill.golden(q, gq_rm, EPS, cos[:, :, :S], sin[:, :, :S])
        gold_k = QkvNormRopePrefill.golden(k, gk_rm, EPS, cos[:, :, :S], sin[:, :, :S])
        gold_v = QkvNormRopePrefill.golden(v, None, EPS)
        oq, ok, ov = QkvNormRopePrefill.op(dev(q), dev(k), dev(v), tgq, tgk, tcos, tsin, EPS)
        for name, g, o in (("q", gold_q, oq), ("k", gold_k, ok), ("v", gold_v, ov)):
            p = _pcc(g, ttnn.to_torch(o).float())
            print(f"[two lengths S{S} {name}] pcc vs golden {p:.6f}")
            assert p > 0.9999, (S, name, p)
