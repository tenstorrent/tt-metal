# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""QkvNormRope (custom op) on the height-sharded head tensors vs the shipped chain (to DRAM, rms_norm per head,
composite RoPE, back to sharded) and the torch golden, single chip."""

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.qkv_norm_rope import QkvNormRope
from models.demos.gemma4.tt.attention.operations import apply_rope_decode_peruser

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
EPS = 1e-6


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _cos_sin(B, D, partial=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randint(0, 4096, (B,), generator=g)
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
    emb = torch.cat([pos[:, None].float() * inv[None, :]] * 2, dim=-1)
    cos, sin = emb.cos(), emb.sin()
    if partial:
        cos[:, partial:] = 1.0
        sin[:, partial:] = 0.0
    return cos.reshape(1, B, 1, D).to(torch.bfloat16), sin.reshape(1, B, 1, D).to(torch.bfloat16)


def _sharded(t, device, B):
    """[1, B, H, D] -> L1 height sharded, one user per core, shard [32, D] (the nlp_create_qkv_heads_decode form)."""
    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(B, grid, True)  # row-major, like nlp_create_qkv_heads_decode
    mem = ttnn.create_sharded_memory_config(
        shape=(32, t.shape[-1]),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    x = ttnn.from_torch(
        t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return ttnn.to_memory_config(x, mem)


def _composite(q, k, v, gq_rm, gk_rm, cos_b, sin_b):
    """the shipped decode path with GEMMA4_HEAD_NORM_4D semantics (rms_norm straight on [1,B,H,D])."""
    sharded_mem = q.memory_config()
    outs = []
    for t, g, rope in ((q, gq_rm, True), (k, gk_rm, True), (v, None, False)):
        x = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.rms_norm(x, weight=g, epsilon=EPS) if g is not None else ttnn.rms_norm(x, epsilon=EPS)
        if rope:
            x = apply_rope_decode_peruser(x, cos_b, sin_b)
        outs.append(ttnn.to_memory_config(x, sharded_mem))
    return outs


@pytest.mark.parametrize("in_place", [True, False], ids=["inplace", "fresh_out"])
@pytest.mark.parametrize(
    "B, Hq, Hkv, D, partial",
    [
        (32, 4, 2, 256, None),
        (32, 4, 2, 256, 64),
        (1, 4, 2, 256, None),
        (8, 16, 8, 256, None),
        (32, 4, 2, 512, 256),
        # dense per-chip shapes: 31B tp4 sliding (8 q / 4 kv) and global (8 q / 1 kv, D 512); 12B tp4 sliding
        # (4 q / 2 kv) and global (4 q / 1 kv replicated, D 512); tp8: 31B 4/2, 12B 2/1 and 2 q / 1 kv global
        (32, 8, 4, 256, None),
        (1, 8, 4, 256, None),
        (8, 8, 4, 256, None),
        (32, 8, 1, 512, 128),
        (1, 8, 1, 512, 128),
        (32, 4, 1, 512, 128),
        (32, 2, 1, 256, None),
        (32, 2, 1, 512, 128),
    ],
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_qkv_norm_rope_matches_composite(mesh_device, device_params, B, Hq, Hkv, D, partial, in_place):
    """``in_place``: outputs written over the input shards (the model's default form) or fresh output tensors."""
    torch.manual_seed(0)
    q = (torch.randn(1, B, Hq, D) * 3).to(torch.bfloat16)
    k = (torch.randn(1, B, Hkv, D) * 3).to(torch.bfloat16)
    v = (torch.randn(1, B, Hkv, D) * 3).to(torch.bfloat16)
    gq = (torch.rand(D) + 0.5).to(torch.bfloat16)
    gk = (torch.rand(D) + 0.5).to(torch.bfloat16)
    cos, sin = _cos_sin(B, D, partial)
    gold = [
        QkvNormRope.golden(q, gq, EPS, cos, sin),
        QkvNormRope.golden(k, gk, EPS, cos, sin),
        QkvNormRope.golden(v, None, EPS),
    ]
    tq, tk, tv = (_sharded(t, mesh_device, B) for t in (q, k, v))
    dev_tile = lambda t: ttnn.from_torch(t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tgq, tgk = dev_tile(gq.reshape(1, 1, 1, D)), dev_tile(gk.reshape(1, 1, 1, D))
    tcos, tsin = dev_tile(cos), dev_tile(sin)  # [1, B, 1, D]: the composite's per-user form
    # the op takes the position-gathered [1, 1, 32, D] rows (row u = user u; rows >= B arbitrary)
    rows = lambda t: torch.cat([t.reshape(B, D), torch.randn(32 - B, D)]).reshape(1, 1, 32, D).to(torch.bfloat16)
    tcos_rows, tsin_rows = dev_tile(rows(cos.float())), dev_tile(rows(sin.float()))
    gq_rm = ttnn.from_torch(
        gq.reshape(1, 1, -1, 32), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )
    gk_rm = ttnn.from_torch(
        gk.reshape(1, 1, -1, 32), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )
    ref = [ttnn.to_torch(t).float() for t in _composite(tq, tk, tv, gq_rm, gk_rm, tcos, tsin)]
    outs = QkvNormRope.op(tq, tk, tv, tgq, tgk, tcos_rows, tsin_rows, EPS, in_place=in_place)
    assert all((o is i) == in_place for o, i in zip(outs, (tq, tk, tv)))
    got = [ttnn.to_torch(t).float() for t in outs]
    # diagnostics first: per tensor, per column tile (q, v) and per head row (q), so a layout bug is localized
    print(
        f"[B{B}] q page size {tq.buffer_page_size()} shard {tq.memory_config().shard_spec.shape} grid {tq.memory_config().shard_spec.grid}"
    )
    for name, g, o in zip("qkv", gold, got):
        cols = [round(_pcc(g[..., i * 32 : (i + 1) * 32], o[..., i * 32 : (i + 1) * 32]), 4) for i in range(D // 32)]
        heads = [round(_pcc(g[:, :, h], o[:, :, h]), 4) for h in range(min(g.shape[2], 4))]
        print(
            f"[B{B} {name}] pcc per column tile {cols}; per head {heads}; |o| mean {o.abs().mean():.3f} |g| mean {g.abs().mean():.3f}"
        )
    # identity RoPE (cos 1, sin 0): the pure norm path on the same shards
    ones = ttnn.from_torch(
        torch.ones(1, 1, 32, D, dtype=torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    zeros = ttnn.from_torch(
        torch.zeros(1, 1, 32, D, dtype=torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    tq_i, tk_i, tv_i = (_sharded(t, mesh_device, B) for t in (q, k, v))
    o_i = ttnn.to_torch(QkvNormRope.op(tq_i, tk_i, tv_i, tgq, tgk, ones, zeros, EPS, in_place=in_place)[0]).float()
    g_i = QkvNormRope.golden(q, gq, EPS)
    print(
        f"[B{B} q identity-rope] pcc vs norm golden {_pcc(g_i, o_i):.6f}; per column tile {[round(_pcc(g_i[..., i*32:(i+1)*32], o_i[..., i*32:(i+1)*32]), 4) for i in range(D // 32)]}"
    )
    # gamma = ones as well: the q path with numerically no gamma and no rotation (structurally still the q path)
    tg1 = ttnn.from_torch(
        torch.ones(1, 1, 1, D, dtype=torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    tq_j, tk_j, tv_j = (_sharded(t, mesh_device, B) for t in (q, k, v))
    o_j = ttnn.to_torch(QkvNormRope.op(tq_j, tk_j, tv_j, tg1, tg1, ones, zeros, EPS, in_place=in_place)[0]).float()
    g_j = QkvNormRope.golden(q, None, EPS)
    print(
        f"[B{B} q gamma=1 identity-rope] pcc vs plain norm {_pcc(g_j, o_j):.6f}; per column tile {[round(_pcc(g_j[..., i*32:(i+1)*32], o_j[..., i*32:(i+1)*32]), 4) for i in range(D // 32)]}"
    )
    # which tile does a wrong tile look like? correlate output tile 1 against every golden tile of the plain norm
    print(
        f"[B{B} q gamma=1 identity-rope] out tile 1 vs golden tiles {[round(_pcc(g_j[..., i*32:(i+1)*32], o_j[..., 32:64]), 3) for i in range(D // 32)]}"
    )
    gq_rows = gq.reshape(1, 1, 1, D).float()
    print(
        f"[B{B}] gamma tile means {[round(float(gq_rows[..., i*32:(i+1)*32].mean()), 3) for i in range(D // 32)]}; out/golden ratio per tile (q real gamma, identity rope) {[round(float((o_i[..., i*32:(i+1)*32] / g_i[..., i*32:(i+1)*32].clamp(min=1e-3)).median()), 3) for i in range(D // 32)]}"
    )
    for name, g, r, o in zip("qkv", gold, ref, got):
        assert torch.isfinite(o).all(), name
        p_g, p_r = _pcc(g, o), _pcc(r, o)
        print(
            f"[B{B} {name} D{D} partial={partial}] pcc vs golden {p_g:.6f} (max|d| {(g - o).abs().max():.4f}), vs composite {p_r:.6f}"
        )
    for name, g, r, o in zip("qkv", gold, ref, got):
        p_g, p_r = _pcc(g, o), _pcc(r, o)
        assert p_g > 0.9999 and p_r > 0.9999, (name, p_g, p_r)
    assert all(o.memory_config() == i.memory_config() for o, i in zip(outs, (tq, tk, tv)))
    # program-cache hit on fresh shards (new L1 addresses: the tensor-backed CBs must follow them)
    tq2, tk2, tv2 = (_sharded(t, mesh_device, B) for t in (q, k, v))
    got2 = [
        ttnn.to_torch(t).float()
        for t in QkvNormRope.op(tq2, tk2, tv2, tgq, tgk, tcos_rows, tsin_rows, EPS, in_place=in_place)
    ]
    for name, a, b in zip("qkv", got, got2):
        assert torch.equal(a, b), f"cache-hit {name} differs: max|diff| {(a - b).abs().max()}"
