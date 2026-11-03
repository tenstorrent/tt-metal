# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The K / V cache only ever holds finite rows (attention.py KV_SANITIZE), on one chip:

* ``finite_kv`` (one ttnn.clamp) clamps to +-KV_LIMIT, maps NaN to one of the bounds (the SFPU clamp's compares are
  not IEEE) and leaves every other value bit for bit, in an interleaved and in the decode heads' height-sharded layout;
* SwaDecodeQkv (the SWA decode's fused qkv split, RoPE and cache update) writes a finite row for a user whose new K or
  V is NaN or infinite, and the other users' rows exactly as without the bad one;
* the SWA decode SDPA (paged MLA decode with the sliding window and sinks, as the SWA layers run it) does not change
  when the rows it masks hold any finite values up to +-KV_LIMIT: past the query's position in its block, and in the
  null block 0 its out-of-window table entries name. That is what the clamp relies on: a NaN or infinite masked row
  does change it (the mask is an added -inf).
"""

import pytest
import torch

import ttnn
from models.experimental.mimo_v2_6_flash.tt.attention import KV_LIMIT, KV_SANITIZE, SwaDecodeQkv, finite_kv
from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan

TILE, BLOCK, NH, DK, DV, WIN = 32, 64, 8, 192, 128, 128

pytestmark = pytest.mark.skipif(not KV_SANITIZE, reason="MIMO_KV_SANITIZE=0")


def _special(shape, seed):
    torch.manual_seed(seed)
    x = torch.randn(shape) * 4
    flat = x.view(-1)
    picks = torch.randperm(flat.numel())[:60]
    for i, value in zip(picks.tolist(), [float("nan"), float("inf"), -float("inf"), 1e38, -1e38, 7e4] * 10):
        flat[i] = value
    return x


def _check(got, x):
    """``got`` is finite_kv of ``x``: x after the bf16 round trip the device sees, clamped, a NaN at a bound."""
    x = x.to(torch.bfloat16).float()
    nan = torch.isnan(x)
    assert torch.isfinite(got).all()
    assert torch.equal(got[~nan], x[~nan].clamp(-KV_LIMIT, KV_LIMIT))
    assert (got[nan].abs() == KV_LIMIT).all()


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_finite_kv(mesh_device):
    x = _special((1, 1, 64, DK), 0)
    t = ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    _check(ttnn.to_torch(ttnn.get_device_tensors(finite_kv(t))[0]).float(), x)

    # The decode layout: the K heads nlp_create_qkv_heads_decode makes (height-sharded, one user per core).
    qkv = _special((1, 1, TILE, (NH + 2) * DK), 1)
    xq = ttnn.from_torch(
        qkv, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    _q, k, _v = ttnn.experimental.nlp_create_qkv_heads_decode(
        xq, num_heads=NH, num_kv_heads=1, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
    )
    before = ttnn.to_torch(ttnn.get_device_tensors(k)[0]).float()
    mc = k.memory_config()
    clean = finite_kv(k)
    assert clean.memory_config() == mc
    got = ttnn.to_torch(ttnn.get_device_tensors(clean)[0]).float()
    _check(got, before)


def _swa_inputs(device, users, positions, block_ids, seed):
    torch.manual_seed(seed)
    blocks = 1 + max(max(b) for b in block_ids) + 1
    width = 64
    table = torch.zeros(TILE, width, dtype=torch.int32)
    pos = torch.full((TILE,), -1, dtype=torch.int32)
    for u in range(users):
        pos[u] = positions[u]
        table[u, : len(block_ids[u])] = torch.tensor(block_ids[u], dtype=torch.int32)
    k_cache = torch.randn(blocks, 1, BLOCK, DK)
    v_cache = torch.randn(blocks, 1, BLOCK, DV)
    cos = torch.randn(1, 1, TILE, DK)
    sin = torch.randn(1, 1, TILE, DK)

    def dev(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(t, device=device, dtype=dtype, layout=layout, memory_config=mc)

    return {
        "table": dev(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        "pos": dev(pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        "k": k_cache,
        "v": v_cache,
        "cos": dev(cos),
        "sin": dev(sin),
        "dev": dev,
    }


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_swa_qkv_writes_finite_rows(mesh_device):
    plan = MeshPlan(mesh_device)
    users = 4
    # A verify's slots: one request at four consecutive positions in one cache tile (one writer), then another request.
    positions = [130, 131, 132, 200]
    block_ids = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [4, 5, 6, 7]]
    inp = _swa_inputs(mesh_device, users, positions, block_ids, 2)
    torch.manual_seed(3)
    qkv = torch.randn(1, 1, TILE, (NH + 2) * DK)
    sample = inp["dev"](qkv, mc=ttnn.L1_MEMORY_CONFIG)
    op = SwaDecodeQkv(plan, NH, DK, users, BLOCK, torch.eye(TILE), sample)

    def run(x):
        k_cache = inp["dev"](inp["k"], ttnn.bfloat8_b)
        v_cache = inp["dev"](inp["v"], ttnn.bfloat8_b)
        op(inp["dev"](x, mc=ttnn.L1_MEMORY_CONFIG), inp["cos"], inp["sin"], inp["pos"], inp["table"], k_cache, v_cache)
        return [ttnn.to_torch(ttnn.get_device_tensors(c)[0]).float() for c in (k_cache, v_cache)]

    clean_k, clean_v = run(qkv)
    bad = qkv.clone()
    bad[0, 0, 1, NH * DK : (NH + 1) * DK] = float("nan")  # user 1's new key
    bad[0, 0, 2, (NH + 1) * DK : (NH + 1) * DK + DV] = float("inf")  # user 2's new value
    bad[0, 0, 3, NH * DK : NH * DK + 4] = 1e38
    got_k, got_v = run(bad)
    for got in (got_k, got_v):
        assert torch.isfinite(got).all(), "the cache holds a non-finite value"
        # The writer clamps to KV_LIMIT in bf16; packing the tile to BFP8 can round that up to 2^16.
        assert got.abs().max() <= 2.0**16
    written = torch.zeros(got_k.shape[0], BLOCK, dtype=torch.bool)
    for u, p in enumerate(positions):
        b, r = block_ids[u][p // BLOCK], p % BLOCK
        written[b, r] = True
        same_k = torch.equal(got_k[b, 0, r], clean_k[b, 0, r])
        same_v = torch.equal(got_v[b, 0, r], clean_v[b, 0, r])
        assert same_k == (u not in (1, 3)), f"user {u}'s key row"
        assert same_v == (u != 2), f"user {u}'s value row"
    for got, clean in ((got_k, clean_k), (got_v, clean_v)):
        assert torch.equal(got[:, 0][~written], clean[:, 0][~written]), "a row no user writes changed"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("where", ["past_position", "null_block"])
@pytest.mark.parametrize("value", [1.0, 30.0, KV_LIMIT])
def test_swa_sdpa_ignores_finite_masked_rows(mesh_device, where, value):
    users, width = 4, 128
    torch.manual_seed(4)
    blocks = 2 + users * width
    k = torch.randn(blocks, 1, BLOCK, DK) * 2
    v = torch.randn(blocks, 1, BLOCK, DV)
    table = torch.zeros(TILE, width, dtype=torch.int32)
    pos = torch.full((TILE,), -1, dtype=torch.int32)
    for u in range(users):
        p = 3010 + 37 * u
        pos[u] = p
        table[u] = 2 + u * width + torch.arange(width, dtype=torch.int32)
        table[u, : max(0, p - WIN + 1) // BLOCK] = 0  # vLLM's SWA table: no window reads these, the null block
    kk, vv = k.clone(), v.clone()
    p = int(pos[0])
    if where == "past_position":
        b, r0 = int(table[0, p // BLOCK]), p % BLOCK + 1
        rows = (slice(b, b + 1), 0, slice(r0, BLOCK))
    else:
        rows = (slice(0, 1), 0, slice(0, BLOCK))
    kk[rows] = torch.where(torch.rand_like(kk[rows]) < 0.5, -1.0, 1.0) * value
    vv[rows] = torch.where(torch.rand_like(vv[rows]) < 0.5, -1.0, 1.0) * value
    qkv = torch.randn(1, 1, TILE, (NH + 2) * DK)
    sinks = torch.zeros(NH, TILE)
    sinks[:, 0] = torch.randn(NH)

    def sdpa(kc, vc):
        x = ttnn.from_torch(
            qkv, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        q, _k, _v = ttnn.experimental.nlp_create_qkv_heads_decode(
            x, num_heads=NH, num_kv_heads=1, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        dev = dict(device=mesh_device)
        cfg = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        out = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q,
            ttnn.from_torch(kc, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, **dev),
            ttnn.from_torch(vc, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, **dev),
            head_dim_v=DV,
            page_table_tensor=ttnn.from_torch(table, dtype=ttnn.int32, **dev),
            cur_pos_tensor=ttnn.from_torch(pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, **dev),
            scale=DK**-0.5,
            sliding_window_size=WIN,
            attention_sink=ttnn.from_torch(sinks, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, **dev),
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                q_chunk_size=0,
                k_chunk_size=128,
                exp_approx_mode=False,
            ),
            compute_kernel_config=cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()

    assert torch.equal(sdpa(kk, vv), sdpa(k, v))
