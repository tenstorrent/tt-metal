# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""attn_fused.rope_concat (post-SDPA glue, DSV41_ATTN_POSTFUSE) vs the op chain it replaces (permute + tilize + rope_inplace +
to_memory_config + nlp_concat_heads_decode + to_memory_config): bit-exact on the token rows 0..T-1 (padding rows reported),
T = 1..16 + 24, 32, 3 seeds.  DSV41_PF_TS / DSV41_PF_SEEDS override; DSV41_PF_PROF_T=4,8,16 additionally runs each variant 4x (device profiler)."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import attn_fused as AF

NH, HD = 9, 512


def _up(md, t, layout, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t.contiguous(),
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


def _inputs(md, T, seed):
    g = torch.Generator().manual_seed(seed)
    o = torch.randn(1, 32, T, HD, generator=g).to(torch.bfloat16)
    ang = torch.rand(T, 32, generator=g) * 6.0
    cos, sin = ang.cos().repeat_interleave(2, -1), ang.sin().repeat_interleave(2, -1)
    C = torch.ones(1, T, 1, HD)
    S = torch.zeros(1, T, 1, HD)
    C[..., 448:], S[..., 448:] = cos.reshape(1, T, 1, 64), -sin.reshape(
        1, T, 1, 64
    )  # the inverse rotation: negated sin
    return (
        _up(md, o, ttnn.ROW_MAJOR_LAYOUT),
        _up(md, C.to(torch.bfloat16), ttnn.TILE_LAYOUT),
        _up(md, S.to(torch.bfloat16), ttnn.TILE_LAYOUT),
    )


def _old(md, o, c, s, T):
    if T == 1:
        x = ttnn.to_layout(ttnn.reshape(o, (1, 1, 32, HD)), ttnn.TILE_LAYOUT)
    else:
        x = ttnn.to_layout(ttnn.permute(o, (0, 2, 1, 3)), ttnn.TILE_LAYOUT)
    AF.rope_inplace(x, c, s, T)
    Tp = (
        T if (T <= 8 or T % 8 == 0) else -(-T // 8) * 8
    )  # nlp_concat_heads_decode needs a rectangular core grid (spec_attention._finish)
    if Tp != T:
        x = ttnn.pad(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), [(0, 0), (0, Tp - T), (0, 0), (0, 0)], 0.0)
    ucfg = ttnn.create_sharded_memory_config(
        shape=(32, HD),
        core_grid=ttnn.num_cores_to_corerangeset(Tp, ttnn.CoreCoord(8, 8), row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    x = ttnn.to_memory_config(x, ucfg)
    r = ttnn.experimental.nlp_concat_heads_decode(x, num_heads=NH)
    return ttnn.to_memory_config(r, ttnn.DRAM_MEMORY_CONFIG)


def _host(md, t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(32, NH * HD)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param({"l1_small_size": 16384}, id="p")], indirect=True)
@torch.no_grad()
def test_postfuse(mesh_device):
    md = mesh_device
    ts = [int(v) for v in os.environ.get("DSV41_PF_TS", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,24,32").split(",")]
    seeds = [int(v) for v in os.environ.get("DSV41_PF_SEEDS", "0,1,2").split(",")]
    prof = [int(v) for v in os.environ.get("DSV41_PF_PROF_T", "").split(",") if v]
    bad = 0
    for T in ts:
        for seed in seeds:
            o, c, s = _inputs(md, T, seed)
            want = _host(md, _old(md, o, c, s, T))
            got = _host(md, AF.rope_concat(o, c, s, T, NH))
            ok = torch.equal(want[:T].view(torch.int16), got[:T].view(torch.int16))
            pad = torch.equal(want[T:].view(torch.int16), got[T:].view(torch.int16)) if T < 32 else True
            print(f"POSTFUSE T={T} seed={seed} rows_bit_equal={ok} padding_equal={pad}", flush=True)
            if not ok:
                d = want[:T].view(torch.int16) != got[:T].view(torch.int16)
                print(
                    "  bad rows",
                    d.any(-1).nonzero().flatten().tolist(),
                    "bad col tiles",
                    sorted(set((d.any(0).nonzero().flatten() // 32).tolist()))[:20],
                    flush=True,
                )
            bad += 0 if ok else 1
    assert bad == 0
    for T in prof:
        o, c, s = _inputs(md, T, 7)
        for fn in (lambda: _old(md, o, c, s, T), lambda: AF.rope_concat(o, c, s, T, NH)):
            for _ in range(4):
                fn()
        print(f"POSTFUSE PROF T={T} done (order: 4x old chain then 4x new)", flush=True)
    if prof:
        ttnn.ReadDeviceProfiler(md)
