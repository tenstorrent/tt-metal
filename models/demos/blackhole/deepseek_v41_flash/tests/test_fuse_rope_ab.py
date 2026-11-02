# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which arithmetic does the unfused rope_pe chain use?  Mismatch counts of torch candidates and of the fused kernel modes 0/1 against ttnn rope_pe on random bf16 data (one chip)."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import fuse_attn as FA
from models.demos.blackhole.deepseek_v41_flash.tt import pf_tune

U, C, NH, D = 2, 512, 8, 512


def swap(t):
    s = torch.zeros_like(t)
    s[..., 0::2] = -t[..., 1::2]
    s[..., 1::2] = t[..., 0::2]
    return s


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param({"l1_small_size": 16384}, id="p")], indirect=True)
@torch.no_grad()
def test_rope_ab(mesh_device):
    md = mesh_device
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t: ttnn.from_torch(
        t.to(torch.bfloat16).contiguous(),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    down = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])
    P = pf_tune.p64(md)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    torch.manual_seed(5)
    pos = torch.arange(U * C).float().view(U, 1, C, 1)
    ang = pos * (1.0 / (10000.0 ** (torch.arange(32).float() / 32))).view(1, 1, 1, 32)
    cos, sin = ang.cos().repeat_interleave(2, -1), ang.sin().repeat_interleave(2, -1)
    c = torch.ones(U, 1, C, D)
    s = torch.zeros(U, 1, C, D)
    c[..., 448:], s[..., 448:] = cos, sin
    tc, ts = up(c), up(s)
    q = (torch.randn(1, 1, U * C, NH * D) * 3).to(torch.bfloat16)
    tq = up(q)
    qh = ttnn.experimental.nlp_create_qkv_heads(
        ttnn.reshape(tq, [U, 1, C, NH * D]), num_heads=NH, num_kv_heads=0, transpose_k_heads=False
    )[0]
    ref = down(pf_tune.rope_pe(qh, tc, ts, P, ckc)).float()
    x = down(qh).float()
    xp = x[..., 448:]
    cb, sb = tc and c.to(torch.bfloat16).float()[..., 448:], s.to(torch.bfloat16).float()[..., 448:]
    bf = lambda t: t.to(torch.bfloat16).float()
    cand = {
        "A two roundings bf(bf(x*c)+bf(p*s))": bf(bf(xp * cb) + bf(swap(xp) * sb)),
        "B single rounding bf(x*c+p*s)": bf(xp * cb + swap(xp) * sb),
        "C bf(bf(x*c)+p*s)": bf(bf(xp * cb) + swap(xp) * sb),
    }
    r = ref[..., 448:]
    n = r.numel()
    for k, v in cand.items():
        print(f"ROPEAB torch {k}: mismatches {int((v != r).sum())} of {n}", flush=True)
    for mode, fp in ((0, "1"), (6, "1")):
        os.environ["DSV41_FA_FP32"] = fp
        os.environ["DSV41_FA_ROPE_MODE"] = str(mode)
        got = down(FA.qprep(tq, (tc, ts, None), P, U, C, NH, sparse=False)).float()
        print(
            f"ROPEAB fused mode {mode} fp32 {fp}: mismatches {int((got[..., 448:] != r).sum())} of {n}; non-rope columns equal {torch.equal(got[..., :448], ref[..., :448])}",
            flush=True,
        )
