# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused attention glue vs the unfused chain on REAL activations of a prefill window (device 0 copies of q = wq_b output, kv = normed latent, the RoPE tables and the raw sparse_sdpa outputs of the first
window of a sparse layer, saved with FA_REAL_DUMP by tests/test_chunk_probe.py).  FA_REAL_DIR=<dir> FA_REAL_FIRST=<index of the q file of the layer, e.g. 5>."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import fuse_attn as FA
from models.demos.blackhole.deepseek_v41_flash.tt import pf_tune

U, C, NH, D = 2, 512, 8, 512


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param({"l1_small_size": 16384}, id="p")], indirect=True)
@torch.no_grad()
def test_real(mesh_device):
    md = mesh_device
    d = os.environ["FA_REAL_DIR"]
    i0 = int(os.environ.get("FA_REAL_FIRST", "5"))
    ld = lambda k, tag: torch.load(f"{d}/{i0 + k:04d}_{tag}.pt")
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(
        t.to(torch.bfloat16).contiguous(),
        device=md,
        dtype=ttnn.bfloat16,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    down = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    q, kvn, tc, ts = ld(0, "q"), ld(1, "kv"), ld(2, "tabc"), ld(3, "tabs")
    o_t = [ld(4, "o"), ld(5, "o")]
    print("real shapes", tuple(q.shape), tuple(kvn.shape), tuple(tc.shape), tuple(o_t[0].shape), flush=True)
    print(
        f"q absmax {q.float().abs().max():.2f} kv absmax {kvn.float().abs().max():.2f} o absmax {o_t[0].float().abs().max():.2f}",
        flush=True,
    )
    tabs = (up(tc), up(ts), up(-ts.float()))
    P = pf_tune.p64(md)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    tq, tkv = up(q), up(kvn)
    qh = ttnn.experimental.nlp_create_qkv_heads(
        ttnn.reshape(tq, [U, 1, C, NH * D]), num_heads=NH, num_kv_heads=0, transpose_k_heads=False
    )[0]
    ref = down(pf_tune.rope_pe(qh, tabs[0], tabs[1], P, ckc))
    got = down(FA.qprep(tq, tabs, P, U, C, NH, sparse=False))
    print(
        f"REAL qprep: pcc {pcc(ref, got):.7f} mismatching elements {int((ref != got).sum())} of {ref.numel()} max|d| {(ref - got).abs().max():.4g} (ref absmax {ref.abs().max():.2f})",
        flush=True,
    )
    bufs = FA.qprep(tq, tabs, P, U, C, NH, sparse=True)
    g0 = down(bufs[0])[:, :NH]
    print(f"REAL qprep sparse u0: pcc {pcc(ref[0:1], g0):.7f} mismatching {int((ref[0:1] != g0).sum())}", flush=True)
    kvr = ttnn.reshape(tkv, [U, 1, C, D])
    ref = down(pf_tune.rope_pe(kvr, tabs[0], tabs[1], P, ckc))
    got = down(FA.kv_rope(tkv, tabs, P, U, C))
    print(f"REAL kv: pcc {pcc(ref, got):.7f} mismatching {int((ref != got).sum())} of {ref.numel()}", flush=True)
    o_list = [up(o, ttnn.ROW_MAJOR_LAYOUT) for o in o_t]
    outs = [ttnn.to_layout(ttnn.slice(o, [0, 0, 0, 0], [1, NH, C, D]), ttnn.TILE_LAYOUT) for o in o_list]
    o = pf_tune.rope_pe(ttnn.concat(outs, dim=0), tabs[0], tabs[2], P, ckc)
    cc = ttnn.experimental.nlp_concat_heads(o)
    zero = ttnn.zeros([U, 1, C, D], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)
    ref = down(ttnn.reshape(ttnn.concat([zero, cc], dim=3), [1, 1, U * C, (NH + 1) * D]))
    got = down(FA.post(o_list, tabs, P, U, C, NH))
    print(f"REAL post: pcc {pcc(ref, got):.7f} mismatching {int((ref != got).sum())} of {ref.numel()}", flush=True)
