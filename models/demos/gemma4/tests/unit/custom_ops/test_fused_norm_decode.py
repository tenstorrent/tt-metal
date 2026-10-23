# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""FusedNormDecode (custom op: N cores exchanging partial row sums, FPU broadcasts, residual / two-input /
multi-output forms) vs the torch golden and ttnn.rms_norm, single chip."""

import os

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.fused_norm_decode import FusedNormDecode

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
EPS = 1e-6


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _dev(t, device, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=mem)


def _gamma_full(g, device):
    row = _dev(g.reshape(1, 1, 1, -1), device)
    return FusedNormDecode.full_gamma(row)


CASES = {
    # name: (R, W, n_out, gammas (bool per out), scalars, residual, two_in, mem, cores)
    "single_b32": (32, 2816, 1, [True], [1.0], False, False, "dram", 22),
    "single_b1": (1, 2816, 1, [True], [1.0], False, False, "dram", 22),
    "single_44cores": (32, 2816, 1, [True], [1.0], False, False, "dram", 44),
    "single_11cores": (32, 2816, 1, [True], [1.0], False, False, "dram", 11),
    "router_scalar": (32, 2816, 1, [True], [2816**-0.5], False, False, "dram", 22),
    "no_gamma": (32, 2816, 1, [False], [1.0], False, False, "dram", 22),
    "residual": (32, 2816, 1, [True], [1.0], True, False, "dram", 22),
    "residual_scalar": (32, 2816, 1, [True], [0.7265625], True, False, "dram", 22),
    "two_inputs": (32, 2816, 1, [True], [1.0], False, True, "dram", 22),
    "two_inputs_44cores": (32, 2816, 1, [True], [1.0], False, True, "dram", 44),
    "three_outputs": (32, 2816, 3, [True, True, False], [1.0, 1.0, 2816**-0.5], False, False, "dram", 22),
    "single_l1": (32, 2816, 1, [True], [1.0], False, False, "l1", 22),
    "w1024_b8": (8, 1024, 1, [True], [1.0], False, False, "dram", 22),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_fused_norm_decode(mesh_device, device_params, case):
    R, W, n_out, has_gamma, scalars, with_res, two_in, mem, cores = CASES[case]
    os.environ["GEMMA4_FUSED_NORM_CORES"] = str(cores)
    torch.manual_seed(0)
    x = (torch.randn(1, 1, R, W) * 4).to(torch.bfloat16)
    x2 = (torch.randn(1, 1, R, W) * 2).to(torch.bfloat16) if two_in else None
    res = (torch.randn(1, 1, R, W) * 3).to(torch.bfloat16) if with_res else None
    gs = [(torch.rand(W) + 0.5).to(torch.bfloat16) if hg else None for hg in has_gamma]
    g2 = (torch.rand(W) + 0.5).to(torch.bfloat16) if two_in else None
    golden = FusedNormDecode.golden(x, gs, EPS, x2=x2, gamma2=g2, residual=res, scalars=scalars)
    memcfg = ttnn.DRAM_MEMORY_CONFIG if mem == "dram" else ttnn.L1_MEMORY_CONFIG
    tx = _dev(x, mesh_device, memcfg)
    tx2 = _dev(x2, mesh_device, memcfg) if two_in else None
    tres = _dev(res, mesh_device, memcfg) if with_res else None
    tgs = [_gamma_full(g, mesh_device) if g is not None else None for g in gs]
    tg2 = _gamma_full(g2, mesh_device) if g2 is not None else None
    N = FusedNormDecode.num_cores(W // 32, 2 if two_in else 1)
    outs = FusedNormDecode.op(tx, tgs, EPS, x2=tx2, gamma2=tg2, residual=tres, scalars=scalars)
    got = [ttnn.to_torch(o).float() for o in outs]
    assert len(got) == n_out
    # the shipped single norm as a second reference
    if n_out == 1 and not two_in and not with_res and gs[0] is not None:
        g_rm = ttnn.from_torch(
            gs[0].reshape(1, 1, -1, 32), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
        )
        ref = ttnn.rms_norm(tx, weight=g_rm, epsilon=EPS)
        if scalars[0] != 1.0:
            ref = ttnn.mul(ref, scalars[0])
        ref = ttnn.to_torch(ref).float()
        print(f"[{case} N={N}] pcc vs ttnn.rms_norm {_pcc(ref, got[0]):.6f} max|diff| {(ref - got[0]).abs().max():.4f}")
    for o, (g, out) in enumerate(zip(golden, got)):
        assert torch.isfinite(out).all(), (case, o)
        cols = [
            round(_pcc(g[..., i * 32 : (i + 1) * 32], out[..., i * 32 : (i + 1) * 32]), 4)
            for i in range(0, W // 32, max(1, W // 32 // 8))
        ]
        p = _pcc(g, out)
        print(
            f"[{case} N={N} out{o}] pcc vs golden {p:.6f} max|diff| {(g - out).abs().max():.4f} (|g| max {g.abs().max():.2f}); sampled column tiles {cols}"
        )
    for o, (g, out) in enumerate(zip(golden, got)):
        assert _pcc(g, out) > 0.9999, (case, o, _pcc(g, out))
        assert (g - out).abs().max() < 0.05 * g.abs().max() + 0.02, (case, o, (g - out).abs().max())
    assert all(o.memory_config() == tx.memory_config() for o in outs)
    # program-cache hits on fresh buffers, twice (the semaphore must be reset between runs)
    for _ in range(2):
        tx_b = _dev(x, mesh_device, memcfg)
        tx2_b = _dev(x2, mesh_device, memcfg) if two_in else None
        tres_b = _dev(res, mesh_device, memcfg) if with_res else None
        outs_b = FusedNormDecode.op(tx_b, tgs, EPS, x2=tx2_b, gamma2=tg2, residual=tres_b, scalars=scalars)
        for o, (a, b) in enumerate(zip(got, outs_b)):
            b = ttnn.to_torch(b).float()
            assert torch.equal(a, b), f"{case} out{o}: cache-hit call differs: max|diff| {(a - b).abs().max()}"
