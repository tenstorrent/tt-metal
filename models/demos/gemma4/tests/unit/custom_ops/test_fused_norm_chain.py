# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""FusedNormChainDecode (custom op: post-attention norm + residual [* scalar] then the pre-FF norm as one program)
vs the two FusedNormDecode programs and the torch golden, single chip, at the dense hidden sizes."""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.custom_ops.fused_norm_chain import FusedNormChainDecode
from models.demos.gemma4.custom_ops.fused_norm_decode import FusedNormDecode

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
EPS = 1e-6
# The chain packs h to bf16 before the second norm, as the two programs round it, so h is bit-identical everywhere.
# y is bit-identical on the Galaxy's unharvested 13x10 P150 grid (the device grid and the forced 8x8 split alike);
# the scalar case (stage A ends with the SFPU scalar multiply, then stage B's squares / reduce / rsqrt run in the
# same kernel) lands y a bf16 ulp or two off the two programs on a few elements elsewhere: CI wh_llmbox
# (31b_scalar_l1: max |diff| 0.0156, PCC vs golden 0.999993, two ulps where |y| < 2) and CI bh_quietbox_2's
# harvested P300c (31b_scalar_l1 under its device grid: max |diff| 0.0312, PCC vs golden 0.999999; its forced 8x8
# split is exact). The planner split is the same at W 5376 on all of them (plan_norm_cores: 21 cores); the
# differing elements are rounding ties of the row scale that fall with the core placement. The bar is therefore
# the golden PCC plus two bf16 ulps of the value (never below two ulps of 1.0) on every arch and grid, and
# bit-exactness is asserted only where it was measured: Blackhole on the full 13x10 grid. The print line carries
# the worst element in ulps so a CI log says what the hardware did.
ULPS = 2
BIT_EXACT_GRID = (13, 10)


def _bf16_ulp(r):
    # one bf16 ulp of r's binade (7 explicit mantissa bits), floored at the ulp of 1.0
    return torch.exp2(torch.floor(torch.log2(r.abs().clamp_min(1.0))) - 7)


def _worst_ulps(o, r):
    d = (o - r).abs() / _bf16_ulp(r)
    i = int(d.argmax())
    return float(d.reshape(-1)[i]), float(r.reshape(-1)[i]), float((o - r).abs().reshape(-1)[i])


def _bit_exact(mesh_device):
    g = mesh_device.compute_with_storage_grid_size()
    return is_blackhole() and (int(g.x), int(g.y)) == BIT_EXACT_GRID


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _dev(t, device, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=mem)


@pytest.mark.parametrize(
    "R, W, scalar, mem",
    [
        (32, 5376, 1.0, "dram"),
        (1, 5376, 1.0, "dram"),
        (32, 3840, 1.0, "dram"),
        (1, 3840, 1.0, "dram"),
        (32, 5376, 0.90625, "l1"),
        (8, 3840, 1.1015625, "dram"),
        (32, 2816, 1.0, "dram"),
    ],
    ids=["31b_b32", "31b_b1", "12b_b32", "12b_b1", "31b_scalar_l1", "12b_b8_scalar", "26b_b32"],
)
@pytest.mark.parametrize("grid", [None, "8x8"], ids=["device_grid", "grid8x8"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_fused_norm_chain_matches_two_programs(mesh_device, device_params, R, W, scalar, mem, grid, monkeypatch):
    if grid is not None:
        if (R, W) not in ((32, 5376), (32, 3840)):
            pytest.skip("one 8x8 case per width")
        monkeypatch.setenv("GEMMA4_TEST_GRID", grid)  # the Wormhole-sized core split on the same chip
    torch.manual_seed(0)
    x = (torch.randn(1, 1, R, W) * 4).to(torch.bfloat16)
    res = (torch.randn(1, 1, R, W) * 3).to(torch.bfloat16)
    g1 = (torch.rand(W) + 0.5).to(torch.bfloat16)
    g2 = (torch.rand(W) + 0.5).to(torch.bfloat16)
    h_gold, y_gold = FusedNormChainDecode.golden(x, res, g1, g2, EPS, scalar)
    memcfg = ttnn.DRAM_MEMORY_CONFIG if mem == "dram" else ttnn.L1_MEMORY_CONFIG
    tx, tres = _dev(x, mesh_device, memcfg), _dev(res, mesh_device, memcfg)
    tg1 = FusedNormDecode.full_gamma(_dev(g1.reshape(1, 1, 1, W), mesh_device))
    tg2 = FusedNormDecode.full_gamma(_dev(g2.reshape(1, 1, 1, W), mesh_device))
    # the two-program chain
    h_ref = FusedNormDecode.op(tx, [tg1], EPS, residual=tres, scalars=[scalar])[0]
    y_ref = FusedNormDecode.op(h_ref, [tg2], EPS)[0]
    h_ref_t, y_ref_t = ttnn.to_torch(h_ref).float(), ttnn.to_torch(y_ref).float()
    tx2, tres2 = _dev(x, mesh_device, memcfg), _dev(res, mesh_device, memcfg)
    h, y = FusedNormChainDecode.op(tx2, tres2, tg1, tg2, EPS, scalar)
    h_t, y_t = ttnn.to_torch(h).float(), ttnn.to_torch(y).float()
    for name, g, r, o in (("h", h_gold, h_ref_t, h_t), ("y", y_gold, y_ref_t, y_t)):
        assert torch.isfinite(o).all(), name
        p = _pcc(g, o)
        ulps, at, diff = _worst_ulps(o, r)
        print(
            f"[R{R} W{W} s{scalar} {mem} {name}] pcc vs golden {p:.6f} max|diff| vs two programs "
            f"{(r - o).abs().max():.4f} worst {ulps:.1f} ulp (|diff| {diff:.4f} at value {at:.4f})"
        )
        assert p > 0.9999, (name, p)
        assert ulps <= ULPS, f"{name} differs from the two FusedNormDecode programs by {ulps:.1f} bf16 ulps at {at:.4f}"
        if _bit_exact(mesh_device):
            assert torch.equal(o, r), f"{name} differs from the two FusedNormDecode programs on the 13x10 grid"
    assert h.memory_config() == tx2.memory_config() and y.memory_config() == tx2.memory_config()
    # program-cache hits on fresh buffers, twice (the semaphore must be reset between runs)
    for _ in range(2):
        hb, yb = FusedNormChainDecode.op(
            _dev(x, mesh_device, memcfg), _dev(res, mesh_device, memcfg), tg1, tg2, EPS, scalar
        )
        assert torch.equal(ttnn.to_torch(hb).float(), h_t) and torch.equal(ttnn.to_torch(yb).float(), y_t)
