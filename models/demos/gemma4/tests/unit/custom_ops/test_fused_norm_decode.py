# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""FusedNormDecode (custom op: N cores exchanging partial row sums, FPU broadcasts, residual / two-input /
multi-output forms) vs the torch golden and ttnn.rms_norm, single chip."""

import os

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.fused_norm_decode import FusedNormDecode
from models.demos.gemma4.custom_ops.fused_norm_decode.op import FusedNormDecode
from models.demos.gemma4.tt.dram_sharded import DramShardedLinear, can_dram_shard

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
EPS = 1e-6


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _dev(t, device, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=mem)


def _gamma_full(g, device):
    """The op's row-replicated gamma built the model's way (tt/rms_norm.py): the ROW_MAJOR ``[1, 1, W/32, 32]``
    weight -> reshape to the ``[1, 1, 1, W]`` row -> tilize -> ``full_gamma``."""
    w = ttnn.from_torch(g.reshape(1, 1, -1, 32), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    row = ttnn.to_layout(ttnn.reshape(w, (1, 1, 1, g.numel())), ttnn.TILE_LAYOUT)
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
    # dense variants: 31B hidden 5376 (168 column tiles -> 21 cores x 8), 12B hidden 3840 (120 -> 20 x 6)
    "dense31b_single": (32, 5376, 1, [True], [1.0], False, False, "dram", 22),
    "dense31b_residual_scalar": (32, 5376, 1, [True], [0.90625], True, False, "dram", 22),
    "dense31b_b1_residual": (1, 5376, 1, [True], [1.0], True, False, "dram", 22),
    "dense31b_l1": (32, 5376, 1, [True], [1.0], False, False, "l1", 22),
    "dense12b_single": (32, 3840, 1, [True], [1.0], False, False, "dram", 22),
    "dense12b_residual_scalar": (32, 3840, 1, [True], [1.1015625], True, False, "dram", 22),
    "dense12b_b1_residual": (1, 3840, 1, [True], [1.0], True, False, "dram", 22),
    # the 26B layer's forms as the model issues them: the three-output call with a gamma AND the hidden**-0.5
    # scalar on the router output, and the residual / two-input forms at batch 1
    "three_outputs_router": (32, 2816, 3, [True, True, True], [1.0, 1.0, 2816**-0.5], False, False, "dram", 22),
    "three_outputs_router_b1": (1, 2816, 3, [True, True, True], [1.0, 1.0, 2816**-0.5], False, False, "dram", 22),
    "residual_scalar_b1": (1, 2816, 1, [True], [0.7265625], True, False, "dram", 22),
    "two_inputs_b1": (1, 2816, 1, [True], [1.0], False, True, "dram", 22),
    # prefill rows (one N-core exchange group per row tile): the 128-token layer tail's four forms on 4 x 22 cores
    "prefill_r128_residual": (128, 2816, 1, [True], [1.0], True, False, "dram", 22),
    "prefill_r128_two_outputs": (128, 2816, 2, [True, True], [1.0, 1.0], False, False, "dram", 22),
    "prefill_r128_two_inputs": (128, 2816, 1, [True], [1.0], False, True, "dram", 22),
    "prefill_r128_residual_scalar": (128, 2816, 1, [True], [0.7265625], True, False, "dram", 22),
    "prefill_r64_l1": (64, 2816, 1, [True], [1.0], False, False, "l1", 22),
    # the Wormhole split forced on Blackhole (GEMMA4_TEST_GRID=8x8 on the "wh_" cases): 11 cores x 4 row tiles in
    # one pass, the two-input form in two passes, 256 rows in two passes; and a many-pass case on the full grid
    "wh_prefill_r128_residual": (128, 2816, 1, [True], [1.0], True, False, "dram", 22),
    "wh_prefill_r128_two_inputs": (128, 2816, 1, [True], [1.0], False, True, "dram", 22),
    "wh_prefill_r256_two_outputs": (256, 2816, 2, [True, True], [1.0, 1.0], False, False, "dram", 22),
    "prefill_r1024_residual_scalar": (1024, 2816, 1, [True], [0.7265625], True, False, "dram", 22),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_fused_norm_decode(mesh_device, device_params, case, monkeypatch):
    R, W, n_out, has_gamma, scalars, with_res, two_in, mem, cores = CASES[case]
    if case.startswith("wh_"):
        monkeypatch.setenv("GEMMA4_TEST_GRID", "8x8")  # the Wormhole grid on this chip
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
    N = FusedNormDecode.plan(W // 32, 2 if two_in else 1, (R + 31) // 32, 64 if case.startswith("wh_") else 130)[0]
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


# ===== merged from tests/unit/custom_ops/test_fused_norm_plan_host.py: Host-only checks of the fused norm op's work planner (custom_ops/fused_norm_decode): the core count per row tile,
WH, BH, ONE = 8 * 8, 13 * 10, 1


COLS_26B = 2816 // 32  # 88 column tiles


@pytest.fixture()
def _default_cap(monkeypatch):
    monkeypatch.delenv("GEMMA4_FUSED_NORM_CORES", raising=False)


@pytest.mark.usefixtures("_default_cap")
def test_decode_row_keeps_the_22_core_split():
    assert FusedNormDecode.plan(COLS_26B, 1, 1, BH) == (22, 4, 1, 1)
    assert FusedNormDecode.plan(COLS_26B, 1, 1, WH) == (22, 4, 1, 1)
    assert FusedNormDecode.plan(COLS_26B, 2, 1, WH) == (22, 4, 1, 1)
    assert FusedNormDecode.num_cores(COLS_26B, 1) == 22


@pytest.mark.usefixtures("_default_cap")
def test_prefill_rows_derive_from_the_grid():
    # 128 rows (4 row tiles): Blackhole 22 cores x 4 groups in one pass, Wormhole 11 x 4 in one pass
    assert FusedNormDecode.plan(COLS_26B, 1, 4, BH) == (22, 4, 4, 1)
    assert FusedNormDecode.plan(COLS_26B, 1, 4, WH) == (11, 8, 4, 1)
    # the two-input form needs <= 4 tiles per core: 22 cores -> 2 groups per pass on Wormhole, 2 passes
    assert FusedNormDecode.plan(COLS_26B, 2, 4, WH) == (22, 4, 2, 2)
    assert FusedNormDecode.plan(COLS_26B, 2, 4, BH) == (22, 4, 4, 1)
    # 256 rows (8 row tiles): Wormhole 11 x 5 groups -> 2 passes; Blackhole 11 x 8 groups in one pass (88 cores)
    assert FusedNormDecode.plan(COLS_26B, 1, 8, WH) == (11, 8, 5, 2)
    assert FusedNormDecode.plan(COLS_26B, 1, 8, BH) == (11, 8, 8, 1)
    # 1024 rows (32 row tiles) on Blackhole: 3 passes of 11 groups x 11 cores
    assert FusedNormDecode.plan(COLS_26B, 1, 32, BH) == (11, 8, 11, 3)


@pytest.mark.usefixtures("_default_cap")
def test_degenerate_grids():
    # one core cannot hold 88 column tiles of one row in a dest window: no plan (the callers fall back)
    assert not FusedNormDecode.plan_ok(COLS_26B, 1, 4, ONE)
    raised = False  # this host test runs without the repo conftest, so no expect_error fixture
    try:
        FusedNormDecode.plan(COLS_26B, 1, 1, ONE)
    except ValueError:
        raised = True
    assert raised
    # a narrow width on one core: one row tile per pass, four passes
    assert FusedNormDecode.plan(8, 1, 4, ONE) == (1, 8, 1, 4)
    # the cap knob bounds the preferred count (fewest passes first), the grid still decides the groups
    import os

    os.environ["GEMMA4_FUSED_NORM_CORES"] = "11"
    try:
        assert FusedNormDecode.plan(COLS_26B, 1, 4, BH) == (11, 8, 4, 1)
        assert FusedNormDecode.plan(COLS_26B, 1, 1, BH) == (11, 8, 1, 1)
    finally:
        del os.environ["GEMMA4_FUSED_NORM_CORES"]
    # nothing under the cap fits: the split goes above it and, as uncapped, takes the most cores at the pass count
    os.environ["GEMMA4_FUSED_NORM_CORES"] = "4"
    try:
        assert FusedNormDecode.plan(COLS_26B, 1, 4, BH) == (22, 4, 4, 1)
    finally:
        del os.environ["GEMMA4_FUSED_NORM_CORES"]


# ===== merged from tests/unit/custom_ops/test_fused_norm_sharded_out.py: The fused decode norm writing a DRAM-sharded decode matmul's L1 activation layout (GEMMA4_NORM_SHARDED_OUT):
SINGLE_CHIP_fused_norm_sharded_out = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]


EPS_fused_norm_sharded_out = 1e-6


def _dev_fused_norm_sharded_out(t, device, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=mem)


@pytest.mark.parametrize(
    "R, W, N",
    [(32, 5376, 4096), (1, 5376, 4096), (32, 3840, 2048), (1, 3840, 2048), (32, 5376, 10752), (32, 3840, 7680)],
    ids=["31b_wqkv_b32", "31b_wqkv_b1", "12b_wqkv_b32", "12b_wqkv_b1", "31b_gateup_b32", "12b_gateup_b32"],
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP_fused_norm_sharded_out, indirect=True)
def test_fused_norm_sharded_out(mesh_device, device_params, R, W, N, monkeypatch):
    if not can_dram_shard(W, N, dtype=ttnn.bfloat8_b):
        pytest.skip(f"[{W}, {N}] has no DRAM-sharded decode config on this grid")
    monkeypatch.setenv("GEMMA4_DRAM_SHARD_ROW_VIEW", "1")
    torch.manual_seed(0)
    x = (torch.randn(1, 1, R, W) * 4).to(torch.bfloat16)
    g1 = (torch.rand(W) + 0.5).to(torch.bfloat16)
    w = (torch.randn(1, 1, W, N) * 0.02).to(torch.bfloat16)
    lin = DramShardedLinear(w, mesh_device, None, k=W, n=N, dtype=ttnn.bfloat8_b, cache_file_name=None)
    assert lin.accepts_sharded_rows()
    mem = lin.act_memory_config
    tg1 = FusedNormDecode.full_gamma(_dev_fused_norm_sharded_out(g1.reshape(1, 1, 1, W), mesh_device))
    # single form: norm -> matmul
    n_i = FusedNormDecode.op(_dev_fused_norm_sharded_out(x, mesh_device), [tg1], EPS_fused_norm_sharded_out)[0]
    n_s = FusedNormDecode.op(
        _dev_fused_norm_sharded_out(x, mesh_device), [tg1], EPS_fused_norm_sharded_out, out_memory_configs=[mem]
    )[0]
    assert n_s.memory_config() == mem and list(n_s.shape) == [1, 1, R, W]
    n_i_t, n_s_t = ttnn.to_torch(n_i).float(), ttnn.to_torch(n_s).float()
    assert torch.equal(n_s_t, n_i_t), "sharded-output norm differs from the interleaved output"
    y_i = ttnn.to_torch(lin(n_i)).float()
    y_s = ttnn.to_torch(lin(n_s)).float()
    assert list(y_s.shape) == [1, 1, R, N] and torch.equal(y_s, y_i), "matmul on the sharded norm output differs"
    print(f"[R{R} W{W} N{N}] sharded-output norms bit-identical; matmul outputs bit-identical")
