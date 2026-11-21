# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests of rms_norm's thread plan and blocking knobs (op_design.md, Work Distribution).

- regime-pinned: one shape per C_eff (1, 2 by row count, 2 by the ring cap, 4), a padded round, several
  rounds per cluster, gamma blocks > 1;
- threads ran: the test-only hook makes NEO t multiply its output by t + 1 (a correct result alone would not
  show that every NEO ran);
- sessions: the op after another program, and before one of tt-metal's own (`to_layout` to row-major);
- the `rows_per_block` knob turned to 2 (a ragged last unit included).
"""

import pytest
import torch

import ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_artifacts as artifacts
from ttnn.operations.rms_norm.rms_norm import default_compute_kernel_config


def reference(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    xd = x.double()
    w = xd.shape[-1]
    return xd * torch.rsqrt(xd.pow(2).mean(dim=-1, keepdim=True) + eps) * gamma.double().reshape(-1)[:w]


def make_inputs(device, shape, seed=0, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    torch.manual_seed(seed)
    w = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn((1, 1, -(-w // 32), 32), dtype=torch.bfloat16)
    tx = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=memory_config)
    tg = ttnn.from_torch(g, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
    return x, g, tx, tg


def check(got: torch.Tensor, expected: torch.Tensor):
    got, expected = got.double(), expected.double()
    pcc = torch.corrcoef(torch.stack([got.flatten(), expected.flatten()]))[0, 1].item()
    err = (got - expected).abs()
    assert pcc > 0.999, f"pcc {pcc}"
    assert bool((err <= 0.052 + 0.039 * expected.abs()).all()), f"max abs err {err.max().item()}"


def plan_for(device, shape):
    x = ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    return artifacts.select_regime(x, default_compute_kernel_config())


# (shape, expected C_eff): op_design.md, Work Distribution, test (3).
REGIME_CASES = [
    ((1, 1, 32, 2048), 1),  # one tile-row; Wt = 64, four gamma blocks of 16
    ((1, 1, 64, 128), 2),  # two tile-rows
    ((1, 1, 128, 4096), 2),  # four tile-rows, but Wt = 128 at C = 4 is over the ring cap
    ((2, 4, 128, 512), 4),  # 32 tile-rows
    ((1, 1, 96, 128), 2),  # three tile-rows: one padded round (a recomputed row)
    ((1, 3, 1600, 64), 4),  # 150 tile-rows: 38 rounds over 32 clusters (two rounds on some), a padded round
    ((1, 1, 32, 8192), 1),  # Wt = 256: the widest PHASE0 row, C = 1 by the ring cap
]


@pytest.mark.parametrize("shape,c_eff", REGIME_CASES, ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v))
def test_regime_pinned(device, shape, c_eff):
    assert plan_for(device, shape)["C"] == c_eff
    x, g, tx, tg = make_inputs(device, shape)
    got = ttnn.to_torch(rms_norm(tx, weight=tg, epsilon=1e-5)).float()
    check(got, reference(x, g, 1e-5))


def test_l1_input_multi_neo(device):
    """An L1 input with C_eff = 4; the output lands in L1 (rule R11)."""
    shape = (2, 4, 128, 512)
    x, g, tx, tg = make_inputs(device, shape, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = rms_norm(tx, weight=tg, epsilon=1e-6)
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    check(ttnn.to_torch(out).float(), reference(x, g, 1e-6))


def test_each_neo_computes_its_own_rows(device, monkeypatch):
    """NEO t of C_eff = 4 multiplies its rows by t + 1: tile-row unit_start + k*4 + t must carry factor t + 1.

    (1, 1, 2048, 256): 64 tile-rows, 16 rounds dealt one per cluster, no padded round (a recomputed row would
    be written by two NEOs with different factors)."""
    shape = (1, 1, 2048, 256)
    assert plan_for(device, shape)["C"] == 4
    monkeypatch.setattr(artifacts, "_TEST_TAG_THREADS", True)
    x, g, tx, tg = make_inputs(device, shape)
    got = ttnn.to_torch(rms_norm(tx, weight=tg, epsilon=1e-5)).float()
    expected = reference(x, g, 1e-5)
    factor = torch.ones(shape, dtype=torch.float64)
    for tile_row in range(shape[-2] // 32):
        factor[..., tile_row * 32 : (tile_row + 1) * 32, :] = tile_row % 4 + 1  # unit_start is a multiple of 4
    check(got, expected * factor)


@pytest.mark.parametrize("before", ["quasar_add", "quasar_softmax", "rms_norm"])
def test_after_another_program(device, before):
    """Our program after another program in one device session: tt-metal's Quasar add and softmax (implicit
    sync, tagged transfers) and rms_norm itself."""
    shape = (2, 4, 128, 512)
    x, g, tx, tg = make_inputs(device, shape)
    if before == "quasar_add":
        ttnn.experimental.quasar.add(tx, tx)
    elif before == "quasar_softmax":
        ttnn.experimental.quasar.softmax(tx, dim=-1)
    else:
        rms_norm(tx, weight=tg, epsilon=1e-5)
    got = ttnn.to_torch(rms_norm(tx, weight=tg, epsilon=1e-5)).float()
    check(got, reference(x, g, 1e-5))


@pytest.mark.parametrize("shape", [(1, 1, 32, 2048), (2, 4, 128, 512)], ids=["C1", "C4"])
def test_a_tt_metal_program_follows(device, shape):
    """`to_layout` to row-major (an untilize with untagged transfers) after our program: the readers and the
    writers must have handed their tag registers back."""
    x, g, tx, tg = make_inputs(device, shape)
    out = rms_norm(tx, weight=tg, epsilon=1e-5)
    got = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT))).float()
    check(got, reference(x, g, 1e-5))


@pytest.mark.parametrize(
    "shape", [(1, 1, 96, 128), (2, 4, 128, 512), (1, 1, 17, 64)], ids=lambda s: "x".join(map(str, s))
)
def test_rows_per_block_knob(device, monkeypatch, shape):
    """The `rows_per_block` knob at 2: units of two tile-rows (DEST accumulation per row, the reduce and the
    finalize over a column of tiles, inv_rms indexed by row). (1, 1, 96, 128) has a ragged last unit."""
    monkeypatch.setattr(artifacts, "ROWS_PER_BLOCK", 2)
    x, g, tx, tg = make_inputs(device, shape)
    got = ttnn.to_torch(rms_norm(tx, weight=tg, epsilon=1e-5)).float()
    check(got, reference(x, g, 1e-5))


def test_device_compute_config_accepted(device):
    """Rule R4: tt-metal's device compute kernel config is read by attribute, not refused for its type."""
    shape = (1, 1, 64, 128)
    x, g, tx, tg = make_inputs(device, shape)
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    got = ttnn.to_torch(rms_norm(tx, weight=tg, epsilon=1e-5, compute_kernel_config=cfg)).float()
    check(got, reference(x, g, 1e-5))


@pytest.mark.parametrize("threads", [(2, 4, 2), (1, 4, 1), (1, 1, 1)], ids=lambda p: "R%dC%dW%d" % p)
@pytest.mark.parametrize("shape", [(2, 4, 128, 512), (1, 3, 1600, 64)], ids=lambda s: "x".join(map(str, s)))
def test_other_thread_plans(device, monkeypatch, threads, shape):
    """THREAD_PLAN turned: a reader hart feeding several NEOs (gamma blocks and input tiles in rotation), a
    writer hart draining four, and the single-thread plan."""
    monkeypatch.setattr(artifacts, "THREAD_PLAN", threads)
    x, g, tx, tg = make_inputs(device, shape)
    got = ttnn.to_torch(rms_norm(tx, weight=tg, epsilon=1e-5)).float()
    check(got, reference(x, g, 1e-5))
