# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regime-, plan- and rule-pinned tests of softmax (op_design.md, Work Distribution, "Tests the plan needs" and
"Required regime-pinned tests"). The device comes from the module-scoped conftest fixture."""

import pytest
import torch

import ttnn
from ttnn.operations.softmax import softmax
from ttnn.operations.softmax import softmax_program_artifacts as artifacts

PCC_MIN, REL_RMS_MAX = 0.99, 0.05  # NUMERIC["legacy_tolerances"][(bfloat16, True)]


def _put(t, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=memory_config)


def _check(got, t, dim=-2):
    ref = torch.softmax(t.double(), dim=dim)
    got = got.double()
    assert got.shape == ref.shape
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    rel_rms = ((got - ref).pow(2).mean().sqrt() / ref.std()).item()
    assert pcc >= PCC_MIN and rel_rms <= REL_RMS_MAX, f"pcc {pcc} rel_rms {rel_rms}"
    # Every column of every image must sum to 1: a column left unwritten or computed by the wrong NEO fails here.
    sums = got.sum(dim=dim)
    assert torch.allclose(sums, torch.ones_like(sums), atol=0.05), (sums - 1).abs().max()


def _randn(shape, seed, scale=1.0):
    torch.manual_seed(seed)
    return (torch.randn(shape, dtype=torch.float32) * scale).to(torch.bfloat16)


# (a) multi-block with a ragged last block: [1,4,16,8400] has 9 or 8 rounds per cluster; capped at 2 columns per
# block that is blocks 2,2,2,2,1 (in_depth = 2), so the input ring holds two blocks and the last block is short.
def test_multi_block_ragged_last_block(device, monkeypatch):
    monkeypatch.setattr(artifacts, "BLOCK_COLS_CAP", 2)
    assert artifacts.select_blocking(9, 1, 2048, 4) == (2, artifacts.IN_DEPTH_MULTI)
    t = _randn((1, 4, 16, 8400), 1)
    _check(ttnn.to_torch(softmax(_put(t, device), dim=-2)), t)


@pytest.mark.parametrize("cap", [1, 3])
def test_block_cols_knob(device, monkeypatch, cap):
    """block_cols = 1 (one column per block) and 3 (2 full blocks + a short one on 8 rounds) on a masked, Ht = 2
    shape: the knob is live and every value gives the same result."""
    monkeypatch.setattr(artifacts, "BLOCK_COLS_CAP", cap)
    t = _randn((2, 8, 50, 1024), 2)  # 16 images x 32 columns = 512 columns -> 128 rounds -> 4 per cluster
    _check(ttnn.to_torch(softmax(_put(t, device), dim=-2)), t)


# (b) the tall resident column: Ht = 64, 8 columns on 2 clusters; non-monotonic data along H exercises the max fold.
def test_tall_resident_column(device):
    t = _randn((1, 1, 2048, 256), 3)
    _check(ttnn.to_torch(softmax(_put(t, device), dim=-2)), t)


def test_max_fold_non_monotonic(device):
    """A hand-built column whose max sits in the middle tile-row: a fold that kept op(first, last) fails."""
    t = torch.full((1, 1, 96, 32), -5.0)
    t[0, 0, 40, :] = 4.0  # middle tile-row holds the max
    t[0, 0, 0, :] = 1.0
    t[0, 0, 95, :] = 2.0
    t = t.to(torch.bfloat16)
    _check(ttnn.to_torch(softmax(_put(t, device), dim=-2)), t)


# (c) the refusal: a column of 128 tiles does not fit one ring at C = 4; refused before any dispatch.
def test_too_tall_column_is_refused(device):
    x = _put(torch.zeros((1, 1, 4096, 32), dtype=torch.bfloat16), device)
    with pytest.raises(NotImplementedError):
        softmax(x, dim=-2)


# (d) the H-mask under approximation (rule R5): LoFi with approximation on, the model's L1 call, at scale 1 and 10.
@pytest.mark.parametrize("scale", [1.0, 10.0])
def test_lofi_approx_masked(device, scale):
    t = _randn((1, 4, 16, 400), 4, scale)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=True)
    out = softmax(_put(t, device, ttnn.L1_MEMORY_CONFIG), dim=2, compute_kernel_config=cfg)
    _check(ttnn.to_torch(out), t)


def test_large_magnitude_masked(device):
    """Rule R1: scale-10 inputs need the max subtraction; the padding rows (H = 16 of 32) must not enter it."""
    t = _randn((1, 4, 16, 400), 42, 10.0)
    _check(ttnn.to_torch(softmax(_put(t, device), dim=-2)), t)


# Thread plan (1): every NEO ran. Two calls in one session on a shape whose columns cover all 4 NEOs of a cluster
# ([1,4,16,400]: 52 columns, 13 clusters x 4 NEOs x 1 column); a NEO that did not run would leave the first call's
# values in its columns of the second output (the allocator reuses the freed buffer).
def test_every_neo_ran(device):
    spec, _, _ = artifacts.create_program_artifacts(
        _put(torch.zeros((1, 4, 16, 400), dtype=torch.bfloat16), device),
        _put(torch.zeros((1, 4, 16, 400), dtype=torch.bfloat16), device),
        ttnn.operations.softmax.default_compute_kernel_config(),
    )
    threads = {k.unique_id: k.num_threads for k in spec.kernels}
    assert threads == {"reader": 2, "compute": 4, "writer": 2}, threads

    first = _randn((1, 4, 16, 400), 5)
    out1 = softmax(_put(first, device), dim=-2)
    _check(ttnn.to_torch(out1), first)
    ttnn.deallocate(out1)
    second = _randn((1, 4, 16, 400), 6, 3.0)
    _check(ttnn.to_torch(softmax(_put(second, device), dim=-2)), second)


@pytest.mark.parametrize("plan", [(1, 1, 1), (1, 2, 1), (2, 4, 2), (4, 4, 2)], ids=lambda p: "R%dC%dW%d" % p)
def test_thread_plans(device, monkeypatch, plan):
    """The plan is one constant; the kernels size everything from its counts. Each plan must give the same result,
    including a shape with a padded last round (52 columns: not a multiple of C on the last cluster for C = 4 when
    rounds are dealt; 263 columns below)."""
    monkeypatch.setattr(artifacts, "THREAD_PLAN", plan)
    t = _randn((1, 1, 50, 263 * 32 - 7), 7)  # 263 columns, Ht = 2, masked, ragged W
    _check(ttnn.to_torch(softmax(_put(t, device), dim=-2)), t)


# Thread plan (2): one device session: the op after another program, and a program of tt-metal's own after it.
def test_device_session_neighbours(device):
    # A program of tt-metal's own before ours: a device-side tilize (ttnn.add is Metal 1 and does not run on Quasar).
    t = _randn((1, 4, 16, 400), 10)
    x_rm = ttnn.from_torch(t, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
    x = ttnn.to_layout(x_rm, ttnn.TILE_LAYOUT)

    out = softmax(x, dim=-2)
    rm = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)))
    _check(rm, t)

    t2 = _randn((1, 1, 64, 128), 11)
    _check(ttnn.to_torch(softmax(_put(t2, device), dim=-2)), t2)


# Rules R2, R8, R10 and rejection X1.
def test_positive_dim_alias(device):
    t = _randn((1, 1, 64, 128), 12)
    _check(ttnn.to_torch(softmax(_put(t, device), dim=2)), t)


@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["DRAM", "L1"])
def test_output_in_input_buffer(device, memory_config):
    t = _randn((1, 4, 16, 400), 13)
    out = softmax(_put(t, device, memory_config), dim=-2)
    assert out.memory_config().buffer_type == memory_config.buffer_type
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert out.dtype == ttnn.bfloat16 and out.layout == ttnn.TILE_LAYOUT
    _check(ttnn.to_torch(out), t)


def test_device_compute_kernel_config(device):
    """Rule R10: tt-metal's device config (what YOLOv8's small-grid test passes) is taken as is."""
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    t = _randn((1, 4, 16, 400), 14)
    _check(ttnn.to_torch(softmax(_put(t, device, ttnn.L1_MEMORY_CONFIG), dim=2, compute_kernel_config=cfg)), t)


@pytest.mark.parametrize("dim", [0, 1, -3, -4])
def test_rejects_non_reducible_dim(device, dim):
    """X1: an in-range dim that canonicalises outside {-1, -2} is refused with the registry's typed refusal (a
    NotImplementedError, one of the types X1 allows): tt-metal's translated tests call dim=-3 / dim=0 and the
    harness counts anything else as a wrong-mode failure."""
    from ttnn.operations._op_contract import SupportRefusal

    with pytest.raises(SupportRefusal):
        softmax(_put(torch.zeros((1, 1, 32, 64), dtype=torch.bfloat16), device), dim=dim)


@pytest.mark.parametrize("dim", [4, -5])
def test_rejects_out_of_range_dim(device, dim):
    with pytest.raises(ValueError):
        softmax(_put(torch.zeros((1, 1, 32, 64), dtype=torch.bfloat16), device), dim=dim)


def test_refuses_outside_phase0(device):
    x = _put(torch.zeros((1, 1, 32, 64), dtype=torch.bfloat16), device)
    with pytest.raises(NotImplementedError):
        softmax(x)  # dim=-1, the signature's default: a refinement
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False)
    with pytest.raises(NotImplementedError):
        softmax(x, dim=-2, compute_kernel_config=cfg)
    xf = ttnn.from_torch(torch.zeros((1, 1, 32, 64)), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    with pytest.raises(NotImplementedError):
        softmax(xf, dim=-2)
