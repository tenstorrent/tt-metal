# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests the design (op_design.md) asks of the implementation beyond the acceptance cells:

- regime pinning: which regime the host chose, read off the plan it built, on a multi-block over_h shape and
  on a G > 1 shape;
- row broadcast with distinct b values per column and non-constant a rows, multi-block runs, short blocks,
  and b broadcast along leading dims;
- the thread plan: kernels carry THREAD_PLAN as num_threads and the per-cluster work sums to the total; the
  flat_stream kernels at other legal plans;
- one device session: another program before multiply, and tt-metal's to_layout after it, in both regimes.

Products of bfloat16 values are exact in fp32 DEST, so the output must equal torch's bfloat16 product bit
for bit.
"""

import pytest
import torch

import ttnn
from ttnn.operations.multiply import multiply, validate
from ttnn.operations.multiply.multiply import _dispatch
from ttnn.operations.multiply.multiply_program_artifacts import THREAD_PLAN, create_program_artifacts


def _put(t, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def _pair(a_shape, b_shape, seed=0):
    torch.manual_seed(seed)
    return torch.randn(a_shape).to(torch.bfloat16), torch.randn(b_shape).to(torch.bfloat16)


def _expect_exact(got, ta, tb):
    expected = (ta.float() * tb.float()).to(torch.bfloat16)
    got = got.to(torch.bfloat16)
    assert got.shape == expected.shape, (got.shape, expected.shape)
    mismatch = (got.float() != expected.float()).sum().item()
    assert mismatch == 0, f"{mismatch} elements differ; max diff {(got.float() - expected.float()).abs().max()}"


def _run(device, a_shape, b_shape, **options):
    ta, tb = _pair(a_shape, b_shape)
    a, b = _put(ta, device), _put(tb, device)
    axes = validate(a, b)
    out, plan = _dispatch(a, b, axes, None, **options)
    _expect_exact(ttnn.to_torch(out), ta, tb)
    return plan


@pytest.mark.parametrize(
    "a_shape,b_shape,regime,G,block_cols,blocks_rows",
    [
        # 8 images x 16 columns = 128 strips, 4 per cluster: one 4x4 block each.
        ((2, 4, 128, 512), (2, 4, 1, 512), "row_bcast_strips", 1, 4, 4),
        # 16 strips under-fill 32 clusters: H split into G = 2 row groups, b read twice.
        ((128, 512), (1, 512), "row_bcast_strips", 2, 1, 2),
        ((1, 4, 8400), (1, 1, 8400), "row_bcast_strips", 1, 9, 1),
        ((4, 8, 32, 256), (4, 8, 32, 256), "flat_stream", 1, None, None),
    ],
)
def test_regime_pinned(device, a_shape, b_shape, regime, G, block_cols, blocks_rows):
    plan = _run(device, a_shape, b_shape)
    assert plan.regime == regime
    assert plan.G == G
    if regime == "row_bcast_strips":
        assert plan.block_cols == block_cols
        assert plan.geo.ht // plan.G == blocks_rows


def test_row_bcast_distinct_columns(device):
    """b distinct in every column, a distinct in every row: a wrong b column or a wrong broadcast row shows."""
    H, W = 96, 160
    ta = (torch.arange(H).float()[:, None] / 8 + 1).expand(H, W).reshape(1, 1, H, W).to(torch.bfloat16)
    tb = (torch.arange(W).float() / 16 - 3).reshape(1, 1, 1, W).to(torch.bfloat16)
    a, b = _put(ta, device), _put(tb, device)
    _expect_exact(ttnn.to_torch(multiply(a, b)), ta, tb)


@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((4, 16, 64, 64), (4, 16, 1, 64)),  # 4 strips per cluster, two 2-column blocks: window ring wraps
        ((1, 40, 64, 96), (1, 40, 1, 96)),  # runs cut images mid-row: short blocks (cols < block_cols)
        ((2, 4, 64, 64), (1, 4, 1, 64)),  # b broadcast along dim 0 too
        ((2, 3, 64, 96), (1, 1, 1, 96)),  # b broadcast along both leading dims
        ((3, 64, 96), (96,)),  # b of rank 1
    ],
)
def test_row_bcast_blocks(device, a_shape, b_shape):
    plan = _run(device, a_shape, b_shape)
    assert plan.regime == "row_bcast_strips"


def test_thread_plan_carried_and_work_covers_output(device):
    """The program carries THREAD_PLAN on its kernels and the clusters' work adds up to the whole output."""
    for a_shape, b_shape in (((2, 4, 128, 512), (2, 4, 1, 512)), ((1, 3, 320, 320), (1, 3, 320, 320))):
        ta, tb = _pair(a_shape, b_shape)
        a, b = _put(ta, device), _put(tb, device)
        axes = validate(a, b)
        out = ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(ttnn.Shape(list(a.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.BufferType.DRAM), device
        )
        spec, _, _, plan = create_program_artifacts(
            a, b, out, broadcast=axes["broadcast"], compute_kernel_config=ttnn.ComputeConfigDescriptor()
        )
        assert [k.num_threads for k in spec.kernels] == list(THREAD_PLAN)
        assert sum(plan.counts.values()) == plan.num_units
        tiles = plan.num_units * (plan.geo.ht // plan.G if plan.regime == "row_bcast_strips" else 1)
        assert tiles == out.buffer_num_pages()


@pytest.mark.parametrize("threads", [(2, 2, 2), (4, 4, 2), (1, 4, 1)], ids=lambda p: "R%dC%dW%d" % p)
def test_flat_stream_other_plans(device, threads):
    """flat_stream deals per tile: the plan is a live knob (one constant), not a fixed (1, 1, 1)."""
    plan = _run(device, (1, 3, 320, 320), (1, 3, 320, 320), threads=threads)
    assert plan.threads == threads


def test_one_device_session(device):
    """Another program before multiply, tt-metal's untilize after it, both regimes, in one session."""
    for seed, (a_shape, b_shape) in enumerate(
        [((1, 3, 320, 320), (1, 3, 320, 320)), ((2, 4, 128, 512), (2, 4, 1, 512)), ((1, 4, 400), (1, 1, 400))]
    ):
        ta, tb = _pair(a_shape, b_shape, seed)
        # Another program first: a tilize of b on the device.
        b_rm = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        b = ttnn.to_layout(b_rm, ttnn.TILE_LAYOUT)
        a = _put(ta, device)
        out = multiply(a, b)
        # A program of tt-metal's own after multiply: an untilize with untagged transfers.
        got = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)))
        _expect_exact(got, ta, tb)


def test_output_placement_follows_first_operand(device):
    """R9: the output is interleaved in the first operand's buffer."""
    ta, tb = _pair((1, 4, 400), (1, 1, 400))
    a, b = _put(ta, device, ttnn.L1_MEMORY_CONFIG), _put(tb, device, ttnn.L1_MEMORY_CONFIG)
    out = multiply(a, b, dtype=ttnn.bfloat16)
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    _expect_exact(ttnn.to_torch(out), ta, tb)


def test_refusals(device):
    ta, tb = _pair((1, 1, 32, 64), (1, 1, 32, 50))
    a, b = _put(ta, device), _put(tb, device)
    with pytest.raises(ValueError):  # X1
        multiply(a, b)
    ta, tb = _pair((1, 1, 32, 64), (1, 1, 32, 64))
    a, b = _put(ta, device), _put(tb, device)
    with pytest.raises(NotImplementedError):  # R2
        multiply(a, b, dtype=ttnn.float32, compute_kernel_config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False))


@pytest.mark.parametrize(
    "fidelity,approx,full_sync",
    [(ttnn.MathFidelity.HiFi4, True, True), (ttnn.MathFidelity.LoFi, False, False)],
    ids=["HiFi4-approx-fullsync", "LoFi"],
)
def test_compute_config_passed_through(device, fidelity, approx, full_sync):
    """R4: any math_fidelity / math_approx_mode is accepted, and the whole config reaches the compute kernel
    (math_approx_mode -> sfpu_precision_mode, dst_full_sync_en -> single-buffered DEST)."""
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=approx, dst_full_sync_en=full_sync
    )
    for a_shape, b_shape in (((1, 1, 64, 128), (1, 1, 64, 128)), ((1, 4, 400), (1, 1, 400))):
        ta, tb = _pair(a_shape, b_shape)
        a, b = _put(ta, device), _put(tb, device)
        axes = validate(a, b, compute_kernel_config=cfg)
        out = ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(ttnn.Shape(list(a.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.BufferType.DRAM), device
        )
        spec, _, _, _ = create_program_artifacts(a, b, out, broadcast=axes["broadcast"], compute_kernel_config=cfg)
        hw = spec.kernels[1].hw_config
        assert hw.fpu_math_fidelity == fidelity
        assert hw.double_buffer_dest == (not full_sync)
        assert hw.sfpu_precision_mode == (
            ttnn.experimental.Precision.Approximate if approx else ttnn.experimental.Precision.Precise
        )
        got = ttnn.to_torch(multiply(a, b, compute_kernel_config=cfg))
        if fidelity == ttnn.MathFidelity.HiFi4:
            _expect_exact(got, ta, tb)
        else:  # LoFi drops multiplier passes: close, not exact
            expected = ta.float() * tb.float()
            assert torch.corrcoef(torch.stack([got.float().flatten(), expected.flatten()]))[0, 1] > 0.99
