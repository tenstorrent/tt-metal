# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.softmax``  (YOLOv8 detect-head DFL).

Model call site (Distribution Focal Loss decode):
  * yolov8l  ttnn_yolov8l.py:644-645
        x = ttnn.reshape(x, (b, 4, c1, a), ...)   # c1 = reg_max = 16
        x = ttnn.softmax(x, dim=2, ...)           # softmax over the reg_max axis
  * yolov8s  ttnn_yolov8s.py:507-508  (identical)

The DFL input is ``box`` = ``[1, 64, A]`` (ttnn_yolov8l.py:747, 64 = reg_max*4),
reshaped to ``[b, 4, c1=16, a=A]`` (TtDFL.__call__, c1=16 default at :642) and
softmaxed over dim=2 (the reg_max=16 axis). ``A`` = total detect anchors
(8400 @ 640, 33600 @ 1280). Detect head is identical for yolov8l/yolov8s; only
yolov8l also runs 1280.

Note: reg_max=16 is not tile-aligned; ttnn pads to 32 in TILE layout but tracks the
logical width, so the softmax reduction must respect the logical 16. Reference is
torch.softmax over the same logical shape.  PCC 0.99.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U

_REG_MAX = 16  # c1 (TtDFL.__call__ default, ttnn_yolov8l.py:642)

# (id, dfl_shape) — [b, 4, reg_max, A]; softmax dim=2. shape tuple param -> conftest
# classifies by element count (both are large -> Blackhole, not emulator).
_DFL_SITES = [
    pytest.param((1, 4, _REG_MAX, 8400), id="l_s-640-dfl"),  # yolov8l & yolov8s @ 640
    pytest.param((1, 4, _REG_MAX, 33600), id="l-1280-dfl"),  # yolov8l @ 1280 only
]


@U.with_default_mesh()
@pytest.mark.parametrize("shape", _DFL_SITES)
def test_softmax(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)

    out = ttnn.softmax(x, dim=2)

    ref = torch.softmax(x_torch.float(), dim=2)
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)


# ---------------------------------------------------------------------------
# Model-faithful (class C): L1-interleaved TILE input, matching the real DFL
# buffer state. The DFL softmax at ttnn_yolov8l.py:645 runs on _DETECT_MEM_CONFIG,
# which TtDetectionModel pins to L1_MEMORY_CONFIG for res <= 640
# (ttnn_yolov8l.py:772-773). Same shapes as test_softmax, DRAM -> L1 input only.
# ---------------------------------------------------------------------------
@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("shape", _DFL_SITES)
def test_softmax_l1(ttnn_mesh_device, reset_seeds, shape):
    """DFL softmax(dim=2) on L1 detect buffer (ttnn_yolov8l.py:645, mem cfg :772-773)."""
    run_softmax_l1(ttnn_mesh_device, shape)


def run_softmax_l1(mesh, shape, compute_kernel_config=None, label=None):
    """Body of test_softmax_l1. ``compute_kernel_config=None`` is the model's call (op default).

    Logs accuracy metrics (see ``_log_softmax_accuracy``) under ``label`` before the PCC assert,
    so a failing case still reports them.
    """
    x_torch = U.torch_rand(shape)
    x = U.to_tile_l1(x_torch, mesh)

    if compute_kernel_config is None:
        out = ttnn.softmax(x, dim=2)
    else:
        out = ttnn.softmax(x, dim=2, compute_kernel_config=compute_kernel_config)

    # Read back once: the metrics and the PCC assert share it (readback dominates sim/emulator time).
    got = U.from_tt(out, mesh)
    _log_softmax_accuracy(x_torch, got, dim=2, label=label or f"shape={tuple(shape)}")

    ref = torch.softmax(x_torch.float(), dim=2)
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh, host=got)


def _bf16_ordered(t):
    """bf16 bit patterns mapped to integers that are monotonic in value, so differences are ulp counts."""
    b = t.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    return torch.where(b >= 0, b, -(b & 0x7FFF))


def _log_softmax_accuracy(x_torch, got, *, dim, label):
    """Log softmax accuracy against an fp64 reference computed from the same bf16 input.

    One line prefixed ``SOFTMAX_ACCURACY`` (grep-friendly), with:
      pcc            Pearson correlation vs the fp64 reference
      abs max/mean   |got - ref|
      rel max/mean   |got - ref| / |ref|
      ulp max/mean   bf16 ulp distance from the correctly rounded (RNE) bf16 reference
      ulp0/1/2       fraction of outputs within 0, 1, 2 bf16 ulp of it (ulp0 = exactly rounded)
      rowsum max/mean |sum(got, dim) - 1|  (the reference row sums to 1 to fp64 precision)
    """
    ref = torch.softmax(x_torch.double(), dim=dim)
    got = got.double()
    if got.shape != ref.shape and got.numel() >= ref.numel():
        got = got.reshape(-1)[: ref.numel()].reshape(ref.shape)

    g, r = got.reshape(-1), ref.reshape(-1)
    pcc = torch.corrcoef(torch.stack([g, r]))[0, 1].item()

    abs_err = (got - ref).abs()
    rel_err = abs_err / ref.abs().clamp_min(torch.finfo(torch.float64).tiny)

    ulp = (_bf16_ordered(got) - _bf16_ordered(ref)).abs().double()

    rowsum_err = (got.sum(dim=dim) - 1.0).abs()

    nonfinite = (~torch.isfinite(got)).sum().item()

    logger.info(
        f"SOFTMAX_ACCURACY [{label}] n={ref.numel()} pcc={pcc:.6f} "
        f"abs_max={abs_err.max().item():.3e} abs_mean={abs_err.mean().item():.3e} "
        f"rel_max={rel_err.max().item():.3e} rel_mean={rel_err.mean().item():.3e} "
        f"ulp_max={int(ulp.max().item())} ulp_mean={ulp.mean().item():.3f} "
        f"ulp0={(ulp == 0).double().mean().item():.4f} ulp1={(ulp <= 1).double().mean().item():.4f} "
        f"ulp2={(ulp <= 2).double().mean().item():.4f} "
        f"rowsum_max={rowsum_err.max().item():.3e} rowsum_mean={rowsum_err.mean().item():.3e} "
        f"nonfinite={nonfinite}"
    )
