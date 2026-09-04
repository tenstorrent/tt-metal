# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device performance pins for rms_norm_ttnn.

DO NOT DELETE.  Two jobs, and only the first is a gate:

  1. NO-REGRESSION AGAINST THE SEED.  The configurations that supply no
     optional operand -- and the gamma-only ones -- must be as fast as
     `ttnn/ttnn/operations/rms_norm`, the designated seed this op extends.  The
     programs should be byte-identical there by construction (every new CB, CT
     arg and blocking term is multiplied by its HAS_* flag), so this is the
     measurement that says so rather than an argument that it must be.
  2. OPERAND COST, recorded.  What a residual / a bias actually costs on the
     same shape, so the next perf round starts from a number instead of a
     guess.

Run under the profiler; the ratio is what is asserted, never an absolute
nanosecond count (those are board- and clock-specific):

    scripts/run_safe_pytest.sh --profile \\
        tests/ttnn/unit_tests/operations/rms_norm_ttnn/test_rms_norm_ttnn_perf.py

Off the profiler these tests still run (they just check correctness of the
shapes they touch), which is why they are safe to leave in the ordinary suite.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from ttnn.operations.rms_norm import rms_norm as rms_norm_seed
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn

# (rows, hidden) — one decode profile and one prefill profile, both at the
# precision corner the feature spec's perf cases pin (bf16 / HiFi2 / 16-bit
# DEST), plus one wide decode row that forces the cross-core width split.
SHAPES = [(32, 1024), (8192, 1024), (32, 7168)]


def _config():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def _tensors(device, rows, hidden):
    torch.manual_seed(0)
    x = torch.randn(1, 1, rows, hidden, dtype=torch.float32).to(torch.bfloat16)
    g = torch.randn(1, 1, 1, hidden, dtype=torch.float32).to(torch.bfloat16)
    return (
        ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )


@pytest.mark.parametrize("rows, hidden", SHAPES, ids=[f"{r}x{h}" for r, h in SHAPES])
@pytest.mark.parametrize("op", ["seed", "ttnn"], ids=["seed", "ttnn"])
@pytest.mark.parametrize("mode", ["no_gamma", "gamma"])
def test_seed_parity(device, rows, hidden, op, mode):
    """The seed and this op, same shape, same config, adjacent rows in the CSV.

    Read the two DEVICE KERNEL DURATION values for a (shape, mode) pair and
    divide: anything materially above 1.0 for `ttnn` is a regression against the
    seed on a configuration that is supposed to build the same program.
    """
    x, g = _tensors(device, rows, hidden)
    kwargs = {"epsilon": 1e-12, "compute_kernel_config": _config()}
    if op == "seed":
        out = rms_norm_seed(x, gamma=(g if mode == "gamma" else None), **kwargs)
    else:
        out = rms_norm_ttnn(x, weight=(g if mode == "gamma" else None), **kwargs)
    assert list(out.shape) == [1, 1, rows, hidden]


@pytest.mark.parametrize("rows, hidden", SHAPES, ids=[f"{r}x{h}" for r, h in SHAPES])
@pytest.mark.parametrize(
    "mode",
    ["gamma", "gamma_bias", "residual", "gamma_bias_residual"],
)
def test_operand_cost(device, rows, hidden, mode):
    """What each operand costs on the same shape, as a profiled row.

    Not a gate -- there is no seed number to compare against, because the seed
    has no residual and no bias.  It exists so the operand multiplier is a
    MEASUREMENT the next perf round can act on: the residual doubles every
    activation DRAM crossing, which is what moves a shape between the RESIDENT /
    ROW_RESIDENT / STREAM regimes.
    """
    x, g = _tensors(device, rows, hidden)
    kwargs = {"epsilon": 1e-12, "compute_kernel_config": _config()}
    if "gamma" in mode:
        kwargs["weight"] = g
    if "bias" in mode:
        torch.manual_seed(4)
        kwargs["bias"] = ttnn.from_torch(
            torch.randn(1, 1, 1, hidden, dtype=torch.float32).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
    if "residual" in mode:
        torch.manual_seed(5)
        kwargs["residual_input_tensor"] = ttnn.from_torch(
            torch.randn(1, 1, rows, hidden, dtype=torch.float32).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
    out = rms_norm_ttnn(x, **kwargs)
    assert list(out.shape) == [1, 1, rows, hidden]
