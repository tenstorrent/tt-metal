# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for genify_matmul (Phase 0 cell). IMMUTABLE: the implementer must not edit it.

Phase 0 cell: A bf16 x B bf16, TILE, fp32_dest_acc_en=True, interleaved, epilogue none,
tile-aligned, single (shared) weight. Regime R1 (interleaved_out_split) is pinned with:
  - a 2-D logical grid (grid_m' > 1 and grid_n' > 1) and a 1-D grid (M = 32 -> grid_m' = 1),
  - a single K-block (no accumulator CB) and a multi-K-block walk with a ragged last K-block,
  - folded leading dims (rank 4, single B), rank 2, an odd N-tile count (edge-core skip),
  - L1-interleaved A and output.
The shapes keep these properties on any compute grid of at least 4x4.

Accuracy gate: helpers.TOLERANCES[(output dtype, fp32_dest_acc_en)], PCC and relative RMS together.
"""

import math

import pytest
import torch

import ttnn

from eval.golden_tests.genify_matmul.helpers import (
    TOLERANCES,
    check_output,
    create_ttnn_input_tensor,
    pytorch_genify_matmul,
    quantize,
)
from ttnn.operations.genify_matmul import default_compute_kernel_config, genify_matmul


def _hifi4_fp32_acc():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _run(device, a_shape, b_shape, *, act_mc=None, out_mc=None, explicit_config=True):
    torch.manual_seed(42)
    A = torch.randn(a_shape)
    B = torch.randn(b_shape) / math.sqrt(b_shape[-2])
    ttnn_A = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=act_mc)
    ttnn_B = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    expected = pytorch_genify_matmul(quantize(A, ttnn.bfloat16), quantize(B, ttnn.bfloat16))

    kwargs = {}
    if explicit_config:
        kwargs["compute_kernel_config"] = _hifi4_fp32_acc()
    if out_mc is not None:
        kwargs["memory_config"] = out_mc
    out = genify_matmul(ttnn_A, ttnn_B, **kwargs)

    out_shape = list(a_shape[:-1]) + [b_shape[-1]]
    check_output(
        out,
        expected,
        shape=out_shape,
        dtype=ttnn.bfloat16,
        expected_layout=ttnn.TILE_LAYOUT,
        tolerance=TOLERANCES[(ttnn.bfloat16, True)],
    )
    return out


SHAPES = [
    # (A shape, B shape)                         what it pins
    ((1, 1, 32, 32), (32, 32)),  # single tile, num_k_blocks == 1
    ((1, 1, 128, 256), (256, 128)),  # multi-tile, square-ish, 2-D grid
    ((1, 384, 1536), (1536, 1024)),  # non-square prefill / DiT (rank 3), 2-D grid
    ((2, 3, 64, 128), (128, 192)),  # multi-batch: rank-4 leading dims fold into M under a single B
    ((1, 1, 32, 1024), (1024, 2048)),  # M = 32: 1-D grid (grid_m' = 1), N split only
    ((64, 256), (256, 160)),  # rank 2, odd N-tile count (5): edge-core column skip
    ((1, 1, 2048, 256), (256, 512)),  # tall M, small K and N (PDL-like), 2-D grid
    ((1, 1, 128, 8032), (8032, 256)),  # deep K, Kt = 251: multi K-block walk with a ragged last K-block
]
IDS = [f"A{'x'.join(map(str, a))}_B{'x'.join(map(str, b))}" for a, b in SHAPES]


@pytest.mark.parametrize("a_shape, b_shape", SHAPES, ids=IDS)
def test_genify_matmul(a_shape, b_shape, device):
    _run(device, a_shape, b_shape)


def test_genify_matmul_default_compute_config(device):
    """compute_kernel_config=None resolves through default_compute_kernel_config()."""
    cfg = default_compute_kernel_config()
    assert cfg.fp32_dest_acc_en is True
    assert cfg.math_fidelity == ttnn.MathFidelity.HiFi4
    assert cfg.math_approx_mode is False
    _run(device, (1, 1, 64, 512), (512, 256), explicit_config=False)


def test_genify_matmul_l1_interleaved(device):
    """A and the output in L1 interleaved (model cases keep them in L1); the op must address them
    through the buffer's accessor and size its CBs beside them."""
    out = _run(
        device,
        (1, 1, 128, 512),
        (512, 1024),
        act_mc=ttnn.L1_MEMORY_CONFIG,
        out_mc=ttnn.L1_MEMORY_CONFIG,
    )
    mc = out.memory_config()
    assert mc.buffer_type == ttnn.BufferType.L1
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def test_genify_matmul_default_output_placement(device):
    """memory_config=None -> DRAM interleaved; dtype=None -> A's dtype."""
    out = _run(device, (1, 1, 64, 128), (128, 64))
    mc = out.memory_config()
    assert mc.buffer_type == ttnn.BufferType.DRAM
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert out.dtype == ttnn.bfloat16


def test_genify_matmul_repeat_bit_identical(device):
    """No timing-dependent combine: two calls give bit-identical output."""
    first = ttnn.to_torch(_run(device, (1, 1, 128, 8032), (8032, 256)))
    second = ttnn.to_torch(_run(device, (1, 1, 128, 8032), (8032, 256)))
    assert torch.equal(first, second)


# ---------------------------------------------------------------------------
# Contract violations: ValueError / RuntimeError (raised before the SUPPORTED gate).
# The message pattern is empty on purpose: the wording is the implementer's.
# ---------------------------------------------------------------------------


def _bf16(device, shape):
    return create_ttnn_input_tensor(torch.randn(shape), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def test_rejects_k_mismatch(device, expect_error):
    with expect_error((ValueError, RuntimeError), ""):
        genify_matmul(_bf16(device, (1, 1, 32, 64)), _bf16(device, (96, 32)))


def test_rejects_bad_weight_leading_dims(device, expect_error):
    with expect_error((ValueError, RuntimeError), ""):
        genify_matmul(_bf16(device, (2, 64, 64)), _bf16(device, (3, 64, 64)))


def test_rejects_unknown_activation(device, expect_error):
    with expect_error((ValueError, RuntimeError), ""):
        genify_matmul(_bf16(device, (1, 1, 32, 64)), _bf16(device, (64, 64)), activation="gelu")


def test_rejects_swiglu_n_not_multiple_of_64(device, expect_error):
    with expect_error((ValueError, RuntimeError), ""):
        genify_matmul(_bf16(device, (1, 1, 32, 64)), _bf16(device, (64, 96)), activation="swiglu")


def test_rejects_residual_without_gate(device, expect_error):
    with expect_error((ValueError, RuntimeError), ""):
        genify_matmul(_bf16(device, (1, 1, 32, 64)), _bf16(device, (64, 64)), residual=_bf16(device, (1, 1, 32, 64)))


def test_rejects_gate_without_residual(device, expect_error):
    with expect_error((ValueError, RuntimeError), ""):
        genify_matmul(_bf16(device, (1, 1, 32, 64)), _bf16(device, (64, 64)), gate=_bf16(device, (1, 64)))


def test_rejects_bad_bias_shape(device, expect_error):
    with expect_error((ValueError, RuntimeError), ""):
        genify_matmul(_bf16(device, (1, 1, 32, 64)), _bf16(device, (64, 64)), bias=_bf16(device, (1, 128)))
