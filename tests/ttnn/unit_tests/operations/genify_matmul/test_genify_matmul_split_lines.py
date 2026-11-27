# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 16: R1a split lines (R1A_SPLIT_LINES, kernels/line_stream.hpp) -- the A row lines on the writer RISC,
the B column lines on the reader, both with rotating injectors reading each turn into a staging buffer, and the K
head ramp on every split walk (R1A_SPLIT_RAMP).

Pins:
  - mid-M, deep-K shapes take the split lines (the plan asserts it) and stay within helpers.TOLERANCES under every
    epilogue, with a ragged M / K, A as K segments and a split output;
  - with the ramp off the split lines are bit-identical to the fixed-injector R1a walk (same K-blocks, same per-core
    K order: only who reads and multicasts each K-block changes);
  - the issue-pacing knobs (window, startup share) only move timing: bit-identical results.
Accuracy gate: helpers.TOLERANCES via run_genify_matmul (the golden-suite runner).
"""

import math

import pytest
import torch
import ttnn

from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor, run_genify_matmul
from ttnn.operations.genify_matmul import genify_matmul
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd


# The split lines pay on data-movement-heavy walks (R1A_SPLIT_MIN_RATIO): the model configs (Qwen3.8 prefill bf16 x
# bfp4 LoFi; text encoder bf16 x bfp8 HiFi2).
LOFI, HIFI2 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2
BFP4, BFP8 = ttnn.bfloat4_b, ttnn.bfloat8_b


def _plan(device, a_shape, b_shape, epilogue="none", w_dtype=BFP4, fidelity=LOFI):
    n_out = b_shape[-1] // 2 if epilogue == "swiglu" else b_shape[-1]
    a = ttnn.allocate_tensor_on_device(ttnn.Shape(list(a_shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    b = ttnn.allocate_tensor_on_device(ttnn.Shape(list(b_shape)), w_dtype, ttnn.TILE_LAYOUT, device)
    o = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(a_shape[:-1]) + [n_out]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device
    )
    geom, grid, blk, _ = pd.plan(a, b, o, True, device, epilogue=epilogue, math_fidelity=fidelity)
    for t in (a, b, o):
        ttnn.deallocate(t)
    return geom, grid, blk


def _run(inputs, epilogue, device, weight_dtype=BFP4, fidelity=LOFI):
    run_genify_matmul(
        inputs,
        dtype=ttnn.bfloat16,
        weight_dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=True,
        placement="interleaved",
        epilogue=epilogue,
        device=device,
        extras={"math_fidelity": fidelity},
    )


@pytest.mark.parametrize(
    "a_shape, b_shape, epilogue, w_dtype, fidelity",
    [
        ((1, 512, 5120), (5120, 4160), "none", BFP4, LOFI),  # Qwen3.8 prefill in_proj (240-tile bfp4 B K-blocks)
        ((1, 512, 4352), (4352, 5120), "bias", BFP4, LOFI),
        ((1, 512, 4352), (4352, 5120), "bias_relu", BFP4, LOFI),
        ((1, 512, 5120), (5120, 8704), "swiglu", BFP4, LOFI),  # MLP gate_up + SwiGLU (2 x 2 subblocks)
        ((1, 512, 4352), (4352, 5120), "addcmul", BFP4, LOFI),  # raw K-walk: compact A under the ramp
        ((1, 500, 5088), (5088, 4160), "none", BFP4, LOFI),  # ragged M (500), ragged last K-block (159 tiles)
        ((2, 256, 5120), (5120, 4160), "none", BFP4, LOFI),  # folded leading dims
        ((1, 512, 4096), (4096, 2560), "none", BFP8, HIFI2),  # text-encoder o_proj
    ],
)
def test_split_lines(a_shape, b_shape, epilogue, w_dtype, fidelity, device):
    geom, grid, blk = _plan(device, a_shape, b_shape, epilogue, w_dtype, fidelity)
    assert pd.r1a_split(geom, grid, blk), f"expected the R1a split lines, got {grid} {blk}"
    _run((a_shape, b_shape), epilogue, device, w_dtype, fidelity)


def _tt(x, device, dtype=ttnn.bfloat16):
    return create_ttnn_input_tensor(x, device, dtype=dtype, layout=ttnn.TILE_LAYOUT)


def _call(a, b, device, **kwargs):
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=LOFI, fp32_dest_acc_en=True, math_approx_mode=False)
    a_t = [_tt(x, device) for x in a] if isinstance(a, list) else _tt(a, device)
    out = genify_matmul(a_t, _tt(b, device, BFP4), compute_kernel_config=cfg, **kwargs)
    return out if isinstance(out, list) else [out]


@pytest.mark.parametrize("epilogue", ["none", "addcmul"])
def test_split_lines_bit_identical_to_fixed_injectors(epilogue, device, monkeypatch):
    m, k, n = 512, 4352, 5120
    torch.manual_seed(1)
    a = torch.randn(1, m, k)
    b = torch.randn(k, n) / math.sqrt(k)
    kwargs = {}
    if epilogue == "addcmul":
        kwargs = {
            "residual": _tt(torch.randn(1, m, n), device),
            "gate": _tt(torch.randn(1, n), device),
        }
    monkeypatch.setattr(pd, "R1A_SPLIT_RAMP", False)
    split = ttnn.to_torch(_call(a, b, device, **kwargs)[0])
    monkeypatch.setattr(pd, "R1A_SPLIT_LINES", False)
    fixed = ttnn.to_torch(_call(a, b, device, **kwargs)[0])
    assert torch.equal(split, fixed)


@pytest.mark.parametrize("window, share", [(1, 1), (3, 0), (8, 4)])
def test_split_lines_pacing_bit_identical(window, share, device, monkeypatch):
    m, k, n = 512, 5120, 4160
    torch.manual_seed(2)
    a = torch.randn(1, m, k)
    b = torch.randn(k, n) / math.sqrt(k)
    ref = ttnn.to_torch(_call(a, b, device)[0])
    monkeypatch.setattr(pd, "R1A_LINE_WINDOW", window)
    monkeypatch.setattr(pd, "R1A_LINE_STARTUP_SHARE", share)
    assert torch.equal(ttnn.to_torch(_call(a, b, device)[0]), ref)


def test_split_lines_segments_and_chunks(device):
    """A as two K segments, the output split in three chunks, on a split-lines walk."""
    m, k0, k1, n = 512, 2048, 3072, 4224
    geom, grid, blk = _plan(device, (1, m, k0 + k1), (k0 + k1, n))
    assert pd.r1a_split(geom, grid, blk), f"{grid} {blk}"
    torch.manual_seed(3)
    a0, a1 = torch.randn(1, m, k0), torch.randn(1, m, k1)
    b = torch.randn(k0 + k1, n) / math.sqrt(k0 + k1)
    outs = _call([a0, a1], b, device, chunks=3)
    ref = torch.cat([a0, a1], dim=-1) @ ttnn.to_torch(_tt(b, device, BFP4)).float()  # the bfp4 weight the op saw
    got = torch.cat([ttnn.to_torch(o).float() for o in outs], dim=-1)
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    assert pcc > 0.9999, pcc
