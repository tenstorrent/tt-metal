# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HIDDEN_SPLIT sweep driver (Refinement 2, perf lamp L1) for the op layout.

Forces one hidden split for every plan via `HIDDEN_SPLIT_CANDIDATES` and runs the op on the golden M classes. Run under
`scripts/run_safe_pytest.sh --profile` and read DEVICE KERNEL DURATION per case (two calls a case, in parametrize
order). Correctness is checked loosely; the acceptance / placement tests are the correctness gate.
"""

import importlib

import pytest
import torch
import ttnn

# The package re-exports the op function under the module's own name, so import the module explicitly.
op_module = importlib.import_module("ttnn.operations.nomic_residual_layer_norm.nomic_residual_layer_norm")

HIDDEN = 768

# (batch, seq): M = batch * ceil(seq / 32) tile rows; the golden M classes 32 .. 128.
SHAPES = [(2, 512), (3, 352), (4, 288), (3, 400), (4, 352), (3, 512), (4, 512), (5, 512), (6, 512), (8, 512)]


@pytest.mark.parametrize("batch, seq", SHAPES, ids=[f"{b}x{s}" for b, s in SHAPES])
@pytest.mark.parametrize("split", [2, 3], ids=["G2", "G3"])
def test_split_sweep(device, monkeypatch, split, batch, seq):
    monkeypatch.setattr(op_module, "HIDDEN_SPLIT_CANDIDATES", (split,))
    shape = (batch, 1, seq, HIDDEN)
    try:
        plan = op_module._plan(shape, device)
    except op_module.UnsupportedAxisValue:
        pytest.skip(f"no G={split} plan for {shape}")
    assert plan.hidden_split == split
    torch.manual_seed(0)
    x = torch.randn(shape).to(torch.bfloat16)
    r = torch.randn(shape).to(torch.bfloat16)
    gamma = 1.0 + 0.25 * torch.randn(HIDDEN)
    beta = (0.1 * torch.randn(HIDDEN)).to(torch.bfloat16)
    layout = op_module.input_memory_config(shape, device)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=layout)
    tt_r = ttnn.from_torch(r, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=layout)
    tt_g = ttnn.from_torch(gamma.reshape(1, 1, 1, HIDDEN), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(beta.reshape(1, 1, 1, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    for _ in range(2):
        out = op_module.nomic_residual_layer_norm(
            tt_x,
            tt_g,
            tt_b,
            residual=tt_r,
            epsilon=1e-5,
            compute_kernel_config=op_module.default_compute_kernel_config(),
        )
    y = x.double() + r.double()
    expected = torch.nn.functional.layer_norm(y, (HIDDEN,), gamma.double(), beta.double(), eps=1e-5)
    got = ttnn.to_torch(out).double()
    assert torch.corrcoef(torch.stack([got.flatten(), expected.flatten()]))[0, 1] > 0.9999
