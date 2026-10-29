# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf shapes for rms_norm. Run under `run_safe_pytest.sh --profile`; correctness is a loose PCC gate."""

import os

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor

# Blocking sweep hook: RMS_NORM_MIN_BLOCKS_PER_CORE overrides the R knob's divisor for this run.
_MIN_BLOCKS_OVERRIDE = os.environ.get("RMS_NORM_MIN_BLOCKS_PER_CORE")

SHAPES = [
    pytest.param((1, 1, 32, 4096), id="decode_4096"),
    pytest.param((1, 1, 32, 8192), id="decode_8192"),
    pytest.param((1, 1, 1024, 1024), id="1024x1024"),
    pytest.param((1, 1, 4096, 2048), id="4096x2048"),
    pytest.param((1, 1, 8192, 4096), id="prefill_8192x4096"),
    pytest.param((1, 1, 16384, 64), id="narrow_16384x64"),
]


@pytest.mark.parametrize("with_gamma", [True, False], ids=["gamma", "no_gamma"])
@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm_perf(device, shape, with_gamma, monkeypatch):
    if _MIN_BLOCKS_OVERRIDE is not None:
        monkeypatch.setattr(rms_norm_program_descriptor, "MIN_BLOCKS_PER_CORE", int(_MIN_BLOCKS_OVERRIDE))
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    x = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kwargs = {}
    torch_gamma = None
    if with_gamma:
        torch_gamma = torch.randn(shape[-1], dtype=torch.bfloat16)
        kwargs["gamma"] = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
    out = ttnn.to_torch(rms_norm(x, **kwargs)).float()

    ref = torch_input.float()
    ref = ref * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + 1e-6)
    if torch_gamma is not None:
        ref = ref * torch_gamma.float()
    pcc = torch.corrcoef(torch.stack([ref.flatten(), out.flatten()]))[0, 1].item()
    assert pcc > 0.995, pcc


@pytest.mark.parametrize("shape", [(1, 1, 8192, 4096), (1, 1, 4096, 2048)], ids=["8192x4096", "4096x2048"])
def test_dram_copy_baseline(device, shape):
    """Read-once / write-once reference (ttnn.clone) for the DRAM ceiling of the same bytes."""
    x = ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.clone(x)
