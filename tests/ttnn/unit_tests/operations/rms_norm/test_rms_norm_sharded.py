# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4 — Multi-core row distribution + HEIGHT_SHARDED.

Two things land together, both the Row-axis knob-turn (each core owns a
contiguous span of tile-rows and reduces LOCALLY over W — no cross-core comms):

  * Interleaved multi-core: the tile-row range is spread over the grid via
    split_work_to_cores. Exercised implicitly by the multi-tile-row shapes here
    (and by test_rms_norm.py's multi-batch shapes) — total_tile_rows > 1 now
    dispatches to multiple cores.
  * HEIGHT_SHARDED: the SAME split with the row->core assignment pinned by the
    shard spec. The resident L1 shard is streamed through the SAME bounded
    scratch CBs via TensorAccessor (a local L1->L1 read); output inherits the
    input's shard spec.

Kernels are unchanged from Phase 0/R3 — they already key off per-core
(start_tile_row, num_tile_rows) RT args. This file guards the host-side work
distribution against regression. Sharded configs are built with the golden
`eval.sharding.auto_shard_config` (the same legal-shard synthesizer the golden
suite uses) so the unit + golden nets agree.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995}
_TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}
HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def pytorch_rms_norm(x, gamma=None, epsilon=1e-6):
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + epsilon)
    out = xf / rms
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out.to(x.dtype)


def _cfg():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


# tile-aligned (single/multi core), non-tile-aligned H and W, multi-image, 3D, 2D.
SHARDED_SHAPES = [
    (1, 1, 256, 512),  # 8 tile-rows -> 8 cores, full W=512 each (design loose case)
    (2, 4, 128, 512),  # multi-image, 32 tile-rows
    (1, 1, 64, 17),  # W non-aligned (masked reduce) + sharded
    (1, 1, 50, 128),  # H non-aligned (padding tile-rows) + sharded
    (4, 128, 512),  # 3D
    (1024, 1024),  # 2D, 32 tile-rows
]


@pytest.mark.parametrize("shape", SHARDED_SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("with_gamma", [False, True])
def test_rms_norm_height_sharded(device, shape, dtype, layout, with_gamma):
    torch.manual_seed(42)
    torch_dtype = _TORCH_DTYPE[dtype]
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch_dtype)
    mem_cfg = auto_shard_config(list(shape), HEIGHT, layout=layout, dtype=dtype, device=device)
    ttnn_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=mem_cfg)

    if with_gamma:
        torch_gamma = torch.randn(W, dtype=torch_dtype)
        ttnn_gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W),
            dtype=dtype,
            layout=layout,  # gamma format follows input layout here (both legs exercised)
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        torch_gamma = None
        ttnn_gamma = None

    expected = pytorch_rms_norm(torch_input, gamma=torch_gamma, epsilon=1e-6)

    # Output inherits the input's shard spec (the norm contract the golden
    # harness enforces for sharded input).
    ttnn_output = rms_norm(
        ttnn_input,
        gamma=ttnn_gamma,
        epsilon=1e-6,
        compute_kernel_config=_cfg(),
        memory_config=ttnn_input.memory_config(),
    )

    assert ttnn_output.layout == layout, "output layout must match input layout"
    assert ttnn_output.memory_config().memory_layout == HEIGHT, "output must stay HEIGHT_SHARDED"

    actual = ttnn.to_torch(ttnn_output).reshape(expected.shape)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), PCC[dtype])


def test_rms_norm_height_sharded_rejects_width_sharded(device, expect_error):
    """WIDTH_SHARDED is a cross-core scheme-change (Refinement 5) — still refused."""
    torch.manual_seed(42)
    shape = (1, 1, 32, 2048)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    mem_cfg = auto_shard_config(
        list(shape),
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_cfg
    )
    with expect_error((NotImplementedError, RuntimeError), ".*"):
        rms_norm(ttnn_input, compute_kernel_config=_cfg(), memory_config=ttnn_input.memory_config())
