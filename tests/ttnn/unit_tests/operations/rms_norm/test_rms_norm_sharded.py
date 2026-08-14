# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — sharded placement (HEIGHT / WIDTH / BLOCK).

Pins the two behaviours that only a *sharded* input can reach, and that the
interleaved suite is structurally blind to:

1. **Native consumption.** The input CB is bound to the caller's resident L1
   buffer, so the op must not need a DRAM round trip for a local shard. This is
   asserted structurally (the descriptor's input CB carries the input tensor's
   buffer) rather than by output values — a TensorAccessor re-read of a local
   shard produces *correct* output, which is exactly why nothing else catches it.

2. **`block_rows < shard_rows`.** A resident shard makes cb_input_tiles' capacity
   the WHOLE shard, so when the L1 solve cuts the row block below the shard the
   in-place rewrite of x must address pages from the CB *base*, not from the
   read window. Getting this wrong drops the 1/rms factor on every block after
   the first — an error the size of the row's rms spread (~1/sqrt(2W)), i.e.
   invisible at large W and only ~7% at W = 96. The tall/narrow shapes below are
   chosen so that spread is large enough to see.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import default_compute_kernel_config, rms_norm
from ttnn.operations.rms_norm.rms_norm_program_descriptor import (
    CB_INPUT_TILES,
    CB_OUTPUT_TILES,
    _plan,
    create_program_descriptor,
)

_ML = ttnn.TensorMemoryLayout
_PLACEMENTS = [_ML.HEIGHT_SHARDED, _ML.WIDTH_SHARDED, _ML.BLOCK_SHARDED]

_TILE_BYTES = {
    "in_tile": 2048,
    "out_tile": 2048,
    "gamma_tile": 2048,
    "stat_tile": 4096,
    "bf16_tile": 2048,
}


def _reference(x, gamma, epsilon=1e-6):
    xf = x.to(torch.float32)
    out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return out * gamma.to(torch.float32).reshape(-1)


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _rel_rms(a, b):
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.std()).item()


def _run(device, shape, memory_layout, layout=ttnn.TILE_LAYOUT):
    torch.manual_seed(0)
    torch_x = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(shape[-1], dtype=torch.bfloat16)

    memory_config = auto_shard_config(list(shape), memory_layout, layout=layout, dtype=ttnn.bfloat16, device=device)
    x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=memory_config)
    gamma = ttnn.from_torch(torch_gamma.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=layout, device=device)

    # The norm contract: a sharded input yields a matching sharded output.
    config = ttnn.ComputeConfigDescriptor()
    config.math_fidelity = ttnn.MathFidelity.HiFi4
    config.fp32_dest_acc_en = False
    config.math_approx_mode = False
    out = rms_norm(x, gamma=gamma, compute_kernel_config=config, memory_config=x.memory_config())

    assert out.memory_config().memory_layout == memory_layout
    assert list(out.memory_config().shard_spec.shape) == list(memory_config.shard_spec.shape)
    return ttnn.to_torch(out), _reference(torch_x, torch_gamma), x


@pytest.mark.parametrize("memory_layout", _PLACEMENTS, ids=lambda m: str(m).split(".")[-1])
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 256, 512), (1, 1, 3232, 96), (1, 1, 32, 4064)],
    ids=lambda s: "x".join(str(d) for d in s),
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
def test_rms_norm_sharded_placements(device, shape, memory_layout, layout):
    """All three placements at both layouts, against the interleaved accuracy bar."""
    got, expected, _ = _run(device, shape, memory_layout, layout)
    assert _pcc(got, expected) >= 0.995, f"PCC {_pcc(got, expected)}"
    assert _rel_rms(got, expected) <= 0.04, f"rel-RMS {_rel_rms(got, expected)}"


@pytest.mark.parametrize(
    "shape",
    # Tall + narrow: the L1 solve must cut block_rows below shard_rows (the gather
    # buffer is num_hidden_slices * block_rows), and W is small enough that a
    # dropped 1/rms factor shows up as a ~5-7% rel-RMS instead of noise.
    [(1, 1, 3232, 96), (1, 1, 4064, 160)],
    ids=lambda s: "x".join(str(d) for d in s),
)
def test_rms_norm_sharded_multi_block_keeps_the_scale(device, shape):
    """block_rows < shard_rows: every block's in-place x*rsqrt must survive."""
    got, expected, x = _run(device, shape, _ML.WIDTH_SHARDED)

    plan = _plan(device, x, has_gamma=True, bytes_=_TILE_BYTES)
    assert plan["block_rows"] < plan["shard_rows"], (
        "shape no longer exercises the multi-block resident-shard path "
        f"(block_rows={plan['block_rows']}, shard_rows={plan['shard_rows']})"
    )
    # A dropped scale factor lands at ~1/sqrt(2W) = 7.2% (W=96) / 5.6% (W=160);
    # the bar below is far under both and at the interleaved path's own level.
    assert _rel_rms(got, expected) <= 0.02, f"rel-RMS {_rel_rms(got, expected)}"


@pytest.mark.parametrize("memory_layout", _PLACEMENTS, ids=lambda m: str(m).split(".")[-1])
def test_rms_norm_sharded_is_zero_copy(device, memory_layout):
    """The shard is consumed natively: the CB is bound to the caller's buffer.

    Checked on the descriptor because it is invisible in the output — an
    accessor re-read of a core's own shard is numerically correct and merely
    re-fetches bytes that already sit in L1.
    """
    shape = (1, 1, 256, 512)
    memory_config = auto_shard_config(
        list(shape), memory_layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    out = ttnn.allocate_tensor_on_device(x.shape, x.dtype, x.layout, x.device(), x.memory_config())
    descriptor = create_program_descriptor(x, out, compute_kernel_config=default_compute_kernel_config())

    bound = {}
    for cb in descriptor.cbs:
        for fmt in cb.format_descriptors:
            if cb.buffer_address() is not None:
                bound[fmt.buffer_index] = cb.buffer_address()

    assert bound.get(CB_INPUT_TILES) == x.buffer_address(), "input CB is not bound to the resident input shard"
    assert bound.get(CB_OUTPUT_TILES) == out.buffer_address(), "output CB is not bound to the resident output shard"
