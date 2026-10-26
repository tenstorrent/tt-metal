# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: the unary SFPU ops the VAD-variant model ran — relu, sigmoid, clamp(min=1),
fill(1e-5), identity (bf16) and relu, sqrt, square, sin, cos (fp32).

All ran through one program factory (``unary_ng``: ``eltwise_sfpu`` /
``eltwise_identity_kernel`` + ``reader/writer_unary_ng``), so they share a file. The op is
decoded from the cached ``SFPU_OP_CHAIN_0`` define [H]:
  * ``clamp_tile(0, 0x3F800000, 0x7F7FFFFF)`` = clamp(min=1.0)  (SCA hit-count clamp)
  * ``fill_tile_bitcast(0, 0x3727c5ac)`` = fill 1e-5          (inverse_sigmoid eps)
  * ``sigmoid_tile<4, 0u>`` = accurate sigmoid; ``sqrt_tile<0>`` = accurate sqrt
HiFi4, approx off; fp32 dest acc on for fp32 inputs [H]. Both L1 and DRAM interleaved
reader/writer variants exist per dtype; which op used which is not recoverable. Shapes are
runtime args (lost): BEV-sized [1, 10000, 256] for encoder-side ops, small decoder-side
tensors otherwise [L]. Hash prefixes are the compute variant.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

Q, E = U.NUM_QUERY, U.EMBED


def _positive(t):
    return t.abs() + 0.1


# fmt: off
# (name, tt_fn, torch_fn, shape, dtype, mem, input_transform)
CASES = [
    pytest.param(lambda x, m: ttnn.relu(x, memory_config=m), torch.relu, (1, Q, E), "bf16", "l1", None, id="U1-relu-bf16-116423"),
    pytest.param(lambda x, m: ttnn.sigmoid(x, memory_config=m), torch.sigmoid, (1, 300, 32), "bf16", "l1", None, id="U2-sigmoid-bf16-229308"),
    pytest.param(lambda x, m: ttnn.clamp(x, min=1.0, memory_config=m), lambda t: t.clamp(min=1.0), (1, Q, 32), "bf16", "l1", None, id="U3-clamp-min1-bf16-106987"),
    pytest.param(lambda x, m: ttnn.identity(x, memory_config=m), lambda t: t, (1, Q, E), "bf16", "dram", None, id="U5-identity-bf16-588607"),
    pytest.param(lambda x, m: ttnn.relu(x, memory_config=m), torch.relu, (1, 1, E), "fp32", "dram", None, id="U6-relu-fp32-157440"),
    pytest.param(lambda x, m: ttnn.sqrt(x, memory_config=m), torch.sqrt, (1, 300, 32), "fp32", "l1", _positive, id="U7-sqrt-fp32-131905"),
    pytest.param(lambda x, m: ttnn.square(x, memory_config=m), torch.square, (1, 300, 32), "fp32", "l1", None, id="U8-square-fp32-181728"),
    pytest.param(lambda x, m: ttnn.sin(x, memory_config=m), torch.sin, (1, 300, E), "fp32", "dram", None, id="U9-sin-fp32-142203"),
    pytest.param(lambda x, m: ttnn.cos(x, memory_config=m), torch.cos, (1, 300, E), "fp32", "dram", None, id="U10-cos-fp32-171532"),
]
# fmt: on


@U.with_default_mesh()
@pytest.mark.parametrize("tt_fn, torch_fn, shape, dt, mem, transform", CASES)
def test_unary(ttnn_mesh_device, reset_seeds, tt_fn, torch_fn, shape, dt, mem, transform):
    mesh = ttnn_mesh_device
    torch_dtype = torch.float32 if dt == "fp32" else torch.bfloat16
    x_torch = U.torch_rand(shape, dtype=torch_dtype)
    if transform is not None:
        x_torch = transform(x_torch)
    x = U.to_tt_mem(x_torch, mesh, mem, dtype=U.DTYPES[dt])
    out = tt_fn(x, U.resolve_memcfg(mem, mesh))
    U.assert_pcc(torch_fn(x_torch.float()), out, pcc=0.999, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize("shape", [pytest.param((1, U.NUM_QUERY, 4, 2), id="U4-fill-1e-5-bf16-911025")])
def test_fill(ttnn_mesh_device, reset_seeds, shape):
    """fill(1e-5): every element (logical region) must be bf16(1e-5)."""
    mesh = ttnn_mesh_device
    x = U.to_tt_mem(U.torch_rand(shape), mesh, "l1")
    out = ttnn.fill(x, 1e-5)
    got = U.from_tt(out, mesh)
    # The kernel fills the fp32 bit pattern 0x3727c5ac and the packer truncates it to bf16
    # (0x3727 = 9.954e-6), while torch rounds to nearest (0x3728 = 1.0014e-5). Accept either.
    fp32_bits = torch.tensor([1e-5], dtype=torch.float32).view(torch.int32)
    truncated = (fp32_bits & ~0xFFFF).view(torch.float32)[0]
    rounded = torch.tensor(1e-5, dtype=torch.bfloat16).float()
    ok = torch.all(got == truncated) or torch.all(got == rounded)
    assert ok, f"fill produced values other than {truncated.item()} (truncated) / {rounded.item()} (rounded)"
