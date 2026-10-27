# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.clone`` and ``ttnn.reallocate`` (copy / move).

Shape source (kernel cache):
  * clone (``clone_program_factory.cpp:118-128``, ct [4, (unit_bytes,) TA]): TILE bf16
    DRAM -> DRAM; RM W=256 DRAM -> L1; RM fp32 W=3 and W=1 DRAM -> DRAM  [M widths / L rows]
  * reallocate / move: copy-path L1 -> L1 for TILE bf16, RM W=256 and RM W=32
    (``copy_same_memory_config_program_factory.cpp``), plus the sharded path
    (``reader_unary_local_l1_copy_backwards``) for bf16 and bfp8 sharded tensors  [M / L rows]
Value-preserving: checked bit-exact.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

Q, E = U.NUM_QUERY, U.EMBED
RM, TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

# fmt: off
# (shape, layout, dtype, in_mem, out_mem)
CLONE_CASES = [
    pytest.param((1, 1, 64, E), TILE, "bf16", "dram", "dram", id="CL1-tile-bf16"),
    pytest.param((1, 100, E), RM, "bf16", "dram", "l1", id="CL2-rm-w256-to-l1"),
    pytest.param((1, 100, 3), RM, "fp32", "dram", "dram", id="CL3-rm-fp32-w3"),
    pytest.param((1, 100, 1), RM, "fp32", "dram", "dram", id="CL4-rm-fp32-w1"),
]
REALLOC_CASES = [
    pytest.param((1, 1, 64, E), TILE, "bf16", "l1", id="CP1-tile-l1"),
    pytest.param((1, 1, 64, E), RM, "bf16", "l1", id="CP2-rm-w256-l1"),
    pytest.param((1, 1, 64, 32), RM, "bf16", "l1", id="CP3-rm-w32-l1"),
]
REALLOC_SHARDED_CASES = [
    pytest.param((1, 1, Q, E), "bf16", id="MV1-hs20-bf16"),
    pytest.param((1, 1, 3840, 512), "bfp8", id="MV1-hs20-bfp8"),
]
# fmt: on


@U.with_default_mesh()
@pytest.mark.parametrize("shape, layout, dt, in_mem, out_mem", CLONE_CASES)
def test_clone(ttnn_mesh_device, reset_seeds, shape, layout, dt, in_mem, out_mem):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape, dtype=torch.float32 if dt == "fp32" else torch.bfloat16)
    x = U.to_tt_mem(x_torch, mesh, in_mem, dtype=U.DTYPES[dt], layout=layout)
    out = ttnn.clone(x, dtype=U.DTYPES[dt], memory_config=U.resolve_memcfg(out_mem, mesh))
    U.assert_lossless(x_torch, out, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize("shape, layout, dt, mem", REALLOC_CASES)
def test_reallocate(ttnn_mesh_device, reset_seeds, shape, layout, dt, mem):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    x = U.to_tt_mem(x_torch, mesh, mem, dtype=U.DTYPES[dt], layout=layout)
    out = ttnn.reallocate(x)
    U.assert_lossless(x_torch, out, mesh_device=mesh)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("shape, dt", REALLOC_SHARDED_CASES)
def test_reallocate_sharded(ttnn_mesh_device, reset_seeds, shape, dt):
    mesh = ttnn_mesh_device
    rows = shape[-2]
    shard_h = -(-rows // U.MODEL_NUM_CORES // U.TILE) * U.TILE  # tile-aligned ceil(rows / 20)
    x_torch = U.torch_rand(shape)
    x = U.to_tt_mem(x_torch, mesh, ("hs", (shard_h, shape[-1])), dtype=U.DTYPES[dt])
    ref = U.seen(x, x_torch, mesh)
    out = ttnn.reallocate(x)
    assert out.is_sharded()
    U.assert_lossless(ref, out, mesh_device=mesh)
