# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.slice`` (TILE and ROW_MAJOR).

Shape source (kernel cache): tile slice reader ct = [rank, TA] => only rank, dtype and memory
are recoverable; RM slice reader/writer page sizes pin the in/out row widths.
  * S6 RM [.., 768] -> [.., 256] (fused q/k/v split), DRAM  [M]
  * S9 RM fp32 32-wide -> 16-wide, DRAM                       [M widths]
  * the rest: rank / dtype / memory determined, ranges representative  [L]
Slice is value-preserving: checked bit-exact.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

Q, E = U.NUM_QUERY, U.EMBED
RM, TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

# fmt: off
# (shape, begins, ends, layout, dtype, mem)
CASES = [
    pytest.param((320, 512), (0, 0), (320, 256), TILE, "bfp8", "dram", id="S1-tile-rank2-bfp8"),
    pytest.param((1, Q, E), (0, 0, 0), (1, Q, 128), TILE, "bf16", "dram", id="S2-tile-rank3-dram"),
    pytest.param((1, Q, E), (0, 0, 0), (1, Q, 128), TILE, "bf16", "l1", id="S3-tile-rank3-l1"),
    pytest.param((1, Q, 8, 4, 64), (0, 0, 0, 0, 0), (1, Q, 8, 4, 32), TILE, "bf16", "dram", id="S4-tile-rank5-dram"),
    pytest.param((1, 1000, 8, 4, 64), (0, 0, 0, 0, 0), (1, 1000, 8, 4, 32), TILE, "bf16", "l1", id="S5-tile-rank5-l1"),
    pytest.param((1, 1, 300, 3 * E), (0, 0, 0, E), (1, 1, 300, 2 * E), RM, "bf16", "dram", id="S6-rm-qkv-split-k"),
    pytest.param((1, Q, E), (0, 0, 0), (1, 5000, E), RM, "bf16", "l1", id="S7-rm-rows-l1"),
    pytest.param((1, 100, 32), (0, 0, 0), (1, 100, 16), RM, "bf16", "dram", id="S8-rm-narrow"),
    pytest.param((1, 100, 32), (0, 0, 0), (1, 100, 16), RM, "fp32", "dram", id="S9-rm-fp32"),
]
# fmt: on


@U.with_default_mesh()
@pytest.mark.parametrize("shape, begins, ends, layout, dt, mem", CASES)
def test_slice(ttnn_mesh_device, reset_seeds, shape, begins, ends, layout, dt, mem):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape, dtype=torch.float32 if dt == "fp32" else torch.bfloat16)
    x = U.to_tt_mem(x_torch, mesh, mem, dtype=U.DTYPES[dt], layout=layout)
    src = U.seen(x, x_torch, mesh)
    out = ttnn.slice(x, begins, ends, memory_config=U.resolve_memcfg(mem, mesh))
    ref = src[tuple(slice(b, e) for b, e in zip(begins, ends))]
    U.assert_lossless(ref, out, mesh_device=mesh)
