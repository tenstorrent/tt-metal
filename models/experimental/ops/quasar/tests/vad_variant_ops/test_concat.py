# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.concat`` (TILE and ROW_MAJOR).

Shape source (kernel cache): concat reader ct = [cb, N, page_size[N], TA[N]] => number of
inputs, their dtypes / memory and (RM) row widths are recoverable; dims and other extents
are not (runtime args).
  * C1 TILE, 6 DRAM inputs — one per camera                              [M]
  * C7 RM, 2 x W=10000 DRAM, concatenated on a non-last dim               [M]
  * the rest: input count / dtype / memory determined, shapes representative  [L]
Concat is value-preserving: checked bit-exact.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

Q, E = U.NUM_QUERY, U.EMBED
RM, TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

# fmt: off
# (shapes, dim, layout, dtype, mems, out_mem)
CASES = [
    pytest.param([(1, 1, 375, E)] * U.NUM_CAMS, 2, TILE, "bf16", ["dram"] * U.NUM_CAMS, "dram", id="C1-tile-6cams-dram"),
    pytest.param([(1, 1, 64, E)] * 3, 3, TILE, "bf16", ["l1"] * 3, "l1", id="C2-tile-3x-l1"),
    pytest.param([(1, 1, 64, E)] * 2, 3, TILE, "bf16", ["dram", "dram"], "dram", id="C3-tile-dram-dram"),
    pytest.param([(1, 1, 64, E)] * 2, 3, TILE, "bf16", ["dram", "l1"], "dram", id="C4-tile-dram-l1"),
    pytest.param([(1, 1, 64, E)] * 2, 3, TILE, "bf16", ["l1", "l1"], "l1", id="C5-tile-l1-l1"),
    pytest.param([(1, 1, 64, E)] * 2, 3, TILE, "bfp8", ["l1", "l1"], "l1", id="C6-tile-bfp8-l1"),
    pytest.param([(1, 4, Q)] * 2, 1, RM, "bf16", ["dram", "dram"], "dram", id="C7-rm-w10000-dram"),
    pytest.param([(1, 50, 32)] * 2, 1, RM, "bf16", ["l1", "l1"], "l1", id="C8-rm-w32-l1"),
    pytest.param([(100, 1)] * 2, 0, RM, "fp32", ["l1", "l1"], "dram", id="C9-rm-fp32-w1"),
]
# fmt: on


@U.with_default_mesh()
@pytest.mark.parametrize("shapes, dim, layout, dt, mems, out_mem", CASES)
def test_concat(ttnn_mesh_device, reset_seeds, shapes, dim, layout, dt, mems, out_mem):
    mesh = ttnn_mesh_device
    torch_dtype = torch.float32 if dt == "fp32" else torch.bfloat16
    parts = [U.torch_rand(s, dtype=torch_dtype) for s in shapes]
    tt_parts = [U.to_tt_mem(p, mesh, m, dtype=U.DTYPES[dt], layout=layout) for p, m in zip(parts, mems)]
    refs = [U.seen(t, p, mesh) for t, p in zip(tt_parts, parts)]
    out = ttnn.concat(tt_parts, dim=dim, memory_config=U.resolve_memcfg(out_mem, mesh))
    U.assert_lossless(torch.cat(refs, dim=dim), out, mesh_device=mesh)
