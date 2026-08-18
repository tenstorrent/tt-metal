# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.untilize``.

Model call site (models/llama32_1b/model.py):
  * L988  return ttnn.untilize(logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)

The final logits are converted from TILE_LAYOUT to ROW_MAJOR_LAYOUT before being
returned to host. ``untilize`` is a pure layout change (value-preserving), so the
reference is the same tensor round-tripped: build a tile-laid-out tensor, untilize
it, and compare the result to the original torch input; PCC 0.999.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U

# (id, shape) — logits-shaped [1, 1, batch, N]; N kept modest (full VOCAB is large).
_UNTILIZE_SITES = [
    ("decode_batch1", (1, 1, 32, U.DIM)),
    ("decode_batch32", (1, 1, U.MAX_BATCH, U.DIM)),
]


@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, shape",
    [pytest.param(*s, id=s[0]) for s in _UNTILIZE_SITES],
)
def test_untilize(ttnn_mesh_device, reset_seeds, name, shape):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh, layout=ttnn.TILE_LAYOUT)

    out = ttnn.experimental.quasar.untilize(x, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT, f"expected ROW_MAJOR output, got {out.layout}"

    # Value-preserving layout change -> compare against the original tensor.
    U.assert_pcc(x_torch.float(), out, pcc=0.999, mesh_device=mesh)
