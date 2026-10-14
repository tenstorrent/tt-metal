# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.concat``.

Model call site (modules/lm_head/lm_head_1d.py):
  * L154  output = ttnn.concat(outputs, dim=-1, memory_config=cfg.output_memcfg)

The LM head splits the vocab projection into several column chunks, runs a matmul
per chunk, then concatenates the chunk outputs back along the last dim to form the
full logits [1, 1, batch, VOCAB]. Value-preserving -> torch reference is
``torch.cat(dim=-1)``; PCC 0.999.

Chunk sizes here mirror the lm_head split structure (a few unequal tail chunks
along dim -1); the real VOCAB (128256) is split into ~num-device-sized columns.
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U

# (id, batch, chunk_widths) — concat is along dim=-1.
_CONCAT_SITES = [
    ("two_equal", 32, (1024, 1024)),
    ("four_chunks", 32, (512, 512, 512, 512)),
    ("uneven_tail", 32, (1024, 1024, 512)),  # last chunk smaller, as in split_sizes tail
]


@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, batch, chunk_widths",
    [pytest.param(*s, id=s[0]) for s in _CONCAT_SITES],
)
def test_concat(ttnn_mesh_device, reset_seeds, name, batch, chunk_widths):
    mesh = ttnn_mesh_device

    parts_torch = [U.torch_rand((1, 1, batch, w)) for w in chunk_widths]
    parts_tt = [U.to_tt(p, mesh) for p in parts_torch]

    out = ttnn.concat(parts_tt, dim=-1)

    ref = torch.cat([p.float() for p in parts_torch], dim=-1)
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)
