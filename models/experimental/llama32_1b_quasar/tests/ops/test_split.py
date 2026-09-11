# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.split``.

Model call site (modules/sampling/sampling_1d.py):
  * L524  x_list       = ttnn.split(x_bf16, x_bf16.shape[-1] // 2, dim=3)
  * L525  indices_list = ttnn.split(self._local_indices, ... // 2, dim=3)

The single-device top-k strategy splits the vocab in half along the last dim
(dim=3) to run two independent top-k passes. Value-preserving -> torch reference
is ``torch.split`` along dim 3; each half compared with PCC 0.999.

NOTE: the QKV projection is *not* split with ttnn.split — it uses
``ttnn.experimental.nlp_create_qkv_heads`` (attention_1d.py:470). The only
real ttnn.split call site is this vocab-halving one, which is what is covered.
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U

# (id, batch, width) — width split in half along dim=3.
_SPLIT_SITES = [
    ("dim2048_half", 32, U.DIM),  # 2048 -> 2 x 1024
    ("dim4096_half", 32, 4096),  # larger vocab-chunk -> 2 x 2048
]


@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, batch, width",
    [pytest.param(*s, id=s[0]) for s in _SPLIT_SITES],
)
def test_split(ttnn_mesh_device, reset_seeds, name, batch, width):
    mesh = ttnn_mesh_device
    half = width // 2

    x_torch = U.torch_rand((1, 1, batch, width))
    x = U.to_tt(x_torch, mesh)

    parts = ttnn.split(x, half, dim=3)
    assert len(parts) == 2, f"expected 2 splits, got {len(parts)}"

    refs = torch.split(x_torch.float(), half, dim=3)
    for i, (ref, part) in enumerate(zip(refs, parts)):
        U.assert_pcc(ref, part, pcc=0.999, mesh_device=mesh)
