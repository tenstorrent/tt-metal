# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.layer_norm`` PCC sweep — vision only (post-block_stack norm).

The captured weight/bias are shaped ``(1, hidden_dim)`` in TILE layout. The
PyTorch reference uses :func:`torch.nn.functional.layer_norm` against a
logical 1D weight + bias.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
import torch.nn.functional as F
import ttnn

from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.matrix_loader import (
    load_op_matrix,
    make_row_id,
    make_row_tags,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.mesh_gather import (
    gather_to_torch,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.pcc import (
    assert_op_pcc,
    op_pcc_threshold,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.ttnn_kwargs import (
    parse_ttnn_dtype,
)


_LN_ROWS = load_op_matrix("ttnn.layer_norm")


@pytest.mark.parametrize("row", _LN_ROWS, ids=[make_row_id(r) for r in _LN_ROWS])
def test_layer_norm(row: Dict[str, Any], mesh_device_t3k_dp):
    torch.manual_seed(0)

    tags = make_row_tags(row)
    row_id = make_row_id(row)
    op_name = row["op"]
    device = mesh_device_t3k_dp

    inputs = row["inputs"]
    kwargs = row.get("kwargs", {}) or {}

    x_shape = list(inputs[0]["shape"])
    x_dtype = parse_ttnn_dtype(inputs[0]["dtype"])
    hidden = int(x_shape[-1])

    eps = float(kwargs.get("epsilon", 1e-5))

    # ---- torch reference ----
    x_torch = torch.randn(*x_shape, dtype=torch.bfloat16) * 0.5
    weight = torch.randn(hidden, dtype=torch.bfloat16) * 0.1 + 1.0
    bias = torch.randn(hidden, dtype=torch.bfloat16) * 0.05

    ref = F.layer_norm(
        x_torch.to(torch.float32), (hidden,), weight=weight.to(torch.float32), bias=bias.to(torch.float32), eps=eps
    )
    ref = ref.to(torch.bfloat16)

    # ---- TTNN inputs ----
    x_tt = ttnn.from_torch(
        x_torch,
        dtype=x_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    w_tt = ttnn.from_torch(
        weight.reshape(1, hidden),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    b_tt = ttnn.from_torch(
        bias.reshape(1, hidden),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    out_tt = ttnn.layer_norm(x_tt, weight=w_tt, bias=b_tt, epsilon=eps)

    out_torch = gather_to_torch(out_tt, device, strategy="replicated")
    while out_torch.dim() > ref.dim() and out_torch.shape[0] == 1:
        out_torch = out_torch.squeeze(0)
    if out_torch.dim() > ref.dim():
        try:
            out_torch = out_torch.reshape(ref.shape)
        except RuntimeError:
            pass

    threshold = op_pcc_threshold(op_name, [x_dtype], tags["math_fidelity"])
    pcc = assert_op_pcc(
        ref.to(torch.float32),
        out_torch.to(torch.float32),
        threshold=threshold,
        op_name=op_name,
        row_id=row_id,
    )
    print(f"\n[{row_id}] PCC={pcc:.5f} (threshold={threshold:.4f})")

    ttnn.deallocate(out_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(w_tt)
    ttnn.deallocate(b_tt)
