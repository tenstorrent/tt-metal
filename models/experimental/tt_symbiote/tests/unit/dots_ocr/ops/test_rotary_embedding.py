# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.experimental.rotary_embedding`` PCC sweep.

The TTNN op takes three operands:

* ``x``           shape ``[B, H, S, D]``
* ``cos`` table   shape ``[1, 1, S, D]``  (broadcast across heads)
* ``sin`` table   shape ``[1, 1, S, D]``

Production cos/sin tables are precomputed; we generate fresh seeded tables
sized to the captured ``x``. PyTorch reference uses
:func:`reference.op_reference.apply_rotary_emb` (half-rotate convention,
matches the Llama family and the production op).

The captured output may have a different ``S`` than the input — for prefill
the TTNN op pads to a tile multiple. We compare against the reference at
the *captured input* sequence length and accept that the gathered TT output
may be longer (we slice it to ``ref.shape``).
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.tests.unit.dots_ocr.reference.op_reference import (
    apply_rotary_emb,
)
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


_ROPE_ROWS = load_op_matrix("ttnn.experimental.rotary_embedding")


@pytest.mark.parametrize("row", _ROPE_ROWS, ids=[make_row_id(r) for r in _ROPE_ROWS])
def test_rotary_embedding(row: Dict[str, Any], mesh_device_t3k_dp):
    torch.manual_seed(0)

    tags = make_row_tags(row)
    row_id = make_row_id(row)
    op_name = row["op"]
    device = mesh_device_t3k_dp

    inputs = row["inputs"]

    x_shape = list(inputs[0]["shape"])
    x_dtype = parse_ttnn_dtype(inputs[0]["dtype"])
    cos_shape = list(inputs[1]["shape"])
    sin_shape = list(inputs[2]["shape"])

    # ---- torch reference (seeded, scaled-down to keep BF16 stable) ----
    x_torch = torch.randn(*x_shape, dtype=torch.bfloat16) * 0.1
    cos = torch.randn(*cos_shape, dtype=torch.bfloat16) * 0.1
    sin = torch.randn(*sin_shape, dtype=torch.bfloat16) * 0.1
    # broadcast cos/sin to x's shape (heads dim)
    ref = apply_rotary_emb(x_torch, cos.to(torch.float32), sin.to(torch.float32))
    ref = ref.to(torch.bfloat16)

    # ---- TTNN ----
    x_tt = ttnn.from_torch(
        x_torch,
        dtype=x_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    cos_tt = ttnn.from_torch(
        cos,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    sin_tt = ttnn.from_torch(
        sin,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    out_tt = ttnn.experimental.rotary_embedding(x_tt, cos_tt, sin_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    out_torch = gather_to_torch(out_tt, device, strategy="replicated")
    # Slice/squeeze to match the reference shape.
    while out_torch.dim() > ref.dim() and out_torch.shape[0] == 1:
        out_torch = out_torch.squeeze(0)
    if out_torch.shape != ref.shape:
        # TTNN may pad S to tile multiple; truncate to the reference S.
        slicer = tuple(slice(0, s) for s in ref.shape)
        out_torch = out_torch[slicer]

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
    ttnn.deallocate(cos_tt)
    ttnn.deallocate(sin_tt)
