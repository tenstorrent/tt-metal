# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.embedding`` PCC sweep.

Input is INT32/UINT32 token ids. Weight (embedding table) is
``[vocab, hidden_dim]`` BF16. Reference: :func:`torch.nn.functional.embedding`.

We seed integer ids in ``[0, vocab)`` rather than full vocab-range to keep the
test fast (we use a tiny in-test embedding table when the captured table
is the full 151936x1536 LM vocab — the data path is identical).
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


_EMB_ROWS = load_op_matrix("ttnn.embedding")


@pytest.mark.parametrize("row", _EMB_ROWS, ids=[make_row_id(r) for r in _EMB_ROWS])
def test_embedding(row: Dict[str, Any], mesh_device_t3k_dp):
    torch.manual_seed(0)

    tags = make_row_tags(row)
    row_id = make_row_id(row)
    op_name = row["op"]
    device = mesh_device_t3k_dp

    inputs = row["inputs"]

    id_shape = list(inputs[0]["shape"])
    id_dtype = parse_ttnn_dtype(inputs[0]["dtype"])
    w_shape = list(inputs[1]["shape"])
    w_dtype = parse_ttnn_dtype(inputs[1]["dtype"])

    vocab, hidden = int(w_shape[0]), int(w_shape[1])

    # ---- torch reference ----
    ids = torch.randint(0, vocab, id_shape, dtype=torch.int32)
    weight = torch.randn(vocab, hidden, dtype=torch.bfloat16) * 0.1
    ref = F.embedding(ids.to(torch.long), weight)

    # ---- TTNN ----
    ids_tt = ttnn.from_torch(
        ids.to(torch.int32),
        dtype=id_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    w_tt = ttnn.from_torch(
        weight,
        dtype=w_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    out_tt = ttnn.embedding(
        ids_tt,
        w_tt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_torch = gather_to_torch(out_tt, device, strategy="replicated")
    while out_torch.dim() > ref.dim() and out_torch.shape[0] == 1:
        out_torch = out_torch.squeeze(0)
    if out_torch.shape != ref.shape:
        slicer = tuple(slice(0, s) for s in ref.shape)
        out_torch = out_torch[slicer]

    threshold = op_pcc_threshold(op_name, [w_dtype], tags["math_fidelity"])
    pcc = assert_op_pcc(
        ref.to(torch.float32),
        out_torch.to(torch.float32),
        threshold=threshold,
        op_name=op_name,
        row_id=row_id,
    )
    print(f"\n[{row_id}] PCC={pcc:.5f} (threshold={threshold:.4f})")

    ttnn.deallocate(out_tt)
    ttnn.deallocate(ids_tt)
    ttnn.deallocate(w_tt)
