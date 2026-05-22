# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.argmax`` PCC sweep — token selection at LM-head end-of-graph.

The reference is :func:`torch.argmax`; we compare integer outputs directly
(PCC degenerates for one-element tensors so we use exact-match).
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.matrix_loader import (
    load_op_matrix,
    make_row_id,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.mesh_gather import (
    gather_to_torch,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.ttnn_kwargs import (
    parse_ttnn_dtype,
    parse_ttnn_layout,
)


_ARGMAX_ROWS = load_op_matrix("ttnn.argmax")


@pytest.mark.parametrize("row", _ARGMAX_ROWS, ids=[make_row_id(r) for r in _ARGMAX_ROWS])
def test_argmax(row: Dict[str, Any], mesh_device_t3k_dp):
    torch.manual_seed(0)

    row_id = make_row_id(row)
    device = mesh_device_t3k_dp

    inputs = row["inputs"]
    kwargs = row.get("kwargs", {}) or {}

    x_shape = list(inputs[0]["shape"])
    x_dtype = parse_ttnn_dtype(inputs[0]["dtype"])
    x_layout = parse_ttnn_layout(inputs[0]["layout"])

    dim = int(kwargs.get("dim", -1))
    keepdim = bool(kwargs.get("keepdim", True))
    use_multicore = bool(kwargs.get("use_multicore", True))

    # Use a distinct max at a known position so the test is deterministic.
    x = torch.randn(*x_shape, dtype=torch.bfloat16) * 0.1
    target = 12345  # arbitrary, in-range token id
    if x.shape[dim] > target:
        # Inject a large positive value so argmax along ``dim`` picks ``target``.
        idx = [slice(None)] * x.dim()
        idx[dim] = target
        x[tuple(idx)] = 5.0

    ref = torch.argmax(x.to(torch.float32), dim=dim, keepdim=keepdim).to(torch.int64)

    x_tt = ttnn.from_torch(
        x,
        dtype=x_dtype,
        layout=x_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    out_tt = ttnn.argmax(x_tt, dim=dim, keepdim=keepdim, use_multicore=use_multicore)

    out = gather_to_torch(out_tt, device, strategy="replicated")
    while out.dim() > ref.dim() and out.shape[0] == 1:
        out = out.squeeze(0)
    if out.shape != ref.shape:
        slicer = tuple(slice(0, s) for s in ref.shape)
        out = out[slicer]

    assert torch.equal(out.to(torch.int64), ref), (
        f"\nArgmax mismatch in row_id={row_id!r}\n"
        f"  expected indices = {ref.flatten().tolist()[:8]}\n"
        f"  actual indices   = {out.flatten().tolist()[:8]}\n"
    )
    print(f"\n[{row_id}] argmax exact-match OK (target={target})")

    ttnn.deallocate(out_tt)
    ttnn.deallocate(x_tt)
