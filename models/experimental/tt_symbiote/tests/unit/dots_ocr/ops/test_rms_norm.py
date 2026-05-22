# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.rms_norm`` PCC sweep — text + vision rows.

Notes
-----

* The capture only contains ``ttnn.rms_norm`` (no ``rms_norm_pre_all_gather``
  or ``rms_norm_post_all_gather``). The production distributed RMSNorm uses
  the pre/post variants only when the mesh has more than one column
  (``device.shape[-1] > 1``) — under the captured (8, 1) DP mesh the single
  ``ttnn.rms_norm`` path is taken. See
  :class:`TTNNDistributedRMSNorm.forward` and §11 of the bottom-up plan.
* The captured **weight** is the "distributed" layout
  ``(1, 1, hidden_dim // 32, 32)`` in ``ROW_MAJOR``. The data is just the
  logical 1D weight tensor reshaped — so the PyTorch reference builds the 1D
  vector, applies a regular RMSNorm, and we reshape that same 1D vector for
  the TTNN call. PCC is asserted post-gather.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.tests.unit.dots_ocr.reference.op_reference import (
    rms_norm as ref_rms_norm,
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
    build_compute_kernel_config,
    parse_ttnn_dtype,
)


_RMS_ROWS = (
    load_op_matrix("ttnn.rms_norm")
    + load_op_matrix("ttnn.rms_norm_pre_all_gather")
    + load_op_matrix("ttnn.rms_norm_post_all_gather")
)


@pytest.mark.parametrize("row", _RMS_ROWS, ids=[make_row_id(r) for r in _RMS_ROWS])
def test_rms_norm(row: Dict[str, Any], mesh_device_t3k_dp):
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

    eps = float(kwargs.get("epsilon", 1e-6))
    weight_record = kwargs.get("weight")

    ckc_record = kwargs.get("compute_kernel_config") or {}
    ckc = build_compute_kernel_config(ckc_record.get("kind"), ckc_record.get("fields"))

    # ---- torch reference (logical 1D weight against the input) ----
    x_torch = torch.randn(*x_shape, dtype=torch.bfloat16) * 0.5
    w_logical = torch.randn(hidden, dtype=torch.bfloat16) * 0.1 + 1.0
    ref = ref_rms_norm(x_torch, w_logical, eps=eps)

    # ---- TTNN inputs ----
    x_tt = ttnn.from_torch(
        x_torch,
        dtype=x_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    # The captured weight is shaped (1, 1, hidden//32, 32) in ROW_MAJOR. Build
    # exactly the same shape using w_logical so the torch & ttnn paths share
    # the same scalar weights.
    if weight_record and isinstance(weight_record, dict):
        w_shape_captured = list(weight_record.get("shape", []))
    else:
        # Fall back to standard distributed layout.
        w_shape_captured = [1, 1, hidden // 32, 32]

    # Reshape the logical 1D weight into the captured layout.
    if len(w_shape_captured) == 4 and w_shape_captured[-1] == 32 and w_shape_captured[-2] * 32 == hidden:
        w_tile = w_logical.reshape(1, 1, hidden // 32, 32)
    else:
        # Defensive: try to match captured shape directly if total elements line up.
        total = 1
        for d in w_shape_captured:
            total *= int(d)
        if total == hidden:
            w_tile = w_logical.reshape(*w_shape_captured)
        else:
            pytest.skip(f"Cannot map captured weight shape {w_shape_captured} to hidden={hidden}")

    w_dtype = parse_ttnn_dtype(weight_record.get("dtype")) if isinstance(weight_record, dict) else ttnn.bfloat16
    w_tt = ttnn.from_torch(
        w_tile,
        dtype=w_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    # ---- run ttnn.rms_norm ----
    if op_name == "ttnn.rms_norm":
        out_tt = ttnn.rms_norm(
            x_tt,
            weight=w_tt,
            epsilon=eps,
            compute_kernel_config=ckc,
        )
    elif op_name == "ttnn.rms_norm_pre_all_gather":
        out_tt = ttnn.rms_norm_pre_all_gather(
            x_tt,
            dtype=x_dtype,
            compute_kernel_config=ckc,
        )
    elif op_name == "ttnn.rms_norm_post_all_gather":
        # Production path requires stats input. We don't have a captured shape
        # for it; skip until we synthesize one in Phase 3.
        pytest.skip(f"{op_name} needs synthesized stats input — Phase 3")
    else:  # pragma: no cover
        pytest.skip(f"Unsupported op {op_name}")

    out_torch = gather_to_torch(out_tt, device, strategy="replicated")

    # Squeeze any extra leading replicated dim.
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
