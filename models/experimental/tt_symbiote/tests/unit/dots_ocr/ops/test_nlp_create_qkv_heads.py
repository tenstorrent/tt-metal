# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.experimental.nlp_create_qkv_heads`` PCC sweep.

The op takes a fused QKV projection of shape ``[..., seq, (Hq + 2*Hkv) * D]``
and emits three tensors of shape ``[..., Hq, seq, D]`` and
``[..., Hkv, seq, D]``.

The PyTorch reference is :func:`reference.op_reference.nlp_create_qkv_heads`.
We assert PCC independently for Q, K, V.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.tests.unit.dots_ocr.reference.op_reference import (
    nlp_create_qkv_heads as ref_create_qkv_heads,
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


_QKV_ROWS = load_op_matrix("ttnn.experimental.nlp_create_qkv_heads")


@pytest.mark.parametrize("row", _QKV_ROWS, ids=[make_row_id(r) for r in _QKV_ROWS])
def test_nlp_create_qkv_heads(row: Dict[str, Any], mesh_device_t3k_dp):
    torch.manual_seed(0)

    tags = make_row_tags(row)
    row_id = make_row_id(row)
    op_name = row["op"]
    device = mesh_device_t3k_dp

    inputs = row["inputs"]
    kwargs = row.get("kwargs", {}) or {}
    outputs = row.get("output", []) or []

    qkv_shape = list(inputs[0]["shape"])
    qkv_dtype = parse_ttnn_dtype(inputs[0]["dtype"])

    num_heads = int(kwargs["num_heads"])
    num_kv_heads = int(kwargs.get("num_kv_heads", num_heads))

    # Derive head_dim from one of the output Q tensors.
    if outputs:
        q_shape = list(outputs[0]["shape"])
        head_dim = int(q_shape[-1])
    else:
        # Fall back: divide channel by (Hq + 2 Hkv).
        head_dim = int(qkv_shape[-1] // (num_heads + 2 * num_kv_heads))

    # ---- torch reference ----
    qkv = torch.randn(*qkv_shape, dtype=torch.bfloat16) * 0.1
    ref_q, ref_k, ref_v = ref_create_qkv_heads(qkv, num_heads, num_kv_heads, head_dim)

    # ---- TTNN ----
    qkv_tt = ttnn.from_torch(
        qkv,
        dtype=qkv_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    q_tt, k_tt, v_tt = ttnn.experimental.nlp_create_qkv_heads(
        qkv_tt,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=bool(kwargs.get("transpose_k_heads", False)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    threshold = op_pcc_threshold(op_name, [qkv_dtype], tags["math_fidelity"])

    for name, ref_t, tt_t in (
        ("Q", ref_q, q_tt),
        ("K", ref_k, k_tt),
        ("V", ref_v, v_tt),
    ):
        out_torch = gather_to_torch(tt_t, device, strategy="replicated")
        while out_torch.dim() > ref_t.dim() and out_torch.shape[0] == 1:
            out_torch = out_torch.squeeze(0)
        if out_torch.dim() > ref_t.dim():
            try:
                out_torch = out_torch.reshape(ref_t.shape)
            except RuntimeError:
                pass
        pcc = assert_op_pcc(
            ref_t.to(torch.float32),
            out_torch.to(torch.float32),
            threshold=threshold,
            op_name=op_name + ":" + name,
            row_id=row_id,
        )
        print(f"\n[{row_id}::{name}] PCC={pcc:.5f} (threshold={threshold:.4f})")
        ttnn.deallocate(tt_t)

    ttnn.deallocate(qkv_tt)
