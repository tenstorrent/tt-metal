# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Elementwise op PCC sweeps: ``ttnn.add``, ``ttnn.mul``, ``ttnn.where``,
``ttnn.typecast``.

One test function per op. Each function parametrizes over its rows from the
combined text + vision dedup matrices. PyTorch reference is the
corresponding ``torch`` operator.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
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
    parse_ttnn_layout,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _gather_match(tt, ref, device) -> torch.Tensor:
    out = gather_to_torch(tt, device, strategy="replicated")
    while out.dim() > ref.dim() and out.shape[0] == 1:
        out = out.squeeze(0)
    if out.shape != ref.shape:
        slicer = tuple(slice(0, s) for s in ref.shape)
        out = out[slicer]
    return out


# ---------------------------------------------------------------------------
# ttnn.add
# ---------------------------------------------------------------------------

_ADD_ROWS = load_op_matrix("ttnn.add")


@pytest.mark.parametrize("row", _ADD_ROWS, ids=[make_row_id(r) for r in _ADD_ROWS])
def test_add(row: Dict[str, Any], mesh_device_t3k_dp):
    torch.manual_seed(0)

    tags = make_row_tags(row)
    row_id = make_row_id(row)
    op_name = row["op"]
    device = mesh_device_t3k_dp

    inputs = row["inputs"]
    if len(inputs) < 2:
        pytest.skip(
            f"row {row_id}: only one TTNN tensor operand captured (scalar "
            "add-with-int); the constant is not in the matrix so we cannot "
            "reconstruct an exact PCC."
        )
    a_shape = list(inputs[0]["shape"])
    b_shape = list(inputs[1]["shape"])
    a_dtype = parse_ttnn_dtype(inputs[0]["dtype"])
    b_dtype = parse_ttnn_dtype(inputs[1]["dtype"])

    a = torch.randn(*a_shape, dtype=torch.bfloat16) * 0.5
    b = torch.randn(*b_shape, dtype=torch.bfloat16) * 0.5
    ref = a + b

    a_tt = ttnn.from_torch(
        a,
        dtype=a_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    b_tt = ttnn.from_torch(
        b,
        dtype=b_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    out_tt = ttnn.add(a_tt, b_tt)
    out = _gather_match(out_tt, ref, device)

    threshold = op_pcc_threshold(op_name, [a_dtype, b_dtype], tags["math_fidelity"])
    pcc = assert_op_pcc(
        ref.to(torch.float32), out.to(torch.float32), threshold=threshold, op_name=op_name, row_id=row_id
    )
    print(f"\n[{row_id}] PCC={pcc:.5f} (threshold={threshold:.4f})")

    ttnn.deallocate(out_tt)
    ttnn.deallocate(a_tt)
    ttnn.deallocate(b_tt)


# ---------------------------------------------------------------------------
# ttnn.mul
# ---------------------------------------------------------------------------

_MUL_ROWS = load_op_matrix("ttnn.mul")


@pytest.mark.parametrize("row", _MUL_ROWS, ids=[make_row_id(r) for r in _MUL_ROWS])
def test_mul(row: Dict[str, Any], mesh_device_t3k_dp):
    torch.manual_seed(0)

    tags = make_row_tags(row)
    row_id = make_row_id(row)
    op_name = row["op"]
    device = mesh_device_t3k_dp

    inputs = row["inputs"]
    a_shape = list(inputs[0]["shape"])
    b_shape = list(inputs[1]["shape"])
    a_dtype = parse_ttnn_dtype(inputs[0]["dtype"])
    b_dtype = parse_ttnn_dtype(inputs[1]["dtype"])

    a = torch.randn(*a_shape, dtype=torch.bfloat16) * 0.5
    b = torch.randn(*b_shape, dtype=torch.bfloat16) * 0.5
    ref = a * b

    a_tt = ttnn.from_torch(
        a,
        dtype=a_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    b_tt = ttnn.from_torch(
        b,
        dtype=b_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    out_tt = ttnn.mul(a_tt, b_tt)
    out = _gather_match(out_tt, ref, device)

    threshold = op_pcc_threshold(op_name, [a_dtype, b_dtype], tags["math_fidelity"])
    pcc = assert_op_pcc(
        ref.to(torch.float32), out.to(torch.float32), threshold=threshold, op_name=op_name, row_id=row_id
    )
    print(f"\n[{row_id}] PCC={pcc:.5f} (threshold={threshold:.4f})")

    ttnn.deallocate(out_tt)
    ttnn.deallocate(a_tt)
    ttnn.deallocate(b_tt)


# ---------------------------------------------------------------------------
# ttnn.where
# ---------------------------------------------------------------------------

_WHERE_ROWS = load_op_matrix("ttnn.where")


@pytest.mark.parametrize("row", _WHERE_ROWS, ids=[make_row_id(r) for r in _WHERE_ROWS])
def test_where(row: Dict[str, Any], mesh_device_t3k_dp):
    torch.manual_seed(0)

    tags = make_row_tags(row)
    row_id = make_row_id(row)
    op_name = row["op"]
    device = mesh_device_t3k_dp

    inputs = row["inputs"]
    cond_shape = list(inputs[0]["shape"])
    a_shape = list(inputs[1]["shape"])
    b_shape = list(inputs[2]["shape"])
    cond_dtype = parse_ttnn_dtype(inputs[0]["dtype"])
    a_dtype = parse_ttnn_dtype(inputs[1]["dtype"])
    b_dtype = parse_ttnn_dtype(inputs[2]["dtype"])

    # Construct a "mostly-true" condition (50/50). For BF16 cond TTNN treats
    # !=0 as true, matching torch.where semantics on float tensors.
    cond = (torch.rand(*cond_shape) > 0.5).to(torch.bfloat16)
    a = torch.randn(*a_shape, dtype=torch.bfloat16) * 0.5
    b = torch.randn(*b_shape, dtype=torch.bfloat16) * 0.5
    # Broadcast cond to a/b shape.
    cond_b = cond.expand_as(a).to(torch.bool)
    ref = torch.where(cond_b, a, b)

    cond_tt = ttnn.from_torch(
        cond,
        dtype=cond_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    a_tt = ttnn.from_torch(
        a,
        dtype=a_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    b_tt = ttnn.from_torch(
        b,
        dtype=b_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    out_tt = ttnn.where(cond_tt, a_tt, b_tt)
    out = _gather_match(out_tt, ref, device)

    threshold = op_pcc_threshold(op_name, [a_dtype, b_dtype], tags["math_fidelity"])
    pcc = assert_op_pcc(
        ref.to(torch.float32), out.to(torch.float32), threshold=threshold, op_name=op_name, row_id=row_id
    )
    print(f"\n[{row_id}] PCC={pcc:.5f} (threshold={threshold:.4f})")

    ttnn.deallocate(out_tt)
    ttnn.deallocate(cond_tt)
    ttnn.deallocate(a_tt)
    ttnn.deallocate(b_tt)


# ---------------------------------------------------------------------------
# ttnn.typecast
# ---------------------------------------------------------------------------

_CAST_ROWS = load_op_matrix("ttnn.typecast")


@pytest.mark.parametrize("row", _CAST_ROWS, ids=[make_row_id(r) for r in _CAST_ROWS])
def test_typecast(row: Dict[str, Any], mesh_device_t3k_dp):
    torch.manual_seed(0)

    tags = make_row_tags(row)
    row_id = make_row_id(row)
    op_name = row["op"]
    device = mesh_device_t3k_dp

    inputs = row["inputs"]
    out_records = row.get("output", []) or []

    x_shape = list(inputs[0]["shape"])
    in_dtype = parse_ttnn_dtype(inputs[0]["dtype"])
    in_layout = parse_ttnn_layout(inputs[0]["layout"])
    out_dtype = parse_ttnn_dtype(out_records[0]["dtype"]) if out_records else in_dtype

    # Casting *down* to BFP4 is inherently lossy (4-bit mantissa group). The
    # captured typecast at cid=21 is the vision attention V-projection going
    # from BFP8 -> BFP4 for the SDPA V-cache; production accepts the precision
    # loss but our op-level threshold (0.999) cannot. xfail to surface it
    # without lowering the threshold globally.
    out_dt_str = str(out_dtype).upper()
    in_dt_str = str(in_dtype).upper()
    if "BFLOAT4" in out_dt_str and "BFLOAT4" not in in_dt_str:
        pytest.xfail(
            f"row {row_id}: typecast to BFP4 is lossy; production-tolerated " "precision drop. Tracked as anomaly."
        )

    # Captured cast is INT32 -> UINT32 for the embedding pre-token-id pathway.
    # Construct ids in a non-negative range so the cast is well-defined.
    is_int = "INT" in str(in_dtype).upper() or "UINT" in str(in_dtype).upper()
    if is_int:
        x = torch.randint(0, 1000, x_shape, dtype=torch.int32)
        torch_ref_dtype = torch.int32 if "INT32" in str(out_dtype).upper() else torch.long
    else:
        x = torch.randn(*x_shape, dtype=torch.bfloat16) * 1.0
        torch_ref_dtype = torch.float32

    ref = x.to(torch_ref_dtype)

    x_tt = ttnn.from_torch(
        x,
        dtype=in_dtype,
        layout=in_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    out_tt = ttnn.typecast(x_tt, out_dtype)

    out = _gather_match(out_tt, ref, device)

    # Use a tight threshold for typecast — it's a value-preserving op.
    threshold = op_pcc_threshold(op_name, [in_dtype], tags["math_fidelity"])
    pcc = assert_op_pcc(
        ref.to(torch.float32), out.to(torch.float32), threshold=threshold, op_name=op_name, row_id=row_id
    )
    print(f"\n[{row_id}] PCC={pcc:.5f} (threshold={threshold:.4f})")

    ttnn.deallocate(out_tt)
    ttnn.deallocate(x_tt)
