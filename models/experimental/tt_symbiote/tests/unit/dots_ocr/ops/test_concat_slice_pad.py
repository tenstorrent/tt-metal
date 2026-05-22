# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Layout-movement ops: ``ttnn.concat``, ``ttnn.slice``, ``ttnn.pad``.

These are bit-exact (modulo dtype quantization in the tile, which we mirror
on the torch side) — we use a **tight** PCC threshold of ``0.9999``.

Slice
-----

The capture does not record the start/end indices — pytorch ``slice`` was
invoked positionally and our recorder only kept named kwargs. We infer the
slice as ``input[:out_shape[0], :out_shape[1], ...]`` — i.e. take the first
``out_shape[i]`` elements along each dimension. This matches every captured
shape pair (e.g. ``[1, 32]`` -> ``[1, 14]``).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch
import torch.nn.functional as F
import ttnn

from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.matrix_loader import (
    load_op_matrix,
    make_row_id,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.mesh_gather import (
    gather_to_torch,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.pcc import (
    assert_op_pcc,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.ttnn_kwargs import (
    parse_ttnn_dtype,
    parse_ttnn_layout,
)


_TIGHT_THRESHOLD = 0.9999


def _gather_match(tt, ref_shape, device) -> torch.Tensor:
    out = gather_to_torch(tt, device, strategy="replicated")
    while out.dim() > len(ref_shape) and out.shape[0] == 1:
        out = out.squeeze(0)
    if tuple(out.shape) != tuple(ref_shape):
        slicer = tuple(slice(0, s) for s in ref_shape)
        out = out[slicer]
    return out


# ---------------------------------------------------------------------------
# ttnn.concat
# ---------------------------------------------------------------------------

_CONCAT_ROWS = load_op_matrix("ttnn.concat")


@pytest.mark.parametrize("row", _CONCAT_ROWS, ids=[make_row_id(r) for r in _CONCAT_ROWS])
def test_concat(row: Dict[str, Any], mesh_device_t3k_dp):
    torch.manual_seed(0)

    row_id = make_row_id(row)
    op_name = row["op"]
    device = mesh_device_t3k_dp

    inputs = row["inputs"]
    kwargs = row.get("kwargs", {}) or {}
    dim = int(kwargs.get("dim", -1))

    is_integer = "INT" in inputs[0]["dtype"].upper() and "BFLOAT" not in inputs[0]["dtype"].upper()

    torch_tensors: List[torch.Tensor] = []
    tt_tensors = []
    for i, in_rec in enumerate(inputs):
        shape = list(in_rec["shape"])
        dtype = parse_ttnn_dtype(in_rec["dtype"])
        layout = parse_ttnn_layout(in_rec["layout"])
        if is_integer:
            x = torch.randint(0, 100, shape, dtype=torch.int32)
        else:
            x = torch.randn(*shape, dtype=torch.bfloat16) * 0.5
        torch_tensors.append(x)
        tt_tensors.append(
            ttnn.from_torch(
                x,
                dtype=dtype,
                layout=layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        )

    ref = torch.cat(torch_tensors, dim=dim)
    out_tt = ttnn.concat(tt_tensors, dim=dim)
    out = _gather_match(out_tt, ref.shape, device)

    pcc = assert_op_pcc(
        ref.to(torch.float32),
        out.to(torch.float32),
        threshold=_TIGHT_THRESHOLD,
        op_name=op_name,
        row_id=row_id,
    )
    print(f"\n[{row_id}] PCC={pcc:.5f} (threshold={_TIGHT_THRESHOLD:.4f})")

    ttnn.deallocate(out_tt)
    for t in tt_tensors:
        ttnn.deallocate(t)


# ---------------------------------------------------------------------------
# ttnn.slice
# ---------------------------------------------------------------------------

_SLICE_ROWS = load_op_matrix("ttnn.slice")


@pytest.mark.parametrize("row", _SLICE_ROWS, ids=[make_row_id(r) for r in _SLICE_ROWS])
def test_slice(row: Dict[str, Any], mesh_device_t3k_dp):
    torch.manual_seed(0)

    row_id = make_row_id(row)
    op_name = row["op"]
    device = mesh_device_t3k_dp

    inputs = row["inputs"]
    outputs = row.get("output", []) or []

    in_shape = list(inputs[0]["shape"])
    in_dtype = parse_ttnn_dtype(inputs[0]["dtype"])
    in_layout = parse_ttnn_layout(inputs[0]["layout"])
    out_shape = list(outputs[0]["shape"]) if outputs else in_shape

    is_integer = "INT" in inputs[0]["dtype"].upper() and "BFLOAT" not in inputs[0]["dtype"].upper()

    if is_integer:
        x = torch.randint(0, 1000, in_shape, dtype=torch.int32)
    else:
        x = torch.randn(*in_shape, dtype=torch.bfloat16) * 0.5

    # Inferred slice: take [:out_shape[i]] along each dim.
    starts = [0] * len(in_shape)
    ends = [int(s) for s in out_shape]
    ref = x[tuple(slice(starts[i], ends[i]) for i in range(len(in_shape)))]

    x_tt = ttnn.from_torch(
        x,
        dtype=in_dtype,
        layout=in_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    out_tt = ttnn.slice(x_tt, starts, ends)
    out = _gather_match(out_tt, ref.shape, device)

    pcc = assert_op_pcc(
        ref.to(torch.float32),
        out.to(torch.float32),
        threshold=_TIGHT_THRESHOLD,
        op_name=op_name,
        row_id=row_id,
    )
    print(f"\n[{row_id}] PCC={pcc:.5f} (threshold={_TIGHT_THRESHOLD:.4f})")

    ttnn.deallocate(out_tt)
    ttnn.deallocate(x_tt)


# ---------------------------------------------------------------------------
# ttnn.pad
# ---------------------------------------------------------------------------

_PAD_ROWS = load_op_matrix("ttnn.pad")


def _ttnn_padding_to_torch(padding: List[List[int]]) -> List[int]:
    """``[[before, after], ...]`` (front-to-back dim order) -> ``F.pad`` arg
    ``[..., before_dim_last, after_dim_last, before_dim_second_to_last, ...]``."""
    pairs = list(padding)
    flat: List[int] = []
    for pair in reversed(pairs):
        flat.extend([int(pair[0]), int(pair[1])])
    return flat


@pytest.mark.parametrize("row", _PAD_ROWS, ids=[make_row_id(r) for r in _PAD_ROWS])
def test_pad(row: Dict[str, Any], mesh_device_t3k_dp):
    torch.manual_seed(0)

    row_id = make_row_id(row)
    op_name = row["op"]
    device = mesh_device_t3k_dp

    inputs = row["inputs"]
    kwargs = row.get("kwargs", {}) or {}

    in_shape = list(inputs[0]["shape"])
    in_dtype = parse_ttnn_dtype(inputs[0]["dtype"])
    in_layout = parse_ttnn_layout(inputs[0]["layout"])

    padding = kwargs.get("padding", [])
    value = kwargs.get("value", 0.0)
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            value = 0.0

    is_integer = "INT" in inputs[0]["dtype"].upper() and "BFLOAT" not in inputs[0]["dtype"].upper()
    if is_integer:
        x = torch.randint(0, 1000, in_shape, dtype=torch.int32)
        ref = F.pad(x, _ttnn_padding_to_torch(padding), value=int(value))
    else:
        x = torch.randn(*in_shape, dtype=torch.bfloat16) * 0.5
        ref = F.pad(x, _ttnn_padding_to_torch(padding), value=float(value))

    x_tt = ttnn.from_torch(
        x,
        dtype=in_dtype,
        layout=in_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    # ttnn.pad signature: (input, padding=[(b,a),...], value=...)
    pad_arg = [(int(p[0]), int(p[1])) for p in padding]
    out_tt = ttnn.pad(x_tt, pad_arg, value=value)
    out = _gather_match(out_tt, ref.shape, device)

    pcc = assert_op_pcc(
        ref.to(torch.float32),
        out.to(torch.float32),
        threshold=_TIGHT_THRESHOLD,
        op_name=op_name,
        row_id=row_id,
    )
    print(f"\n[{row_id}] PCC={pcc:.5f} (threshold={_TIGHT_THRESHOLD:.4f})")

    ttnn.deallocate(out_tt)
    ttnn.deallocate(x_tt)
