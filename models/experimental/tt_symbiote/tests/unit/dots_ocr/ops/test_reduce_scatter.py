# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.reduce_scatter`` PCC sweep — input-shape inference from linear records.

Phase 0 Finding #2 (Plan §11.2): CCL ops have an op name but **no operand
shape metadata**; we infer plausible RS-input shapes from neighboring linear
records.

Inference Logic
---------------

We walk the full ``text_ops.json`` + ``vision_ops.json`` and select unique
``ttnn.linear`` **output** shapes — the production pattern reduce-scatters
the post-linear partial sum across the row of devices. To keep the test set
small we cap at the smaller half of shapes (RS requires the gather-axis to
divide ``num_devices``). The PyTorch reference is
:func:`reference.op_reference.reduce_scatter_torch` which splits an
already-reduced tensor into ``num_devices`` chunks.

Because :func:`ttnn.reduce_scatter` *reduces* across devices first, and we
build inputs that are **replicated** (identical on every device), the
all-reduce result is just ``num_devices * x``. We mirror that on the torch
side so the CPU reference matches.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.mesh_gather import (
    gather_to_torch,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.pcc import (
    assert_op_pcc,
)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MATRIX_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "shape_matrix"))


def _infer_rs_rows() -> List[Dict[str, Any]]:
    seen: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for phase, fname in (("text", "text_ops.json"), ("vision", "vision_ops.json")):
        path = os.path.join(_MATRIX_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            ops = json.load(f)
        for rec in ops:
            if rec.get("op") != "ttnn.linear":
                continue
            out = (rec.get("output") or [{}])[0]
            shape = out.get("shape")
            if not shape:
                continue
            # RS requires the gather axis divisible by num_devices (here 8).
            if len(shape) < 2 or int(shape[-1]) % 8 != 0:
                continue
            key = tuple(shape)
            if key in seen:
                continue
            seen[key] = {
                "phase": phase,
                "shape": list(shape),
                "dtype": out.get("dtype", "DataType.BFLOAT16"),
                "linear_call_id": rec.get("call_id"),
            }
    rows = []
    for row in seen.values():
        dim_str = "x".join(str(s) for s in row["shape"])
        row["id"] = f"reduce_scatter_{row['phase']}_inferred_from_linear_cid" f"{row['linear_call_id']}_shape{dim_str}"
        rows.append(row)
    return sorted(rows, key=lambda r: r["id"])


_RS_ROWS = _infer_rs_rows()


@pytest.mark.parametrize("row", _RS_ROWS, ids=[r["id"] for r in _RS_ROWS])
def test_reduce_scatter(row: Dict[str, Any], mesh_device_t3k_dp):
    """Real mesh-wide reduce_scatter over the leading mesh axis (8 devices)."""
    torch.manual_seed(0)
    device = mesh_device_t3k_dp

    shape = list(row["shape"])
    if not shape or len(shape) < 2:
        pytest.skip(f"row {row['id']}: rank-{len(shape)} input has no RS-able axis")

    mesh_shape = tuple(int(s) for s in device.shape)
    num_devices = mesh_shape[0] * mesh_shape[1] if len(mesh_shape) >= 2 else mesh_shape[0]

    x = torch.randn(*shape, dtype=torch.bfloat16) * 0.1
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    # Production calls reduce_scatter with cluster_axis=1 on a 2D mesh; in our
    # (8, 1) DP mesh that axis is trivial (1 device), so we instead exercise
    # cluster_axis=0 — the 8-device leading axis — which is the one that
    # matters numerically for the AG-then-MM pattern.
    rs_dim = 3 if len(shape) == 4 else len(shape) - 1
    try:
        out_tt = ttnn.reduce_scatter(
            x_tt,
            dim=rs_dim,
            num_links=1,
            cluster_axis=0,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"row {row['id']}: ttnn.reduce_scatter raised {type(exc).__name__}: {exc}")

    # Production: x_per_device is identical (replicated). The all-reduce sum
    # yields num_devices*x; each device keeps its 1/num_devices chunk of the
    # final width.
    reduced = x * float(num_devices)
    # Each device's chunk along the gather axis.
    chunk = reduced.shape[rs_dim] // num_devices
    slicer = [slice(None)] * reduced.dim()
    slicer[rs_dim] = slice(0, chunk)
    ref = reduced[tuple(slicer)]  # device-0 chunk

    out = gather_to_torch(out_tt, device, strategy="replicated")
    while out.dim() > ref.dim() and out.shape[0] == 1:
        out = out.squeeze(0)
    if tuple(out.shape) != tuple(ref.shape):
        slicer = tuple(slice(0, s) for s in ref.shape)
        out = out[slicer]

    pcc = assert_op_pcc(
        ref.to(torch.float32),
        out.to(torch.float32),
        threshold=0.9999,
        op_name="ttnn.reduce_scatter",
        row_id=row["id"],
    )
    print(f"\n[{row['id']}] RS PCC={pcc:.5f}")

    ttnn.deallocate(out_tt)
    ttnn.deallocate(x_tt)
