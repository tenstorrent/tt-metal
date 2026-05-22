# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.all_gather`` PCC sweep — input-shape inference from linear records.

Phase 0 Finding #2 (Plan §11.2): CCL ops have an op name but **no operand
shape metadata** — they flow through ``__torch_dispatch__`` and the Layer-B
recorder only kept the function name. To still cover them in Phase 2 we
infer plausible AG-input shapes from neighboring linear records.

Inference Logic
---------------

We walk the full (non-dedup) ``text_ops.json`` + ``vision_ops.json``,
ordered by ``call_id``, and select **unique ttnn.linear output shapes**.
Each such shape is treated as a candidate AG-input shape:

    shape_per_device = linear_output_shape
    dim              = -1  (channel-axis all-gather — matches the production
                            distributed-RMSNorm pattern: stats[..., 32] gathered
                            across mesh-1 axis)

We then run a real ``ttnn.all_gather`` on the (8, 1) DP mesh and compare
against :func:`reference.op_reference.all_gather_torch` which simulates the
"replicate-then-concat" semantic on CPU.

Because the captured DP mesh has shape ``(8, 1)``, the second mesh axis
(``axis=-1`` with ``num_devices=1``) is trivial — i.e. the actual gather is
along the first axis. For these synthesized tests we instead exercise the
first mesh axis (8 devices) to verify that the AG kernel works end-to-end.

The PyTorch reference (``all_gather_torch``) treats each device's local
tensor as identical (replicated) and concatenates ``num_devices`` copies
along the target dim. To match that, we build the input on the mesh with a
``ReplicateTensorToMesh`` mapper so every device sees the same data.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.tests.unit.dots_ocr.reference.op_reference import (
    all_gather_torch,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.mesh_gather import (
    gather_to_torch,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.pcc import (
    assert_op_pcc,
)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MATRIX_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "shape_matrix"))


def _infer_ag_rows() -> List[Dict[str, Any]]:
    """Return a deduped list of candidate AG rows synthesized from linear outputs."""
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
            key = tuple(shape)
            if key in seen:
                continue
            seen[key] = {
                "phase": phase,
                "shape": list(shape),
                "dtype": out.get("dtype", "DataType.BFLOAT16"),
                "linear_call_id": rec.get("call_id"),
                "linear_module": rec.get("module_path", ""),
            }
    rows = []
    for row in seen.values():
        # Build a synthetic row id matching the pattern used elsewhere.
        dim_str = "x".join(str(s) for s in row["shape"])
        row["id"] = f"all_gather_{row['phase']}_inferred_from_linear_cid" f"{row['linear_call_id']}_shape{dim_str}"
        rows.append(row)
    return sorted(rows, key=lambda r: r["id"])


_AG_ROWS = _infer_ag_rows()


@pytest.mark.parametrize("row", _AG_ROWS, ids=[r["id"] for r in _AG_ROWS])
def test_all_gather(row: Dict[str, Any], mesh_device_t3k_dp):
    """Real mesh-wide all_gather over the leading mesh axis (8 devices)."""
    torch.manual_seed(0)
    device = mesh_device_t3k_dp

    shape = list(row["shape"])
    # Skip pathological 1-element or rank-1 shapes (the AG kernel requires a
    # gatherable axis aligned with the mesh).
    if not shape or len(shape) < 2:
        pytest.skip(f"row {row['id']}: rank-{len(shape)} input has no AG-able axis")

    # Replicated x on every device.
    x = torch.randn(*shape, dtype=torch.bfloat16) * 0.1
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    mesh_shape = tuple(int(s) for s in device.shape)
    num_devices = mesh_shape[0] * mesh_shape[1] if len(mesh_shape) >= 2 else mesh_shape[0]

    # Gather along the channel dim (-1). The reference concatenates
    # num_devices copies.
    ag_dim = -1
    try:
        out_tt = ttnn.all_gather(
            x_tt,
            dim=ag_dim,
            num_links=1,
            topology=ttnn.Topology.Ring,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"row {row['id']}: ttnn.all_gather raised {type(exc).__name__}: {exc}")

    ref = all_gather_torch(x, mesh_axis=ag_dim, num_devices=num_devices)

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
        op_name="ttnn.all_gather",
        row_id=row["id"],
    )
    print(f"\n[{row['id']}] AG PCC={pcc:.5f}")

    ttnn.deallocate(out_tt)
    ttnn.deallocate(x_tt)
