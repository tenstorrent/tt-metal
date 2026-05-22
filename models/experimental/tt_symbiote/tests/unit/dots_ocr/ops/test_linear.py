# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Phase 2 — full ``ttnn.linear`` PCC sweep across text + vision matrices.

Strategy (per Phase 1 implementer's notes):

* Captured tensor shapes are **per-device** — we build inputs at exactly the
  captured shape and replicate them across the mesh (no pre-sharding at
  test-time).
* Production rows may have width-sharded inputs (DRAM-sharded MM) or
  block-sharded outputs. We do **not** reconstruct the sharded ``MemoryConfig``
  here (the harness ``build_memory_config`` falls back to interleaved DRAM if
  a ``shard_spec`` is present). The matmul runs as DRAM-interleaved instead,
  which still validates numerical correctness.
* For the same reason we **do not** pass the captured ``program_config`` — that
  was tuned for the sharded layout. We let TTNN auto-select. The
  ``compute_kernel_config`` (math fidelity) is preserved so the dtype
  threshold still reflects the production error budget.
* ``transpose_b=True`` is honored — see :class:`TTNNDotsVisionPatchEmbed`.
"""

from __future__ import annotations

from typing import Any, Dict, List

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
    build_compute_kernel_config,
    build_memory_config,
    parse_ttnn_dtype,
    parse_ttnn_layout,
)


# ---------------------------------------------------------------------------
# Phase 1 sanity rows (preserved for regression on the harness happy-path)
# ---------------------------------------------------------------------------

_SANITY_CALL_IDS = [857, 23, 11]


def _select_sanity_rows() -> List[Dict[str, Any]]:
    rows = load_op_matrix("ttnn.linear", include_vision=False, include_text=True)
    by_call_id = {r["call_id"]: r for r in rows}
    selected = []
    for cid in _SANITY_CALL_IDS:
        if cid in by_call_id:
            selected.append(by_call_id[cid])
    if not selected:
        rows.sort(key=lambda r: r["inputs"][0]["shape"][-2] * r["inputs"][0]["shape"][-1])
        selected = rows[:3]
    return selected


_SANITY_ROWS = _select_sanity_rows()


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

_ALL_ROWS = load_op_matrix("ttnn.linear", include_vision=True, include_text=True)


def _force_interleaved_dram(mc_record):
    """Whenever the captured memory_config refers to a sharded layout, fall back
    to DRAM-interleaved so :func:`ttnn.from_torch` succeeds without
    reconstructing a shard spec at test-time.

    Returns a dict that ``build_memory_config`` will parse to
    ``MemoryConfig(INTERLEAVED, DRAM)``.
    """
    return {
        "buffer_type": "BufferType.DRAM",
        "memory_layout": "TensorMemoryLayout.INTERLEAVED",
    }


def _run_linear_row(row: Dict[str, Any], mesh_device, *, is_sanity: bool = False):
    """Common test body for both the sanity and full sweep variants."""
    tags = make_row_tags(row)
    row_id = make_row_id(row)
    op_name = row["op"]

    inputs = row["inputs"]
    kwargs = row.get("kwargs", {}) or {}
    out_records = row.get("output", []) or []

    a_shape = list(inputs[0]["shape"])
    w_shape = list(inputs[1]["shape"])
    a_dtype = parse_ttnn_dtype(inputs[0]["dtype"])
    w_dtype = parse_ttnn_dtype(inputs[1]["dtype"])
    a_layout = parse_ttnn_layout(inputs[0]["layout"])
    w_layout = parse_ttnn_layout(inputs[1]["layout"])

    # Sanity test exercises the harness happy-path; full sweep forces interleaved
    # DRAM for input/weight/output memory configs (see module docstring).
    if is_sanity:
        a_mc = build_memory_config(inputs[0].get("memory_config"))
        w_mc = build_memory_config(inputs[1].get("memory_config"))
        out_mc = build_memory_config(kwargs.get("memory_config"))
    else:
        a_mc = build_memory_config(_force_interleaved_dram(None))
        w_mc = build_memory_config(_force_interleaved_dram(None))
        out_mc = build_memory_config(_force_interleaved_dram(None))

    out_dtype = parse_ttnn_dtype(out_records[0].get("dtype")) if out_records else a_dtype

    ckc_record = kwargs.get("compute_kernel_config") or {}
    ckc = build_compute_kernel_config(ckc_record.get("kind"), ckc_record.get("fields"))

    transpose_b = bool(kwargs.get("transpose_b", False))

    # ---- torch reference ----
    torch.manual_seed(0)
    # Scale down to keep BF16 stable on big tensors (vision QKV is 12288x1536).
    a_torch = torch.randn(*a_shape, dtype=torch.bfloat16) * 0.1
    w_torch = torch.randn(*w_shape, dtype=torch.bfloat16) * 0.1

    if transpose_b:
        ref = torch.nn.functional.linear(a_torch, w_torch)
    else:
        ref = a_torch @ w_torch

    # ---- TT inputs (replicated) ----
    a_tt = ttnn.from_torch(
        a_torch,
        dtype=a_dtype,
        layout=a_layout,
        device=mesh_device,
        memory_config=a_mc,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    w_tt = ttnn.from_torch(
        w_torch,
        dtype=w_dtype,
        layout=w_layout,
        device=mesh_device,
        memory_config=w_mc,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    linear_kwargs = dict(
        memory_config=out_mc,
        dtype=out_dtype,
        compute_kernel_config=ckc,
    )
    if transpose_b:
        linear_kwargs["transpose_b"] = True

    out_tt = ttnn.linear(a_tt, w_tt, **linear_kwargs)

    out_torch = gather_to_torch(
        out_tt,
        mesh_device,
        captured_output_record=out_records[0] if out_records else None,
        strategy="replicated",
    )

    while out_torch.dim() > ref.dim() and out_torch.shape[0] == 1:
        out_torch = out_torch.squeeze(0)
    if out_torch.dim() > ref.dim():
        try:
            out_torch = out_torch.reshape(ref.shape)
        except RuntimeError:
            pass

    threshold = op_pcc_threshold(op_name, [a_dtype, w_dtype], tags["math_fidelity"])
    pcc = assert_op_pcc(
        ref.to(torch.float32),
        out_torch.to(torch.float32),
        threshold=threshold,
        op_name=op_name,
        row_id=row_id,
    )

    print(f"\n[{row_id}] PCC={pcc:.5f} (threshold={threshold:.4f})")

    ttnn.deallocate(out_tt)
    ttnn.deallocate(a_tt)
    ttnn.deallocate(w_tt)


# ---------------------------------------------------------------------------
# Sanity test (3 hand-picked rows — kept passing as a smoke test)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", _SANITY_ROWS, ids=[make_row_id(r) for r in _SANITY_ROWS])
def test_linear_sanity(row, mesh_device_t3k_dp):
    """Three hand-picked text rows — uses captured MemoryConfig (DRAM-interleaved)."""
    _run_linear_row(row, mesh_device_t3k_dp, is_sanity=True)


# ---------------------------------------------------------------------------
# Full sweep (text + vision)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", _ALL_ROWS, ids=[make_row_id(r) for r in _ALL_ROWS])
def test_linear(row, mesh_device_t3k_dp):
    """Full ``ttnn.linear`` sweep — text + vision rows from the dedup matrix.

    The test forces a DRAM-interleaved memory layout regardless of what the
    capture saw (production may have width-sharded the input for DRAM-sharded
    MMs). This validates numerical correctness; perf/layout fidelity is the
    responsibility of the module-level + e2e tests.
    """
    _run_linear_row(row, mesh_device_t3k_dp, is_sanity=False)
