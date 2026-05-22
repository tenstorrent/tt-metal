# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.transformer.scaled_dot_product_attention`` PCC sweep.

Both the text prefill (causal, GQA: 12 Q heads, 2 KV heads) and the vision
prefill (non-causal, MHA: 12 heads) variants are exercised.

PyTorch reference: :func:`reference.op_reference.sdpa` which delegates to
:func:`torch.nn.functional.scaled_dot_product_attention`. For the
captured GQA rows we manually expand K/V to match Q's head count before
running the torch reference (torch's built-in SDPA does NOT natively expand
across the head dim).
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.tests.unit.dots_ocr.reference.op_reference import (
    sdpa as ref_sdpa,
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


_SDPA_ROWS = load_op_matrix("ttnn.transformer.scaled_dot_product_attention")


@pytest.mark.parametrize("row", _SDPA_ROWS, ids=[make_row_id(r) for r in _SDPA_ROWS])
def test_sdpa(row: Dict[str, Any], mesh_device_t3k_dp):
    torch.manual_seed(0)

    tags = make_row_tags(row)
    row_id = make_row_id(row)
    op_name = row["op"]
    device = mesh_device_t3k_dp

    inputs = row["inputs"]
    kwargs = row.get("kwargs", {}) or {}

    q_shape = list(inputs[0]["shape"])
    k_shape = list(inputs[1]["shape"])
    v_shape = list(inputs[2]["shape"])
    q_dtype = parse_ttnn_dtype(inputs[0]["dtype"])
    k_dtype = parse_ttnn_dtype(inputs[1]["dtype"])
    v_dtype = parse_ttnn_dtype(inputs[2]["dtype"])

    is_causal = bool(kwargs.get("is_causal", False))
    scale = kwargs.get("scale", None)
    if isinstance(scale, str):
        try:
            scale = float(scale)
        except ValueError:
            scale = None

    ckc_record = kwargs.get("compute_kernel_config") or {}
    ckc = build_compute_kernel_config(ckc_record.get("kind"), ckc_record.get("fields"))

    # ---- torch reference ----
    q = torch.randn(*q_shape, dtype=torch.bfloat16) * 0.1
    k = torch.randn(*k_shape, dtype=torch.bfloat16) * 0.1
    v = torch.randn(*v_shape, dtype=torch.bfloat16) * 0.1

    # GQA expansion for the torch reference. Captured shapes: Q=[B,Hq,S,D], K/V=[B,Hkv,S,D].
    hq = q.shape[1]
    hkv = k.shape[1]
    if hq != hkv:
        repeat = hq // hkv
        k_ref = k.repeat_interleave(repeat, dim=1)
        v_ref = v.repeat_interleave(repeat, dim=1)
    else:
        k_ref = k
        v_ref = v
    ref = ref_sdpa(
        q.to(torch.float32), k_ref.to(torch.float32), v_ref.to(torch.float32), is_causal=is_causal, scale=scale
    )
    ref = ref.to(torch.bfloat16)

    # ---- TTNN ----
    q_tt = ttnn.from_torch(
        q,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    k_tt = ttnn.from_torch(
        k,
        dtype=k_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    v_tt = ttnn.from_torch(
        v,
        dtype=v_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    sdpa_kwargs = dict(
        is_causal=is_causal,
        compute_kernel_config=ckc,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if scale is not None:
        sdpa_kwargs["scale"] = float(scale)

    out_tt = ttnn.transformer.scaled_dot_product_attention(q_tt, k_tt, v_tt, **sdpa_kwargs)

    out_torch = gather_to_torch(out_tt, device, strategy="replicated")
    while out_torch.dim() > ref.dim() and out_torch.shape[0] == 1:
        out_torch = out_torch.squeeze(0)
    if out_torch.shape != ref.shape:
        slicer = tuple(slice(0, s) for s in ref.shape)
        out_torch = out_torch[slicer]

    threshold = op_pcc_threshold(op_name, [q_dtype, k_dtype, v_dtype], tags["math_fidelity"])
    pcc = assert_op_pcc(
        ref.to(torch.float32),
        out_torch.to(torch.float32),
        threshold=threshold,
        op_name=op_name,
        row_id=row_id,
    )
    print(f"\n[{row_id}] PCC={pcc:.5f} (threshold={threshold:.4f})")

    ttnn.deallocate(out_tt)
    ttnn.deallocate(q_tt)
    ttnn.deallocate(k_tt)
    ttnn.deallocate(v_tt)
