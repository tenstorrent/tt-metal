# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.experimental.nlp_concat_heads`` + ``..._decode`` PCC sweep.

The prefill variant (:func:`ttnn.experimental.nlp_concat_heads`) is a
straightforward head-merge:

    ``[B, Hq, S, D] -> [B, 1, S, Hq*D]``

The decode variant (:func:`ttnn.experimental.nlp_concat_heads_decode`)
captures a **sharded** input ``[1, 1, num_heads, D]`` (height-sharded on L1
with a single core) and emits a width-sharded ``[1, 1, 32, Hq*D]`` output.
The harness ``build_memory_config`` does NOT reconstruct shard specs (Phase 1
note #4), so we leave the decode rows xfailed pending a Phase-3 shard-spec
parser.

PyTorch reference: :func:`reference.op_reference.nlp_concat_heads`.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.tests.unit.dots_ocr.reference.op_reference import (
    nlp_concat_heads as ref_concat_heads,
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


_CONCAT_ROWS = load_op_matrix("ttnn.experimental.nlp_concat_heads") + load_op_matrix(
    "ttnn.experimental.nlp_concat_heads_decode"
)


def _input_is_sharded(row: Dict[str, Any]) -> bool:
    ins = row.get("inputs", [])
    if not ins:
        return False
    mc = ins[0].get("memory_config", {}) or {}
    ml = (mc.get("memory_layout") or "").upper()
    return "SHARDED" in ml


@pytest.mark.parametrize("row", _CONCAT_ROWS, ids=[make_row_id(r) for r in _CONCAT_ROWS])
def test_nlp_concat_heads(row: Dict[str, Any], mesh_device_t3k_dp):
    torch.manual_seed(0)

    tags = make_row_tags(row)
    row_id = make_row_id(row)
    op_name = row["op"]
    device = mesh_device_t3k_dp

    if _input_is_sharded(row):
        pytest.xfail(
            f"row {row_id}: sharded input MemoryConfig not reconstructible by harness "
            "(see Phase 1 note #4 — extend build_memory_config in Phase 3)."
        )

    inputs = row["inputs"]
    kwargs = row.get("kwargs", {}) or {}

    x_shape = list(inputs[0]["shape"])
    x_dtype = parse_ttnn_dtype(inputs[0]["dtype"])

    # ---- torch reference ----
    x_torch = torch.randn(*x_shape, dtype=torch.bfloat16) * 0.1
    ref = ref_concat_heads(x_torch)

    # ---- TTNN ----
    x_tt = ttnn.from_torch(
        x_torch,
        dtype=x_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    if op_name == "ttnn.experimental.nlp_concat_heads_decode":
        out_tt = ttnn.experimental.nlp_concat_heads_decode(x_tt, num_heads=int(kwargs.get("num_heads", x_shape[1])))
    else:
        out_tt = ttnn.experimental.nlp_concat_heads(x_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    out_torch = gather_to_torch(out_tt, device, strategy="replicated")
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
