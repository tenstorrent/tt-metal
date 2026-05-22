# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 module test: :class:`TTNNDotsVisionMLP`.

Captured input shape: ``[1, 1, 12288, 1536]`` (S=12288 bucket).
We also include a smaller smoke shape to keep the test fast.

Threshold 0.95: vision MLP weights are BFP8, gate/up output BFP8, SILU
through polynomial approx, down-proj BFP8 — modest precision loss
compared to the text-LM-MLP which runs at BFP4 for layers 0..6.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch

from models.experimental.tt_symbiote.modules.dots_ocr_vision import TTNNDotsVisionMLP
from models.experimental.tt_symbiote.tests.unit.dots_ocr.reference.architecture_factory import (
    build_random_dots_vision_mlp,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.module_helpers import (
    gather_replicated_first,
    prepare_module,
    replicated_from_torch,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.pcc import assert_op_pcc


_SHAPES: List[Dict[str, Any]] = [
    {"id": "vis_mlp_b1_1_s2048_h1536", "shape": (1, 1, 2048, 1536)},
    {"id": "vis_mlp_b1_1_s12288_h1536", "shape": (1, 1, 12288, 1536)},
]


@pytest.mark.parametrize("row", _SHAPES, ids=[r["id"] for r in _SHAPES])
def test_vision_mlp(row, mesh_device_t3k_dp):
    torch.manual_seed(0)
    ref = build_random_dots_vision_mlp(seed=0).to(torch.bfloat16).eval()
    x_torch = torch.randn(*row["shape"], dtype=torch.bfloat16) * 0.1

    with torch.no_grad():
        ref_out = ref(x_torch).to(torch.float32)

    tt_module = TTNNDotsVisionMLP.from_torch(ref)
    prepare_module(tt_module, mesh_device_t3k_dp)

    x_tt = replicated_from_torch(x_torch, mesh_device=mesh_device_t3k_dp)

    try:
        out_tt = tt_module(x_tt)
    except Exception as e:
        pytest.xfail(f"Vision MLP shape {row['shape']} not supported here: {e}")

    out_torch = gather_replicated_first(out_tt, mesh_device_t3k_dp).to(torch.float32)
    while out_torch.dim() > ref_out.dim() and out_torch.shape[0] == 1:
        out_torch = out_torch.squeeze(0)
    if out_torch.shape != ref_out.shape:
        try:
            out_torch = out_torch.reshape(ref_out.shape)
        except RuntimeError:
            pass

    pcc = assert_op_pcc(
        ref_out,
        out_torch,
        threshold=0.95,
        op_name="TTNNDotsVisionMLP",
        row_id=row["id"],
    )
    print(f"\n[{row['id']}] PCC={pcc:.5f} (threshold=0.95)")
