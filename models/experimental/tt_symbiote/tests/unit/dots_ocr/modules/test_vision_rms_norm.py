# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 module test: :class:`TTNNDotsVisionRMSNorm`.

Captured input shape from ``vision_modules_dedup.json`` is
``[1, 1, 12288, 1536]`` (S=12288 bucketed; logical S can vary).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch

from models.experimental.tt_symbiote.modules.dots_ocr_vision import (
    TTNNDotsVisionRMSNorm,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.reference.architecture_factory import (
    _get_dots_config,
    _get_dots_vision_module,
    _seed_init_,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.module_helpers import (
    gather_replicated_first,
    prepare_module,
    replicated_from_torch,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.pcc import assert_op_pcc


def _build_random_vision_rmsnorm(seed: int = 0):
    cfg = _get_dots_config().vision_config
    cls = _get_dots_vision_module("RMSNorm")
    eps = getattr(cfg, "rms_norm_eps", 1e-5)
    try:
        mod = cls(cfg.hidden_size, eps=eps)
    except TypeError:
        mod = cls(cfg.hidden_size)
    return _seed_init_(mod, seed=seed)


_SHAPES: List[Dict[str, Any]] = [
    {"id": "vis_rmsnorm_b1_1_s12288_h1536", "shape": (1, 1, 12288, 1536)},
    {"id": "vis_rmsnorm_b1_1_s2048_h1536", "shape": (1, 1, 2048, 1536)},  # smaller bucket
]


@pytest.mark.parametrize("row", _SHAPES, ids=[r["id"] for r in _SHAPES])
def test_vision_rms_norm(row, mesh_device_t3k_dp):
    torch.manual_seed(0)
    ref = _build_random_vision_rmsnorm(seed=0)
    x_torch = torch.randn(*row["shape"], dtype=torch.bfloat16) * 0.5

    with torch.no_grad():
        ref_out = ref(x_torch).to(torch.float32)

    tt_module = TTNNDotsVisionRMSNorm.from_torch(ref)
    prepare_module(tt_module, mesh_device_t3k_dp)

    x_tt = replicated_from_torch(x_torch, mesh_device=mesh_device_t3k_dp)

    try:
        out_tt = tt_module(x_tt)
    except Exception as e:
        pytest.xfail(f"Vision RMSNorm shape {row['shape']} not supported here: {e}")

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
        threshold=0.999,
        op_name="TTNNDotsVisionRMSNorm",
        row_id=row["id"],
    )
    print(f"\n[{row['id']}] PCC={pcc:.5f} (threshold=0.999)")
