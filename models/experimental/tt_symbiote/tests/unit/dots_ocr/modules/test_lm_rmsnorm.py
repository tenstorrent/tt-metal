# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 module test: text-decoder RMSNorm variants.

Covers:

* ``TTNNDotsOCRLocalShardRMSNorm`` — used as ``input_layernorm`` /
  ``post_attention_layernorm`` inside :class:`TTNNDotsOCRDecoderLayer`.
* ``TTNNDistributedRMSNorm`` — used as the final norm
  (``model.norm``) before the LM head.

Reference: random ``Qwen2RMSNorm`` from
``reference.architecture_factory.build_random_qwen2_rmsnorm()``.

Input shapes from ``shape_matrix/text_modules_dedup.json``:

* prefill (1, 14, 1536)  — ``TTNNDotsOCRLocalShardRMSNorm`` inside decoder
* decode  (1, 1, 1536)   — same class, decode shape
* final norm in production runs at the same shapes (post-stack).

Threshold: 0.999 (per PLAN §5.2 — RMSNorm is a near-exact op).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch

from models.experimental.tt_symbiote.modules.dots_ocr_decoder_layer import (
    TTNNDotsOCRLocalShardRMSNorm,
)
from models.experimental.tt_symbiote.modules.normalization import (
    TTNNDistributedRMSNorm,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.reference.architecture_factory import (
    build_random_qwen2_rmsnorm,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.module_helpers import (
    gather_replicated_first,
    prepare_module,
    replicated_from_torch,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.pcc import assert_op_pcc


# Captured per-device input shapes (text matrix).
_SHAPES: List[Dict[str, Any]] = [
    {"id": "lm_localshard_rmsnorm_prefill_b1_s14_h1536", "shape": (1, 14, 1536), "cls": "local"},
    {"id": "lm_localshard_rmsnorm_decode_b1_s1_h1536", "shape": (1, 1, 1536), "cls": "local"},
    {"id": "lm_distributed_rmsnorm_prefill_b1_s14_h1536", "shape": (1, 14, 1536), "cls": "distributed"},
    {"id": "lm_distributed_rmsnorm_decode_b1_s1_h1536", "shape": (1, 1, 1536), "cls": "distributed"},
]


@pytest.mark.parametrize("row", _SHAPES, ids=[r["id"] for r in _SHAPES])
def test_lm_rmsnorm(row, mesh_device_t3k_dp):
    """RMSNorm PCC at the captured prefill/decode shapes."""
    torch.manual_seed(0)

    ref = build_random_qwen2_rmsnorm(seed=0).to(torch.bfloat16).eval()
    x_torch = torch.randn(*row["shape"], dtype=torch.bfloat16) * 0.5

    with torch.no_grad():
        ref_out = ref(x_torch).to(torch.float32)

    if row["cls"] == "local":
        tt_module = TTNNDotsOCRLocalShardRMSNorm.from_torch(ref)
    else:
        tt_module = TTNNDistributedRMSNorm.from_torch(ref)
    prepare_module(tt_module, mesh_device_t3k_dp)

    x_tt = replicated_from_torch(x_torch, mesh_device=mesh_device_t3k_dp)

    out_tt = tt_module(x_tt)
    out_torch = gather_replicated_first(out_tt, mesh_device_t3k_dp).to(torch.float32)
    # Squeeze leading mesh-replicated dim if the gather produced an extra axis.
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
        op_name=("TTNNDotsOCRLocalShardRMSNorm" if row["cls"] == "local" else "TTNNDistributedRMSNorm"),
        row_id=row["id"],
    )
    print(f"\n[{row['id']}] PCC={pcc:.5f} (threshold=0.999)")
