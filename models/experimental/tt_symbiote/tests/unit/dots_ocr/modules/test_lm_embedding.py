# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 module test: :class:`TTNNEmbedding`.

Reference: random ``torch.nn.Embedding(vocab_size, hidden_size)`` re-init
with seed=0 to match :meth:`TTNNEmbedding.from_torch`.

Captured input shapes (text + vision):

* prefill (1, 14)   — token ids
* prefill (1, 2814) — token ids for multimodal prefill
* decode  (1, 1)    — single next-token id per step
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.modules.embedding import TTNNEmbedding
from models.experimental.tt_symbiote.tests.unit.dots_ocr.reference.architecture_factory import (
    _get_dots_config,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.module_helpers import (
    gather_replicated_first,
    prepare_module,
    replicated_from_torch,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.pcc import assert_op_pcc


def _build_random_embedding(seed: int = 0):
    cfg = _get_dots_config()
    emb = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        tmp = torch.empty_like(emb.weight, dtype=torch.float32)
        tmp.normal_(mean=0.0, std=0.02, generator=g)
        emb.weight.copy_(tmp.to(torch.bfloat16))
    return emb.to(torch.bfloat16).eval()


_SHAPES: List[Dict[str, Any]] = [
    {"id": "lm_embedding_prefill_b1_s14", "shape": (1, 14)},
    {"id": "lm_embedding_prefill_b1_s2814", "shape": (1, 2814)},
    {"id": "lm_embedding_decode_b1_s1", "shape": (1, 1)},
]


@pytest.mark.parametrize("row", _SHAPES, ids=[r["id"] for r in _SHAPES])
def test_lm_embedding(row, mesh_device_t3k_dp):
    torch.manual_seed(0)
    ref = _build_random_embedding(seed=0)

    cfg = _get_dots_config()
    # Generate a deterministic batch of token ids
    g = torch.Generator().manual_seed(123)
    ids = torch.randint(0, cfg.vocab_size, row["shape"], dtype=torch.int32, generator=g)

    with torch.no_grad():
        ref_out = ref(ids.long()).to(torch.float32)

    tt_module = TTNNEmbedding.from_torch(ref)
    prepare_module(tt_module, mesh_device_t3k_dp)

    # The capture record shows uint32/int32 ROW_MAJOR layout. Use INT32 to match.
    ids_tt = replicated_from_torch(ids, mesh_device=mesh_device_t3k_dp, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    out_tt = tt_module(ids_tt)
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
        op_name="TTNNEmbedding",
        row_id=row["id"],
    )
    print(f"\n[{row['id']}] PCC={pcc:.5f} (threshold=0.999)")
