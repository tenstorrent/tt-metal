# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-depth prefill logits vs HuggingFace via prefill_paged.
The masked-bucket and chunk-outer paths are checked against prefill_paged elsewhere."""

import torch
from loguru import logger

from models.demos.blackhole.qwen36.tests.test_factory import get_pcc_threshold

from .full_depth_pcc_common import (
    allocate_paged_kv,
    build_full_depth_model,
    hf_reference,
    parametrize_full_depth,
    report,
    tt_prefill_logits,
)


@torch.no_grad()
@parametrize_full_depth()
def test_full_depth_prefill_logits_pcc(mesh_device, reset_seeds, ensure_gc, request):
    """ALL layers, real weights: prefill logits vs HuggingFace."""
    model, tokenizer, token_ids = build_full_depth_model(mesh_device)
    T = token_ids.shape[1]

    hf_prefill, _, _ = hf_reference(model.args.CKPT_DIR, token_ids)

    page_table = allocate_paged_kv(model)
    tt_prefill = tt_prefill_logits(model, token_ids, page_table)

    pcc = report(f"prefill[pos={T - 1}]", hf_prefill, tt_prefill, tokenizer)
    threshold = get_pcc_threshold(request)
    logger.info(
        f"SUMMARY prefill: {model.args.n_layers} layers, prompt={T}, PCC={pcc:.6f} (threshold {threshold}) "
        f"[{'PASS' if pcc >= threshold else 'FAIL'}]"
    )
    assert pcc >= threshold, f"full-depth ({model.args.n_layers}-layer) prefill logits PCC {pcc:.6f} < {threshold}"
