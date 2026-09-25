# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-depth decode logits vs HuggingFace, teacher-forced with HF's own argmax.
Prefill only populates the KV and GDN state this test then steps."""

import torch
from loguru import logger

from models.demos.blackhole.qwen36.tests.test_factory import get_pcc_threshold

from .full_depth_pcc_common import (
    DECODE_STEPS,
    allocate_paged_kv,
    build_full_depth_model,
    hf_reference,
    parametrize_full_depth,
    report,
    tt_decode_logits,
    tt_prefill_logits,
)


@torch.no_grad()
@parametrize_full_depth()
def test_full_depth_decode_logits_pcc(mesh_device, reset_seeds, ensure_gc, request):
    """ALL layers, real weights: teacher-forced decode-step logits vs HuggingFace."""
    model, tokenizer, token_ids = build_full_depth_model(mesh_device)
    T = token_ids.shape[1]

    hf_prefill, hf_decode, teacher_tokens = hf_reference(model.args.CKPT_DIR, token_ids, decode_steps=DECODE_STEPS)
    logger.info(f"HF teacher-forced decode tokens: {teacher_tokens} ({tokenizer.decode(teacher_tokens)!r})")

    # Prefill PCC is logged here and gated by test_prefill.py.
    page_table = allocate_paged_kv(model)
    report(
        f"prefill[pos={T - 1}] (precondition)", hf_prefill, tt_prefill_logits(model, token_ids, page_table), tokenizer
    )

    pccs = [
        report(
            f"decode[{i}] (pos={T + i}, fed {tok})",
            hf_decode[i],
            tt_decode_logits(model, tok, T + i, page_table),
            tokenizer,
        )
        for i, tok in enumerate(teacher_tokens)
    ]

    threshold = get_pcc_threshold(request)
    below = [(i, p) for i, p in enumerate(pccs) if p < threshold]
    logger.info(
        f"SUMMARY decode: {model.args.n_layers} layers, {len(pccs)} steps from pos={T}, "
        f"PCC min={min(pccs):.6f} mean={sum(pccs) / len(pccs):.6f} (threshold {threshold}), "
        f"steps={[f'{p:.4f}' for p in pccs]} [{'PASS' if not below else 'FAIL'}]"
    )
    detail = ", ".join(f"{i}={p:.6f}" for i, p in below)
    assert not below, f"full-depth ({model.args.n_layers}-layer) decode logits PCC below {threshold} at steps {detail}"
