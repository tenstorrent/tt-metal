# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-depth DECODE logits PCC vs the HuggingFace reference.

``test_prefill.py`` only gates the prefill pass. This gates the decode path at full
depth: after prefilling the same prompt (which is what leaves the paged KV and each
GDN layer's recurrent + conv state in the shape decode reads), it runs N single-token
steps through the vLLM contract chain — ``prepare_inputs_decode`` →
``ttnn_decode_forward`` → ``process_output_decode`` — and compares each step's logits
with HF's.

Every step is **teacher-forced with HF's own argmax token**, so step ``k``'s PCC
measures step ``k`` and not the compounding of a greedy divergence at step 0. What it
does still accumulate is device state: step ``k`` attends to the KV the prefill and
steps ``0…k-1`` wrote, and carries the GDN state they advanced — which is exactly the
part a single decode step against a randomly pre-filled cache would never gate.

One test, both checkpoints: ``HF_MODEL`` picks the model and ``MESH_DEVICE`` the
mesh, so the same function covers Qwen3.5-9B (32 layers) and Qwen3.6-27B (64
layers). See ``full_depth_pcc_common.py`` for the harness, the measured PCCs, and
the env knobs.

Run:
  # 9B
  HF_MODEL=Qwen/Qwen3.5-9B MESH_DEVICE=P150 \
    pytest models/demos/blackhole/qwen36/tests/unit/test_decode.py -v -s

  # 27B
  HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=P150x4 \
    pytest models/demos/blackhole/qwen36/tests/unit/test_decode.py -v -s
"""

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

    # Prefill is the precondition, not the subject: it populates the paged KV + GDN
    # state the decode steps read. Its PCC is logged (a bad prefill would explain a bad
    # decode) but gated by test_prefill.py, not here.
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
