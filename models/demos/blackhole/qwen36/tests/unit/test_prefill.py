# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-depth PREFILL logits PCC vs the HuggingFace reference.

The whole prompt through **every** layer of the real checkpoint via
``prefill_paged`` — the trusted non-traced path the rest of the prefill suite is
validated against — compared with ``Qwen3_5ForCausalLM``'s logits at the same
position. ``tests/test_prefill.py`` (one directory up) checks the masked-bucket and
chunk-outer prefill paths against ``prefill_paged``; this checks ``prefill_paged``
itself against the reference implementation, at full depth.

One test, both checkpoints: ``HF_MODEL`` picks the model and ``MESH_DEVICE`` the
mesh, so the same function covers Qwen3.5-9B (32 layers) and Qwen3.6-27B (64
layers). See ``full_depth_pcc_common.py`` for the harness, the measured PCCs, and
the env knobs.

Run:
  # 9B
  HF_MODEL=Qwen/Qwen3.5-9B MESH_DEVICE=P150 \
    pytest models/demos/blackhole/qwen36/tests/unit/test_prefill.py -v -s

  # 27B
  HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=P150x4 \
    pytest models/demos/blackhole/qwen36/tests/unit/test_prefill.py -v -s
"""

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
