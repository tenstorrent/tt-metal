# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-depth logits PCC of the DFlash verify path vs the HuggingFace reference.

The speculative loop never calls ``prefill_paged``: the prompt and every verify block run through
:class:`~...tt.dflash.target.TtTarget`, i.e. the anchored masked-bucket prefill
(``Qwen36Model.prefill_block_all_logits``) at a bucket-aligned ``chunk_start``, returning the logits
of every row. This checks those logits, at every position, against ``Qwen3_5ForCausalLM`` with all
64 layers and the real checkpoint:

* the prompt, long enough to span two anchor buckets (a whole-bucket forward, then a partial one);
* one 16-token verify block at an unaligned offset, which re-runs its bucket from the anchor.

The block's tokens are the reference sequence's own (teacher forcing), so every row has an HF
counterpart. Traced and eager verify are compared token for token in
``tests/perf/test_dflash_traced_throughput.py`` and in the demo.

The gate is 0.95, the same as the full-depth decode test's and for the same reason: a few rows of
this prompt, where the reference puts nearly all its probability on one token, score lower on a
full-vocab PCC (the near-irrelevant tail dominates) while the argmax still agrees. Those rows score
the same with the anchored bucket and with one bucket from ``chunk_start=0``, so they reflect the
model's bf8/bf16 precision at those positions, not the verify path.

Run::

    MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      pytest -svq models/demos/blackhole/qwen36/tests/test_dflash_verify_pcc.py
"""

import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import get_pcc_threshold
from models.demos.blackhole.qwen36.tests.unit.full_depth_pcc_common import (
    allocate_paged_kv,
    build_full_depth_model,
    hf_reference,
    parametrize_full_depth,
)
from models.demos.blackhole.qwen36.tt.dflash.config import DFlashDrafterConfig, resolve_drafter_path
from models.demos.blackhole.qwen36.tt.dflash.target import TtTarget

PROMPT_LEN = 150
BLOCK_LEN = 16


def _rows_pcc(label, hf_rows, tt_rows):
    """Whole-block PCC plus the worst row and argmax agreement; returns the whole-block PCC."""
    _, pcc = comp_pcc(hf_rows, tt_rows, 0.0)
    per_row = [float(comp_pcc(h, t, 0.0)[1]) for h, t in zip(hf_rows, tt_rows)]
    agree = float((hf_rows.argmax(-1) == tt_rows.argmax(-1)).float().mean())
    logger.info(
        f"{label}: {hf_rows.shape[0]} rows, PCC={float(pcc):.6f}, worst row PCC={min(per_row):.6f}, "
        f"argmax agreement={agree:.4f}"
    )
    return float(pcc)


@torch.no_grad()
@parametrize_full_depth()
def test_dflash_verify_logits_pcc(mesh_device, reset_seeds, ensure_gc, request):
    """ALL layers, real weights: prompt and verify-block logits through TtTarget vs HuggingFace."""
    total = PROMPT_LEN + BLOCK_LEN
    model, _, token_ids = build_full_depth_model(mesh_device, prompt_len=total)
    vocab = model.args.vocab_size

    hf_logits, _, _ = hf_reference(model.args.CKPT_DIR, token_ids, all_positions=True)

    page_table = allocate_paged_kv(model)
    cfg = DFlashDrafterConfig.from_pretrained(resolve_drafter_path())
    target = TtTarget(model, cfg.target_layer_ids, page_table)
    assert PROMPT_LEN > target.ANCHOR, "the prompt must span more than one anchor bucket"
    assert PROMPT_LEN % target.ANCHOR != 0, "the verify block must start at an unaligned offset"
    assert BLOCK_LEN <= target.max_block(PROMPT_LEN), "the verify block must fit its bucket"

    target.reset()
    prompt_logits, _ = target.forward(token_ids[:, :PROMPT_LEN], 0, all_logits=True)
    block_logits, _ = target.forward(token_ids[:, PROMPT_LEN:], PROMPT_LEN, all_logits=True)
    prompt_logits = prompt_logits.float()[0, :, :vocab]
    block_logits = block_logits.float()[0, :, :vocab]
    assert prompt_logits.shape[0] == PROMPT_LEN and block_logits.shape[0] == BLOCK_LEN
    assert not torch.isnan(prompt_logits).any() and not torch.isnan(block_logits).any()

    threshold = get_pcc_threshold(request)
    prompt_pcc = _rows_pcc(f"prompt [0, {PROMPT_LEN})", hf_logits[:PROMPT_LEN], prompt_logits)
    block_pcc = _rows_pcc(f"verify block [{PROMPT_LEN}, {total})", hf_logits[PROMPT_LEN:], block_logits)
    logger.info(
        f"SUMMARY dflash verify: {model.args.n_layers} layers, anchor {target.ANCHOR}, "
        f"prompt PCC={prompt_pcc:.6f}, block PCC={block_pcc:.6f} (threshold {threshold})"
    )
    assert prompt_pcc >= threshold, f"prompt logits PCC {prompt_pcc:.6f} < {threshold}"
    assert block_pcc >= threshold, f"verify-block logits PCC {block_pcc:.6f} < {threshold}"
