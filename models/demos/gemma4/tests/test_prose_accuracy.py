# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Long-prompt prefill accuracy of the TT Gemma 4 stack on natural prose against a cached fp32 HF reference.

The 128-token token-accuracy gate (test_token_accuracy.py) never sees a prompt beyond ~400 tokens; the 12B's prefill
SDPA configuration collapsed beyond ~768 positions for weeks without a gate noticing (inventory 4m, 2026-10-24).
This gate prefills the first ``length`` tokens of A Tale of Two Cities in one call and scores every position's
next-token prediction against the HF model's fp32 top-5 ids recorded by ``generate_prose_reference.py``
(``reference_outputs/prose_fp32_<checkpoint>_<length>.refpt``): top-1 agreement over all rows, fp32 top-1 inside the
TT top-5 over all rows, top-1 agreement over the last 64 rows (the long-range read), each against the recorded TT
number minus a margin, and it prints the per-256-row bands. A checkpoint / length without a reference file skips.
Under CI the 1x4 mesh runs; GEMMA4_PROSE_LENGTH picks the reference length (default 2048).

    HF_MODEL=google/gemma-4-12B-it pytest models/demos/gemma4/tests/test_prose_accuracy.py -k 1x4 -s
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger

import ttnn

from .test_factory import parametrize_mesh_with_fabric, skip_if_config_only_checkpoint

# The gate bars per (checkpoint, length): the measured TT agreement on a (1,4) Blackhole column minus 0.03 (twice that
# on the 64-row metric). Measured: 12B 0.7202 / 0.9448 / 0.6250 (CI QB2 and the Galaxy column read the same), 31B
# 0.7266 / 0.9434 / 0.8125 (Galaxy column); the fused qkv-norm-rope prefill default reads above both. HF bf16 vs the
# same fp32 reference: 12B 0.864 / 0.985 / 0.891, 31B 0.812 / 0.970 / 0.891. The .refpt files carry the same bars.
PROSE_BARS = {
    ("gemma-4-12B-it", 2048): {"top1_all": 0.6902, "top5_all": 0.9148, "top1_last64": 0.565},
    ("gemma-4-31B-it", 2048): {"top1_all": 0.698, "top5_all": 0.913, "top1_last64": 0.752},
}
REFERENCE_DIR = Path(__file__).resolve().parent / "reference_outputs"


def _agree(tt_top5, ref_top5, rows, k):
    return float((tt_top5[rows, :k] == ref_top5[rows, :1]).any(-1).double().mean())


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_prose_accuracy(mesh_device, reset_seeds, request):
    import pytest

    skip_if_config_only_checkpoint()

    from models.demos.gemma4.tt.common import create_tt_model

    model_path = os.environ.get("HF_MODEL") or os.environ.get("GEMMA4_MODEL_PATH", "")
    tag = model_path.rstrip("/").split("/")[-1]
    length = int(os.environ.get("GEMMA4_PROSE_LENGTH", "2048"))
    ref_path = REFERENCE_DIR / f"prose_fp32_{tag}_{length}.refpt"
    if not ref_path.exists():
        pytest.skip(f"no prose reference for {tag} at {length} tokens ({ref_path.name})")
    ref = torch.load(ref_path, map_location="cpu")
    ids = ref["input_ids"].long().unsqueeze(0)
    ref_top5 = ref["top5"].long()
    bars = PROSE_BARS.get((tag, length), ref["bars"])  # the visible table first, the file's record as the fallback

    model_args, tt_model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device=mesh_device, max_batch_size=1, max_seq_len=length, model_path=model_path, create_kv_cache=True
    )
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    tokens_tt = ttnn.from_torch(
        ids.to(torch.int32), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=replicate
    )
    embeds = tt_model.embed_tokens(tokens_tt)
    embeds = ttnn.reshape(embeds, (1, 1, length, model_args.hidden_size))
    embeds = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)
    embed_w = state_dict.get("model.language_model.embed_tokens.weight", state_dict.get("model.embed_tokens.weight"))
    embeds_torch = (F.embedding(ids, embed_w) * tt_model.embed_scale).float() if embed_w is not None else None
    tt_logits = tt_model.ttnn_prefill_forward(
        embeds, page_table=None, kv_cache=tt_kv_cache, input_ids_torch=ids, embeds_torch=embeds_torch
    )
    logits = ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[0]).float()
    logits = logits.reshape(-1, logits.shape[-1])[:length]
    tt_logits.deallocate(True)
    tt_top5 = logits.topk(5, dim=-1).indices

    every, last = slice(0, length), slice(length - 64, length)
    top1_all, top5_all = _agree(tt_top5, ref_top5, every, 1), _agree(tt_top5, ref_top5, every, 5)
    top1_last64 = _agree(tt_top5, ref_top5, last, 1)
    bands = " ".join(
        f"[{s}:{min(s + 256, length)}) {_agree(tt_top5, ref_top5, slice(s, min(s + 256, length)), 1):.3f}"
        for s in range(0, length, 256)
    )
    # GEMMA4_PROSE_REPORT_ONLY=1: log the numbers beside the gated row without asserting (a CI arm on the release SKU,
    # e.g. the fused prefill op on); GEMMA4_PROSE_TAG names the row in the log.
    report_only = os.environ.get("GEMMA4_PROSE_REPORT_ONLY") == "1"
    row = os.environ.get("GEMMA4_PROSE_TAG", "").strip() or ("report-only" if report_only else "")
    logger.info(
        f"prose {tag} {length}: top-1 all {top1_all:.4f} (bar {bars['top1_all']}), fp32 top-1 in TT top-5 {top5_all:.4f} "
        f"(bar {bars['top5_all']}), top-1 last 64 {top1_last64:.4f} (bar {bars['top1_last64']}); bands {bands}"
        f"{' [' + row + ']' if row else ''}"
    )
    if report_only:
        return
    assert top1_all >= bars["top1_all"], f"top-1 agreement over all rows {top1_all:.4f} < {bars['top1_all']}"
    assert top5_all >= bars["top5_all"], f"fp32 top-1 in TT top-5 {top5_all:.4f} < {bars['top5_all']}"
    assert top1_last64 >= bars["top1_last64"], f"top-1 over the last 64 rows {top1_last64:.4f} < {bars['top1_last64']}"
