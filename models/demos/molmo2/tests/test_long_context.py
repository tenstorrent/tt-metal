# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Model-card request lengths through the vLLM adapter (prefill_forward / decode_forward).

allenai/Molmo2-8B: max_position_embeddings=36864, video default num_frames=384 (S~34k) and the card
generates up to max_new_tokens=2048. Two checks:

  * test_long_video_card_defaults: a >=384-frame video (MOLMO2_LONG_VIDEO=/path.mp4) through the card's
    default frame sampling, then generation up to min(2048, 36864 - S) tokens; output must not degenerate.
    With MOLMO2_HF_REF (json from make_hf_adapter_reference.py, case "long-video") teacher-forced
    agreement with the HF greedy tokens must be >= 90%.
  * test_max_length_text_prompt: a text prompt that fills the context (36864 - 16 tokens) plus 16 new tokens.

    MOLMO2_LONG_VIDEO=... pytest models/demos/molmo2/tests/test_long_context.py -v -s
"""

import json
import os

import pytest
import torch
from loguru import logger

# Fixtures (mesh_device, adapter, images) and helpers are shared with the adapter test.
from models.demos.molmo2.tests.test_generator_vllm_adapter import (  # noqa: F401
    _server_generate,
    _teacher_forced,
    _vllm_style_inputs,
    adapter,
    images,
    mesh_device,
)

MAX_POSITIONS = 36864
CARD_MAX_NEW_TOKENS = 2048
LONG_VIDEO = os.environ.get("MOLMO2_LONG_VIDEO")


def _repetition_score(tokens, window=256, n=4):
    """Fraction of distinct n-grams in the last `window` tokens (1.0 = no repetition)."""
    tail = tokens[-window:]
    grams = [tuple(tail[i : i + n]) for i in range(max(len(tail) - n + 1, 0))]
    return len(set(grams)) / max(len(grams), 1)


@pytest.mark.skipif(not LONG_VIDEO, reason="set MOLMO2_LONG_VIDEO to a >=192 s video")
def test_long_video_card_defaults(adapter):
    from models.demos.molmo2.tt.hf_loading import load_molmo2_processor

    processor = load_molmo2_processor()
    eos = processor.tokenizer.eos_token_id
    content = [{"type": "video", "video": "v"}, {"type": "text", "text": "Describe what is happening in this video."}]
    out, tokens, kwargs = _vllm_style_inputs(processor, content, videos=LONG_VIDEO)
    S = tokens.shape[1]
    max_new = min(CARD_MAX_NEW_TOKENS, MAX_POSITIONS - S)
    n_frames = out["pixel_values_videos"].shape[0]
    logger.info(f"long video: {n_frames} frames, S={S}, generating up to {max_new} tokens")
    assert n_frames >= 300 and S > 20000, "video too short to exercise the card's long-request regime"
    assert S + max_new <= MAX_POSITIONS

    generated = _server_generate(adapter, tokens, kwargs, max_new, eos)
    text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    score = _repetition_score(generated)
    logger.info(f"generated {len(generated)} tokens, distinct-4gram={score:.2f}: {text[:300]!r}")
    assert len(generated) > 0 and text.strip()
    assert score > 0.3, f"output degenerated into repetition (distinct 4-grams {score:.2f})"

    ref_path = os.environ.get("MOLMO2_HF_REF")
    refs = json.load(open(ref_path)) if ref_path else {}
    if "long-video" in refs:
        ref = refs["long-video"]
        assert ref["S"] == S, f"prompt length {S} != HF {ref['S']}"
        preds = _teacher_forced(adapter, tokens, kwargs, ref["tokens"])
        agree = sum(a == b for a, b in zip(preds, ref["tokens"])) / len(preds)
        logger.info(f"long-video teacher-forced agreement {agree:.0%} over {len(preds)} tokens")
        assert agree >= 0.9


def test_max_length_text_prompt(adapter):
    from models.demos.molmo2.tt.hf_loading import load_molmo2_processor

    processor = load_molmo2_processor()
    tok = processor.tokenizer
    eos = tok.eos_token_id
    new_tokens = 16
    target = MAX_POSITIONS - new_tokens
    paragraph = (
        "The harbour town of Eskerby keeps a ledger of every ship that has ever passed its lighthouse. "
        "The keeper, Marit, records the name, the cargo and the weather, and she has done so for forty years. "
    )
    head = tok.encode("Read the following ledger and answer the question at the end.\n", add_special_tokens=False)
    tail = tok.encode("\nQuestion: Who is the keeper of the lighthouse? Answer in one word.", add_special_tokens=False)
    body = tok.encode(paragraph, add_special_tokens=False)
    ids = head + (body * (target // len(body) + 1))[: target - len(head) - len(tail)] + tail
    assert len(ids) == target
    tokens = torch.tensor([ids], dtype=torch.int32)
    generated = _server_generate(adapter, tokens, {}, new_tokens, eos)
    text = tok.decode(generated, skip_special_tokens=True)
    logger.info(f"S={len(ids)} -> {text!r}")
    assert len(generated) > 0, "no tokens generated at the maximum context length"
