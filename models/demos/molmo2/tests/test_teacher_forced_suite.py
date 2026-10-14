# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced agreement with HF (transformers==4.57.1) on a set of videos, plus a long-generation check.

* MOLMO2_SUITE_REF   : json {video_path: {"S", "tokens"}} (ref/hf_suite.py in the bring-up notes): each video is
                       prefilled and HF's greedy tokens are fed through the server decode path (traced).
* MOLMO2_LONG_REF    : json {"S", "tokens"} for MOLMO2_LONG_VIDEO (hundreds of HF tokens): teacher-forced through
                       the traced decode AND the demo's untraced decode (forward_decode_step), reporting agreement
                       per 100-token window.
"""

import json
import os

import pytest
from loguru import logger

from models.demos.molmo2.tests.test_generator_vllm_adapter import (  # noqa: F401
    _teacher_forced,
    _vllm_style_inputs,
    adapter,
    images,
    mesh_device,
)
from models.demos.molmo2.tt.vision_inputs import vision_inputs_from_processor

SUITE_REF = os.environ.get("MOLMO2_SUITE_REF")
LONG_REF = os.environ.get("MOLMO2_LONG_REF")
LONG_VIDEO = os.environ.get("MOLMO2_LONG_VIDEO")
VIDEO_PROMPT = [{"type": "video", "video": "v"}, {"type": "text", "text": "Describe what is happening in this video."}]


@pytest.mark.skipif(not SUITE_REF, reason="set MOLMO2_SUITE_REF")
def test_video_suite_teacher_forced(adapter):
    from models.demos.molmo2.tt.hf_loading import load_molmo2_processor

    refs = json.load(open(SUITE_REF))
    processor = load_molmo2_processor()
    rates = {}
    for video, ref in sorted(refs.items()):
        _, tokens, kwargs = _vllm_style_inputs(processor, VIDEO_PROMPT, videos=video)
        assert tokens.shape[1] == ref["S"], f"{video}: S {tokens.shape[1]} != HF {ref['S']}"
        preds = _teacher_forced(adapter, tokens, kwargs, ref["tokens"])
        rates[video] = sum(a == b for a, b in zip(preds, ref["tokens"])) / len(preds)
        logger.info(f"{os.path.basename(video)[:8]} S={ref['S']} teacher-forced {rates[video]:.0%}")
    mean = sum(rates.values()) / len(rates)
    logger.info(
        f"SUITE mean teacher-forced agreement {mean:.1%} over {len(rates)} videos; min {min(rates.values()):.0%}"
    )
    assert mean >= 0.9


@pytest.mark.skipif(not (LONG_REF and LONG_VIDEO), reason="set MOLMO2_LONG_REF and MOLMO2_LONG_VIDEO")
def test_long_generation_teacher_forced(adapter):
    from models.demos.molmo2.tt.hf_loading import load_molmo2_processor

    ref = json.load(open(LONG_REF))
    processor = load_molmo2_processor()
    out, tokens, kwargs = _vllm_style_inputs(processor, VIDEO_PROMPT, videos=LONG_VIDEO)
    S, ref_tokens = tokens.shape[1], ref["tokens"]
    assert S == ref["S"]
    traced = _teacher_forced(adapter, tokens, kwargs, ref_tokens)

    model = adapter.models[0]
    pv, pool = vision_inputs_from_processor(out)
    logits = model.forward_prefill(
        input_ids=out["input_ids"],
        pixel_values=pv,
        pooled_patches_idx=pool,
        token_type_ids=out.get("token_type_ids"),
        user_id=0,
    )
    untraced = [int(logits[0, -1].argmax())]
    for i, tok in enumerate(ref_tokens[:-1]):
        untraced.append(int(model.forward_decode_step(tok, S + i)[0].argmax()))

    def windows(preds):
        return [
            round(
                sum(a == b for a, b in zip(preds[i : i + 100], ref_tokens[i : i + 100])) / len(ref_tokens[i : i + 100]),
                2,
            )
            for i in range(0, len(ref_tokens), 100)
        ]

    logger.info(f"LONG S={S} n={len(ref_tokens)} traced per-100-token agreement {windows(traced)}")
    logger.info(f"LONG S={S} n={len(ref_tokens)} untraced per-100-token agreement {windows(untraced)}")
    t_all = sum(a == b for a, b in zip(traced, ref_tokens)) / len(ref_tokens)
    u_all = sum(a == b for a, b in zip(untraced, ref_tokens)) / len(ref_tokens)
    logger.info(f"LONG overall traced {t_all:.1%} untraced {u_all:.1%}")
    assert t_all >= 0.9
