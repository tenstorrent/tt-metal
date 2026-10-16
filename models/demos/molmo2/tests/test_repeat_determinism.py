# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The same request run repeatedly through the server adapter must give identical logits / greedy tokens.

A second (or later) request must not see state left by an earlier one (vision buffers, KV cache, masks, traces).

    MOLMO2_REPEAT_VIDEO=/abs/path/video.mp4 [MOLMO2_REPEAT_N=3] pytest models/demos/molmo2/tests/test_repeat_determinism.py -s
"""

import os

import pytest
import torch
from loguru import logger

from models.demos.molmo2.tests.test_generator_vllm_adapter import (  # noqa: F401
    _vllm_style_inputs,
    adapter,
    images,
    mesh_device,
)

VIDEO = os.environ.get("MOLMO2_REPEAT_VIDEO")
REPEATS = int(os.environ.get("MOLMO2_REPEAT_N", "3"))
DECODE_STEPS = int(os.environ.get("MOLMO2_REPEAT_STEPS", "40"))
PROMPT = [{"type": "video", "video": "v"}, {"type": "text", "text": "Describe what is happening in this video."}]


@pytest.mark.skipif(not VIDEO, reason="set MOLMO2_REPEAT_VIDEO")
def test_repeated_request_is_deterministic(adapter):
    from models.demos.molmo2.tt.hf_loading import load_molmo2_processor

    _, tokens, kwargs = _vllm_style_inputs(load_molmo2_processor(), PROMPT, videos=VIDEO)
    S = tokens.shape[1]
    if os.environ.get("MOLMO2_REPEAT_TRACE_FIRST", "1") == "1":
        # The server captures the decode trace during warm-up, before any real prefill: do the same here.
        adapter.decode_forward(torch.tensor([[0]], dtype=torch.int32), torch.tensor([0]), None, None)
    runs = []
    for r in range(REPEATS):
        logits = adapter.prefill_forward(tokens, None, None, torch.tensor([S]), **kwargs)
        last = logits[0, -1].float().clone()
        toks = [int(last.argmax())]
        for i in range(DECODE_STEPS):
            step = adapter.decode_forward(
                torch.tensor([[toks[-1]]], dtype=torch.int32), torch.tensor([S + i]), None, None
            )
            toks.append(int(step[0].argmax()))
        runs.append((last, toks))
        logger.info(f"run {r}: S={S} first-token logit max={last.max():.4f} tokens={toks[:12]}")
    for r in range(1, REPEATS):
        diff = (runs[r][0] - runs[0][0]).abs().max().item()
        same = runs[r][1] == runs[0][1]
        logger.info(f"run {r} vs 0: prefill logits max|diff|={diff:.5f} greedy tokens identical={same}")
        assert same, f"run {r} differs from run 0: {runs[r][1]} vs {runs[0][1]}"
