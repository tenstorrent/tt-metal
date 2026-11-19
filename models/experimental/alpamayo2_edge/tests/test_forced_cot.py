# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from models.experimental.alpamayo2_edge.reference.forced_cot import (
    DRIVING_COT_TEXT,
    DRIVING_COT_TEXT_TOKENS,
    driving_text_cot,
)
from models.experimental.alpamayo2_edge.reference.inputs import _checkpoint_assets


def test_driving_text_cot(cfg):
    cot = driving_text_cot(cfg)
    tok = cfg.tokens
    assert cot.dtype == torch.int64
    assert cot.shape == (DRIVING_COT_TEXT_TOKENS + 1,)
    assert int(cot[-1]) == tok.future_start
    text = cot[:-1]
    assert bool((text < tok.history_id0).all()), "trajectory or anchor token in the CoT text"
    assert not bool((text == tok.eos).any())
    decoded = _checkpoint_assets(cfg.model_dir).tokenizer.decode(text.tolist(), skip_special_tokens=False)
    assert decoded.startswith(DRIVING_COT_TEXT + DRIVING_COT_TEXT)
    assert torch.equal(driving_text_cot(cfg), cot)
