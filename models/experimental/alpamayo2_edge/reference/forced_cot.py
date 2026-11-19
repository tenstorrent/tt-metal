# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A fixed long teacher-forcing CoT, shared by the official-stack validation and the TT decode, expert and e2e tests.

Sampled CoTs on synthetic noise images are a handful of tokens; this one exercises decode positions and expert
attention over many CoT rows with the same ids in every test.
"""

from pathlib import Path

import torch

from models.experimental.alpamayo2_edge.reference.config import Alpamayo2EdgeConfig
from models.experimental.alpamayo2_edge.reference.inputs import _checkpoint_assets
from models.experimental.alpamayo2_edge.reference.model import make_forced_cot

DRIVING_COT_TEXT = (
    "The ego vehicle keeps its lane and continues straight at a steady speed, since the road ahead is clear and "
    "no other agents are close. "
)
DRIVING_COT_TEXT_TOKENS = 127


def driving_text_cot(cfg: Alpamayo2EdgeConfig, text_tokens: int = DRIVING_COT_TEXT_TOKENS) -> torch.Tensor:
    """[text_tokens + 1] ids: the first `text_tokens` tokens of the repeated driving text, then the anchor."""
    tokenizer = _checkpoint_assets(Path(cfg.model_dir)).tokenizer
    per_repeat = len(tokenizer.encode(DRIVING_COT_TEXT, add_special_tokens=False).ids)
    text = DRIVING_COT_TEXT * (text_tokens // per_repeat + 2)
    ids = tokenizer.encode(text, add_special_tokens=False).ids[:text_tokens]
    return make_forced_cot(torch.tensor(ids, dtype=torch.int64), cfg.tokens.future_start)
