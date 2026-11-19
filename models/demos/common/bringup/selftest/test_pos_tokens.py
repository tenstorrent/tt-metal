# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""F275: a position run's tokens are real text by default (the canonical prompt, repeated); random ids on request."""

import torch

from models.demos.common.bringup.reference import prompt
from models.demos.common.bringup.testing import profile


class _S:
    def layers(self):
        return [0]

    def get(self, k, d=None):
        return 1000


def test_position_tokens_text_by_default(monkeypatch):
    ids = list(range(10, 17))  # a 7-token "prompt"
    monkeypatch.setattr(prompt, "load", lambda s: {"token_ids": ids})
    monkeypatch.setattr(prompt, "tokens", lambda s, n, tok=None: torch.tensor(ids[:n]))
    monkeypatch.delenv(profile.POS_TOKENS_ENV, raising=False)
    t = profile.PositionGolden(_S(), 16, 4).tokens()
    assert t.tolist() == (ids * 3)[:20]
    monkeypatch.setenv(profile.POS_TOKENS_ENV, "random")
    r = profile.PositionGolden(_S(), 16, 4).tokens()
    assert len(r) == 20 and r.tolist() != t.tolist()
