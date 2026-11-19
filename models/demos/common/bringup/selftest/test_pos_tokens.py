# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""F275 / F276: a position run's tokens are real text (prose by default, several named texts); random on request;
a run on a named text tags its metrics with _t<name>."""


from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference import texts
from models.demos.common.bringup.testing import profile


class _S:
    def layers(self):
        return [0]

    def get(self, k, d=None):
        return 1000


def test_position_tokens(monkeypatch):
    ids = list(range(10, 17))  # a 7-token "prompt"
    monkeypatch.setattr(
        texts, "build", lambda s, w: {"name": w, "source": "t", "n": 7, "sha256": "x", "token_ids": ids}
    )
    monkeypatch.setattr(metrics, "record", lambda *a, **k: None)
    monkeypatch.delenv(profile.POS_TOKENS_ENV, raising=False)
    g = profile.PositionGolden(_S(), 16, 4)
    assert g.tokens().tolist() == (ids * 3)[:20] and g.text["name"] == "prose" and g.text["repeats"] == 3
    monkeypatch.setattr(
        texts, "build", lambda s, w: {"name": w, "source": "t", "n": 5, "sha256": "y", "token_ids": [1, 2, 3, 4, 5]}
    )  # noqa
    monkeypatch.setenv(profile.POS_TOKENS_ENV, "code")
    assert profile.PositionGolden(_S(), 6, 4).tokens().tolist() == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    monkeypatch.setenv(profile.POS_TOKENS_ENV, "random")
    assert len(profile.PositionGolden(_S(), 16, 4).tokens()) == 20


def test_named_text_tags_metrics(monkeypatch):
    monkeypatch.delenv(metrics.SUFFIX_ENV, raising=False)
    monkeypatch.delenv(texts.ENV, raising=False)
    assert metrics.suffixed("steady_wall_ms_p1") == "steady_wall_ms_p1"
    monkeypatch.setenv(texts.ENV, "agentic")
    assert metrics.suffixed("steady_wall_ms_p1") == "steady_wall_ms_p1_tagentic"
