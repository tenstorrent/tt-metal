# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from models.experimental.audiox.reference.conditioners import Conditioner, MultiConditioner


class _StubConditioner(Conditioner):
    """Records calls and returns a fixed-shape (embed, mask) tuple. Lets us
    test MultiConditioner without pulling in HF models."""

    def __init__(self, output_dim: int, seq_len: int):
        super().__init__(output_dim, output_dim)
        self.seq_len = seq_len
        self.last_inputs = None

    def forward(self, inputs, device):
        self.last_inputs = inputs
        batch = len(inputs)
        return (
            torch.zeros(batch, self.seq_len, self.output_dim),
            torch.ones(batch, self.seq_len, dtype=torch.bool),
        )


def test_multi_conditioner_dispatches_each_key():
    text_cond = _StubConditioner(output_dim=8, seq_len=4)
    video_cond = _StubConditioner(output_dim=8, seq_len=2)

    multi = MultiConditioner({"text_prompt": text_cond, "video_prompt": video_cond})

    batch = [
        {"text_prompt": "hello", "video_prompt": torch.zeros(1, 1, 3, 4, 4)},
        {"text_prompt": "world", "video_prompt": torch.zeros(1, 1, 3, 4, 4)},
    ]
    out = multi(batch, "cpu")

    assert set(out.keys()) == {"text_prompt", "video_prompt"}
    assert out["text_prompt"][0].shape == (2, 4, 8)
    assert out["video_prompt"][0].shape == (2, 2, 8)
    assert text_cond.last_inputs == ["hello", "world"]


def test_multi_conditioner_falls_back_to_default_key():
    cond = _StubConditioner(output_dim=8, seq_len=4)
    multi = MultiConditioner({"text_prompt": cond}, default_keys={"text_prompt": "prompt"})
    out = multi([{"prompt": "hi"}], "cpu")
    assert out["text_prompt"][0].shape == (1, 4, 8)
    assert cond.last_inputs == ["hi"]


def test_multi_conditioner_unwraps_singleton_list():
    cond = _StubConditioner(output_dim=8, seq_len=4)
    multi = MultiConditioner({"text_prompt": cond})
    multi([{"text_prompt": ["only"]}], "cpu")
    assert cond.last_inputs == ["only"]


def test_multi_conditioner_raises_on_missing_key():
    cond = _StubConditioner(output_dim=8, seq_len=4)
    multi = MultiConditioner({"text_prompt": cond})
    with pytest.raises(ValueError, match="text_prompt"):
        multi([{"other": "x"}], "cpu")


def test_conditioner_proj_out_identity_when_dims_match():
    c = Conditioner(dim=8, output_dim=8)
    assert isinstance(c.proj_out, nn.Identity)


def test_conditioner_proj_out_linear_when_dims_differ():
    c = Conditioner(dim=4, output_dim=8)
    assert isinstance(c.proj_out, nn.Linear)
    assert c.proj_out.in_features == 4
    assert c.proj_out.out_features == 8
