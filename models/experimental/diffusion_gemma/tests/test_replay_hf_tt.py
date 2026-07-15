# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from models.experimental.diffusion_gemma.demo.replay_hf_tt import (
    _layer_hidden_summary,
    _logits_topk_summary,
    _make_replay_noise,
    _tensor_sha256,
)


def test_seeded_replay_noise_is_deterministic_and_nonzero():
    kwargs = {
        "seed": 7,
        "steps": 2,
        "canvas_length": 4,
        "vocab_size": 16,
        "mode": "seeded",
    }

    first_gumbel, first_renoise = _make_replay_noise(**kwargs)
    second_gumbel, second_renoise = _make_replay_noise(**kwargs)

    assert len(first_gumbel) == len(first_renoise) == 2
    assert torch.equal(first_gumbel[0], second_gumbel[0])
    assert torch.equal(first_renoise[1], second_renoise[1])
    assert torch.isfinite(first_gumbel[0]).all()
    assert torch.count_nonzero(first_gumbel[0]) > 0
    assert _tensor_sha256(first_gumbel[0]) == _tensor_sha256(second_gumbel[0])


def test_zero_replay_noise_preserves_clean_argmax_probe():
    gumbel, renoise = _make_replay_noise(
        seed=3,
        steps=1,
        canvas_length=4,
        vocab_size=8,
        mode="zero",
    )

    assert torch.count_nonzero(gumbel[0]) == 0
    assert torch.count_nonzero(renoise[0]) == 0


def test_logits_and_layer_summaries_keep_absolute_positions():
    logits = torch.tensor([[[[1.0, 3.0, 2.0], [4.0, 0.0, 2.0]]]])
    topk = _logits_topk_summary(logits, [0, 1], k=2, step=5)

    assert topk == {
        "step": 5,
        "rows": [
            {"pos": 0, "token_ids": [1, 2], "values": [3.0, 2.0]},
            {"pos": 1, "token_ids": [0, 2], "values": [4.0, 2.0]},
        ],
    }

    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    summary = _layer_hidden_summary([hidden], [hidden.clone()], [2, 4])
    assert summary[0]["pcc"] == 1.0
    assert summary[0]["max_abs"] == 0.0
    assert [row["pos"] for row in summary[0]["per_position"]] == [2, 4]
