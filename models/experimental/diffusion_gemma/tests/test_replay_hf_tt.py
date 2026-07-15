# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from models.experimental.diffusion_gemma.demo.replay_hf_tt import (
    _layer_hidden_summary,
    _logits_topk_summary,
    _make_replay_noise,
    _stage_gate_active_step_indices,
    _tensor_sha256,
    _validate_stage_gate_args,
    build_arg_parser,
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


def test_stage_gate_requires_canonical_production_replay(monkeypatch, expect_error):
    monkeypatch.setenv("DG_SPARSE_MOE", "1")
    args = build_arg_parser().parse_args(["--stage-gate", "--noise-mode", "seeded", "--max-denoising-steps", "8"])
    _validate_stage_gate_args(args)

    args.max_denoising_steps = 1
    with expect_error(ValueError, match="max-denoising-steps 8"):
        _validate_stage_gate_args(args)


def test_stage_gate_rejects_fp32_hf_reference(monkeypatch, expect_error):
    # The bf16-floor self-consistency control uses --hf-dtype float32, but the
    # production gate must keep the bf16 reference (#48291 doc/decision_fidelity).
    monkeypatch.setenv("DG_SPARSE_MOE", "1")
    args = build_arg_parser().parse_args(
        ["--stage-gate", "--noise-mode", "seeded", "--max-denoising-steps", "8", "--hf-dtype", "float32"]
    )
    with expect_error(ValueError, match="bf16 HF reference"):
        _validate_stage_gate_args(args)


def test_stage_gate_stops_entropy_metric_at_common_all_accept():
    active = torch.tensor([[False, True]])
    saturated = torch.tensor([[True, True]])
    hf_traj = SimpleNamespace(
        per_step=[
            SimpleNamespace(accept_mask=active),
            SimpleNamespace(accept_mask=saturated),
            SimpleNamespace(accept_mask=saturated),
        ]
    )
    tt_traj = SimpleNamespace(
        per_step=[
            SimpleNamespace(accept_mask=active),
            SimpleNamespace(accept_mask=saturated),
            SimpleNamespace(accept_mask=saturated),
        ]
    )

    assert _stage_gate_active_step_indices(hf_traj, tt_traj) == [0, 1]
