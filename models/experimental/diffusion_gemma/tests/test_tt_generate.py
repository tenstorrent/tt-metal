# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory
from models.experimental.diffusion_gemma.tt.generate import denoise_and_commit_block


class _FakeLogitsFn:
    q_rope_offset = None


def test_denoise_and_commit_block_threads_position_and_commits():
    calls = {}
    committed = torch.tensor([[7, 8, 9]], dtype=torch.long)
    trajectory = DenoiseTrajectory(committed=committed, num_steps=1, halted=True, per_step=[])

    def fake_denoise_block(logits_fn, init_canvas, config, *, gumbel_noise_fn=None, noise_tokens_fn=None):
        calls["denoise"] = (logits_fn, init_canvas, config, gumbel_noise_fn, noise_tokens_fn)
        return trajectory

    def fake_commit(tt_model, canvas_tokens, *, start_pos, page_table=None, page_tables_per_layer=None):
        calls["commit"] = (tt_model, canvas_tokens, start_pos, page_table, page_tables_per_layer)

    logits_fn = _FakeLogitsFn()
    config = DiffusionConfig(canvas_length=3)
    gumbel_noise_fn = object()
    noise_tokens_fn = object()
    page_tables_per_layer = ["pages"]

    out = denoise_and_commit_block(
        "model",
        logits_fn,
        "init-canvas",
        config,
        start_pos=32 + 2 * 256,
        gumbel_noise_fn=gumbel_noise_fn,
        noise_tokens_fn=noise_tokens_fn,
        page_tables_per_layer=page_tables_per_layer,
        denoise_block_fn=fake_denoise_block,
        commit_fn=fake_commit,
    )

    assert logits_fn.q_rope_offset == 544
    assert calls["denoise"] == (logits_fn, "init-canvas", config, gumbel_noise_fn, noise_tokens_fn)
    assert calls["commit"] == ("model", committed, 544, None, page_tables_per_layer)
    assert out.committed is committed
    assert out.next_pos == 547
    assert out.trajectory is trajectory


def test_denoise_and_commit_block_rejects_missing_commit():
    trajectory = DenoiseTrajectory(committed=None, num_steps=0, halted=False, per_step=[])

    with pytest.raises(RuntimeError, match="did not produce committed"):
        denoise_and_commit_block(
            object(),
            object(),
            object(),
            DiffusionConfig(canvas_length=3),
            start_pos=0,
            denoise_block_fn=lambda *args, **kwargs: trajectory,
            commit_fn=lambda *args, **kwargs: None,
        )
