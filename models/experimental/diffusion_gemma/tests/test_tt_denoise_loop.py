# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.tt import denoise_loop as DL


class _FakeTensor:
    def __init__(self, name):
        self.name = name
        self.deallocated = False

    def deallocate(self, force):
        assert force is True
        assert not self.deallocated, self.name
        self.deallocated = True


def test_denoise_block_deallocates_consumed_injected_noise(monkeypatch):
    gumbel_noise = _FakeTensor("gumbel")
    noise_tokens = _FakeTensor("noise")
    init_canvas = _FakeTensor("init-canvas")
    next_canvas = _FakeTensor("next-canvas")

    result = DL.TtDenoiseStepResult(
        canvas=next_canvas,
        accept_mask=_FakeTensor("accept"),
        entropy=_FakeTensor("entropy"),
        sampled=_FakeTensor("sampled"),
        argmax=_FakeTensor("argmax"),
    )

    def fake_denoise_step(logits, *, temperature, entropy_budget, gumbel_noise, noise_tokens):
        assert logits.name == "logits"
        assert gumbel_noise is not None and not gumbel_noise.deallocated
        assert noise_tokens is not None and not noise_tokens.deallocated
        return result

    monkeypatch.setattr(DL, "denoise_step", fake_denoise_step)
    monkeypatch.setattr(DL, "_ids_to_torch", lambda tensor: torch.ones(1, 1, dtype=torch.long))
    monkeypatch.setattr(DL, "_entropy_to_torch", lambda tensor: torch.zeros(1, 1, dtype=torch.float32))
    monkeypatch.setattr(DL, "_accept_to_torch", lambda tensor: torch.ones(1, 1, dtype=torch.bool))

    trajectory = DL.denoise_block(
        lambda canvas, step: _FakeTensor("logits"),
        init_canvas,
        DiffusionConfig(max_denoise_steps=1, entropy_stop_threshold=1.0, stable_steps_to_halt=0),
        gumbel_noise_fn=lambda step: gumbel_noise,
        noise_tokens_fn=lambda step: noise_tokens,
    )

    assert trajectory.halted
    assert gumbel_noise.deallocated
    assert noise_tokens.deallocated
