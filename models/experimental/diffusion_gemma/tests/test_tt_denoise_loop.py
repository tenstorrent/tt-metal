# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

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


def test_to_host_torch_uses_first_device_tensor_for_mesh_readback(monkeypatch):
    class _FakeMeshDevice:
        def get_num_devices(self):
            return 4

    class _FakeMeshTensor:
        def device(self):
            return _FakeMeshDevice()

    mesh_tensor = _FakeMeshTensor()
    shard0 = _FakeTensor("shard0")
    calls = []

    class _FakeTtnn:
        @staticmethod
        def get_device_tensors(tensor):
            assert tensor is mesh_tensor
            return [shard0]

        @staticmethod
        def to_torch(tensor):
            calls.append(tensor.name)
            assert tensor is shard0
            return torch.tensor([7])

    monkeypatch.setattr(DL, "ttnn", _FakeTtnn)

    assert torch.equal(DL._to_host_torch(mesh_tensor), torch.tensor([7]))
    assert calls == ["shard0"]


@pytest.mark.parametrize(
    "gumbel_noise_fn,noise_tokens_fn",
    [
        (None, lambda step: _FakeTensor("noise")),
        (lambda step: _FakeTensor("gumbel"), None),
        (None, None),
    ],
)
def test_denoise_block_requires_injected_noise_hooks(gumbel_noise_fn, noise_tokens_fn):
    with pytest.raises(ValueError, match="requires injected gumbel_noise_fn and noise_tokens_fn"):
        DL.denoise_block(
            lambda canvas, step: _FakeTensor("logits"),
            _FakeTensor("init-canvas"),
            DiffusionConfig(max_denoise_steps=1),
            gumbel_noise_fn=gumbel_noise_fn,
            noise_tokens_fn=noise_tokens_fn,
        )


def test_denoise_block_deallocates_consumed_injected_noise(monkeypatch):
    gumbel_noise = _FakeTensor("gumbel")
    noise_tokens = _FakeTensor("noise")
    logits = _FakeTensor("logits")
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
        lambda canvas, step: logits,
        init_canvas,
        DiffusionConfig(max_denoise_steps=1, entropy_stop_threshold=1.0, stable_steps_to_halt=0),
        gumbel_noise_fn=lambda step: gumbel_noise,
        noise_tokens_fn=lambda step: noise_tokens,
    )

    assert trajectory.halted
    assert gumbel_noise.deallocated
    assert noise_tokens.deallocated
    assert logits.deallocated


def test_denoise_block_allows_argmax_sampling_without_gumbel_tensor(monkeypatch):
    noise_tokens = _FakeTensor("noise")
    logits = _FakeTensor("logits")
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
        assert gumbel_noise is None
        assert noise_tokens is not None and not noise_tokens.deallocated
        return result

    monkeypatch.setattr(DL, "denoise_step", fake_denoise_step)
    monkeypatch.setattr(DL, "_ids_to_torch", lambda tensor: torch.ones(1, 1, dtype=torch.long))
    monkeypatch.setattr(DL, "_entropy_to_torch", lambda tensor: torch.zeros(1, 1, dtype=torch.float32))
    monkeypatch.setattr(DL, "_accept_to_torch", lambda tensor: torch.ones(1, 1, dtype=torch.bool))

    trajectory = DL.denoise_block(
        lambda canvas, step: logits,
        init_canvas,
        DiffusionConfig(max_denoise_steps=1, entropy_stop_threshold=1.0, stable_steps_to_halt=0),
        gumbel_noise_fn=lambda step: None,
        noise_tokens_fn=lambda step: noise_tokens,
    )

    assert trajectory.halted
    assert noise_tokens.deallocated
    assert logits.deallocated


def test_denoise_block_allows_descriptor_gumbel_without_deallocate(monkeypatch):
    descriptor = object()
    noise_tokens = _FakeTensor("noise")
    logits = _FakeTensor("logits")
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
        assert gumbel_noise is descriptor
        assert noise_tokens is not None and not noise_tokens.deallocated
        return result

    monkeypatch.setattr(DL, "denoise_step", fake_denoise_step)
    monkeypatch.setattr(DL, "_ids_to_torch", lambda tensor: torch.ones(1, 1, dtype=torch.long))
    monkeypatch.setattr(DL, "_entropy_to_torch", lambda tensor: torch.zeros(1, 1, dtype=torch.float32))
    monkeypatch.setattr(DL, "_accept_to_torch", lambda tensor: torch.ones(1, 1, dtype=torch.bool))

    trajectory = DL.denoise_block(
        lambda canvas, step: logits,
        init_canvas,
        DiffusionConfig(max_denoise_steps=1, entropy_stop_threshold=1.0, stable_steps_to_halt=0),
        gumbel_noise_fn=lambda step: descriptor,
        noise_tokens_fn=lambda step: noise_tokens,
    )

    assert trajectory.halted
    assert noise_tokens.deallocated
    assert logits.deallocated


def test_denoise_block_leaves_callback_owned_logits_for_self_conditioning(monkeypatch):
    gumbel_noise = _FakeTensor("gumbel")
    noise_tokens = _FakeTensor("noise")
    logits = _FakeTensor("logits")
    init_canvas = _FakeTensor("init-canvas")
    next_canvas = _FakeTensor("next-canvas")

    result = DL.TtDenoiseStepResult(
        canvas=next_canvas,
        accept_mask=_FakeTensor("accept"),
        entropy=_FakeTensor("entropy"),
        sampled=_FakeTensor("sampled"),
        argmax=_FakeTensor("argmax"),
    )

    class _StatefulLogits:
        prev_logits = None

        def __call__(self, canvas, step):
            del canvas, step
            self.prev_logits = logits
            return logits

        def owns_logits(self, value):
            return self.prev_logits is value

        def reset(self):
            self.prev_logits.deallocate(True)
            self.prev_logits = None

    logits_fn = _StatefulLogits()

    monkeypatch.setattr(DL, "denoise_step", lambda *args, **kwargs: result)
    monkeypatch.setattr(DL, "_ids_to_torch", lambda tensor: torch.ones(1, 1, dtype=torch.long))
    monkeypatch.setattr(DL, "_entropy_to_torch", lambda tensor: torch.zeros(1, 1, dtype=torch.float32))
    monkeypatch.setattr(DL, "_accept_to_torch", lambda tensor: torch.ones(1, 1, dtype=torch.bool))

    trajectory = DL.denoise_block(
        logits_fn,
        init_canvas,
        DiffusionConfig(max_denoise_steps=1, entropy_stop_threshold=1.0, stable_steps_to_halt=0),
        gumbel_noise_fn=lambda step: gumbel_noise,
        noise_tokens_fn=lambda step: noise_tokens,
    )

    assert trajectory.halted
    assert logits.deallocated
