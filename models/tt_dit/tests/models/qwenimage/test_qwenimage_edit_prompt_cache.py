# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for the Qwen-Image-Edit device-transformer prompt cache and trace reuse.

The device, the tt_dit tensor helpers and the Tracer are replaced by fakes, so these run without
hardware. They pin two serving properties of ``_DeviceTransformer``:

- a new pipeline call with the same prompt length uses its *own* prompt embeddings (the step-
  invariant cache is keyed on lengths, and the embeddings depend on the prompt and the input image);
- prompts whose lengths fall in the same bucket share one captured trace.

Run (no conftest, so no device is opened):
  python_env/bin/python -m pytest --noconftest \
    models/tt_dit/tests/models/qwenimage/test_qwenimage_edit_prompt_cache.py
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ....pipelines.qwenimage_edit import pipeline_qwenimage_edit as mod

_COMBINED_SEQ = 8  # divisible by the fake SP factor (4)
_CHANNELS = 3
_JOINT_DIM = 2


class _FakeTT:
    """Stand-in for a device tensor: wraps a torch tensor, has a stable identity like a buffer."""

    def __init__(self, data: torch.Tensor) -> None:
        self.data = data


class _FakeTracer:
    instances: list[_FakeTracer] = []

    def __init__(self, function, *, device, prep_run=True) -> None:  # noqa: ANN001, ARG002
        self._function = function
        self._inputs: dict | None = None
        self.prompt_copies = 0
        self.released = False
        _FakeTracer.instances.append(self)

    @property
    def inputs(self) -> dict:
        return self._inputs

    def __call__(self, *, traced: bool, **kwargs):  # noqa: ANN003, ANN204, ARG002
        if self._inputs is None:  # capture: the first call's tensors become the trace buffers
            self._inputs = dict(kwargs)
        else:  # replay: copy changed tensors into the captured buffers
            for name, new in kwargs.items():
                prev = self._inputs[name]
                if isinstance(new, _FakeTT) and new is not prev:
                    prev.data = new.data.clone()
                    if name == "prompt":
                        self.prompt_copies += 1
        return self._function(**self._inputs)

    def release_trace(self) -> None:
        self.released = True


class _FakeModel:
    def __init__(self) -> None:
        self.prompt_lengths: list[int] = []

    def forward(self, *, spatial, prompt, prompt_sequence_length, **_):  # noqa: ANN001, ANN003, ANN201
        assert prompt.data.shape[1] == prompt_sequence_length
        self.prompt_lengths.append(prompt_sequence_length)
        # Output depends on the prompt content, so a stale prompt shows up in the result.
        return _FakeTT(spatial.data + prompt.data.sum())


class _FakePosEmbed:
    def forward(self, img_shapes, txt_seq_lens, device):  # noqa: ANN001, ANN201, ARG002
        return torch.zeros(_COMBINED_SEQ, 2, dtype=torch.complex64), torch.zeros(
            txt_seq_lens[0], 2, dtype=torch.complex64
        )


@pytest.fixture
def transformer_factory(monkeypatch):
    fake_tensor = SimpleNamespace(
        from_torch=lambda t, **_: _FakeTT(t.clone()),
        to_torch=lambda t, **_: t.data,
    )
    fake_ttnn = SimpleNamespace(synchronize_device=lambda _dev: None, float32="float32")
    monkeypatch.setattr(mod, "tensor", fake_tensor)
    monkeypatch.setattr(mod, "ttnn", fake_ttnn)
    monkeypatch.setattr(mod, "Tracer", _FakeTracer)
    _FakeTracer.instances = []

    def make(prompt_bucket):  # noqa: ANN001, ANN202
        model = _FakeModel()
        dt = mod._DeviceTransformer(
            tt_model=model,
            config=None,
            pos_embed=_FakePosEmbed(),
            mesh_device=SimpleNamespace(shape=(4, 8)),
            sp_axis=0,
            trace=True,
            batch_cfg=False,
            prompt_bucket=prompt_bucket,
        )
        return dt, model

    return make


def _forward(dt, prompt: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
    dt.cache_context("cond")
    (out,) = dt(
        hidden_states=torch.zeros(1, _COMBINED_SEQ, _CHANNELS),
        timestep=torch.tensor([0.5]),
        guidance=None,
        encoder_hidden_states=prompt,
        encoder_hidden_states_mask=torch.ones(1, prompt.shape[1]),
        img_shapes=[[(1, 2, 2), (1, 2, 2)]],
    )
    return out


def _new_call(dt) -> None:  # noqa: ANN001
    dt.forward_times.clear()
    dt.reset_cfg_state()  # what QwenImageEditPipeline.__call__ does per request


def test_same_length_new_call_uses_its_own_prompt(transformer_factory) -> None:
    dt, _ = transformer_factory(prompt_bucket=None)
    first = torch.full((1, 5, _JOINT_DIM), 1.0)
    second = torch.full((1, 5, _JOINT_DIM), 2.0)

    out_a = [_forward(dt, first) for _ in range(3)]  # three denoise steps
    _new_call(dt)
    out_b = [_forward(dt, second) for _ in range(3)]

    assert all(torch.equal(o, torch.full_like(o, first.sum().item())) for o in out_a)
    assert all(torch.equal(o, torch.full_like(o, second.sum().item())) for o in out_b)
    # One trace serves both requests; the new prompt is copied in once, not per step.
    assert len(_FakeTracer.instances) == 1
    assert _FakeTracer.instances[0].prompt_copies == 1


def test_lengths_in_one_bucket_share_a_trace(transformer_factory) -> None:
    dt, model = transformer_factory(prompt_bucket=8)
    _forward(dt, torch.ones(1, 5, _JOINT_DIM))
    _new_call(dt)
    out = _forward(dt, torch.full((1, 7, _JOINT_DIM), 3.0))

    assert model.prompt_lengths == [8, 8]
    assert len(_FakeTracer.instances) == 1 and not _FakeTracer.instances[0].released
    # Zero padding adds nothing to the fake's prompt sum: the real rows came through.
    assert torch.equal(out, torch.full_like(out, 3.0 * 7 * _JOINT_DIM))


def test_crossing_a_bucket_recaptures(transformer_factory) -> None:
    dt, model = transformer_factory(prompt_bucket=8)
    _forward(dt, torch.ones(1, 5, _JOINT_DIM))
    _new_call(dt)
    _forward(dt, torch.ones(1, 9, _JOINT_DIM))

    assert model.prompt_lengths == [8, 16]
    assert len(_FakeTracer.instances) == 2 and _FakeTracer.instances[0].released


def test_no_bucket_keeps_exact_lengths(transformer_factory) -> None:
    dt, model = transformer_factory(prompt_bucket=None)
    _forward(dt, torch.ones(1, 5, _JOINT_DIM))
    _new_call(dt)
    _forward(dt, torch.ones(1, 6, _JOINT_DIM))

    assert model.prompt_lengths == [5, 6]
    assert len(_FakeTracer.instances) == 2
