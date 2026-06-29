# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest

from models.experimental.diffusion_gemma.tt import sampling as TS
from models.experimental.diffusion_gemma.tt.sampling_params import (
    MODEL_CAPABILITIES,
    canvas_sample_from_params,
    canvas_sampling_config_from_params,
)


@dataclass(frozen=True)
class DuckTypedTTSamplingParams:
    temperature: float | list[float]
    top_k: int | list[int]
    top_p: float | list[float]
    seed: int | list[int] | None = None


class _FakeLogits:
    shape = (2, 1, 4, 16)

    def device(self):
        return "mesh"


class _FakeNoise:
    def __init__(self, name):
        self.name = name
        self.deallocated = False

    def deallocate(self, force):
        self.deallocated = force


def test_canvas_sampling_params_defaults_and_capability():
    config = canvas_sampling_config_from_params(None, default_temperature=0.8, default_seed=47472)

    assert MODEL_CAPABILITIES["supports_sample_on_device"] is True
    assert config.temperature == pytest.approx(0.8)
    assert config.seed == 47472
    assert config.top_k is None
    assert config.top_p is None
    assert config.top_k_top_p_supported is False


def test_canvas_sampling_params_duck_type_vllm_fields():
    params = DuckTypedTTSamplingParams(
        temperature=[0.6, 0.7],
        top_k=[64, 32],
        top_p=[0.95, 0.9],
        seed=[1234, 5678],
    )

    config = canvas_sampling_config_from_params(params, default_temperature=0.8)

    assert config.temperature == pytest.approx(0.6)
    assert config.seed == 1234
    assert config.top_k == 64
    assert config.top_p == pytest.approx(0.95)
    assert config.top_k_top_p_supported is False


def test_canvas_sampling_params_rejects_greedy_temperature():
    params = {"temperature": 0.0, "top_k": 1, "top_p": 1.0}

    with pytest.raises(ValueError, match="temperature > 0"):
        canvas_sampling_config_from_params(params, default_temperature=0.8)


@pytest.mark.parametrize("seed", [0, -1])
def test_canvas_sampling_params_rejects_nonpositive_seed(seed):
    params = {"temperature": 0.8, "seed": seed}

    with pytest.raises(ValueError, match="positive nonzero"):
        canvas_sampling_config_from_params(params, default_temperature=0.8)


def test_canvas_sample_from_params_requires_noise_or_seed():
    params = {"temperature": 0.8}

    with pytest.raises(ValueError, match="gumbel_noise or a sampling seed"):
        canvas_sample_from_params(
            logits=None,
            sampling_params=params,
            default_temperature=0.8,
        )


def test_canvas_sample_from_params_rejects_multiple_rng_workarounds():
    params = {"temperature": 0.8, "seed": 47472}

    with pytest.raises(ValueError, match="at most one"):
        canvas_sample_from_params(
            logits=None,
            sampling_params=params,
            default_temperature=0.8,
            use_vocab_chunked_noise=True,
            use_vocab_permuted_noise=True,
        )


def test_canvas_sample_from_params_defaults_to_permuted_vocab_rng(monkeypatch):
    calls = {}

    def fake_permuted_noise(shape, *, device, seed):
        calls["noise"] = (shape, device, seed)
        return "permuted-gumbel"

    def fake_canvas_sample(logits, temperature, gumbel_noise):
        calls["sample"] = (logits, temperature, gumbel_noise)
        return "samples"

    monkeypatch.setattr(TS, "sample_gumbel_noise_with_permuted_vocab", fake_permuted_noise)
    monkeypatch.setattr(TS, "canvas_sample", fake_canvas_sample)

    logits = _FakeLogits()
    out = canvas_sample_from_params(
        logits,
        {"temperature": 0.7, "seed": 47472},
        default_temperature=0.8,
    )

    assert out == "samples"
    assert calls["noise"] == (_FakeLogits.shape, "mesh", 47472)
    assert calls["sample"] == (logits, 0.7, "permuted-gumbel")


def test_canvas_sample_from_params_deallocates_generated_gumbel_noise(monkeypatch):
    calls = {}
    noise = _FakeNoise("permuted-gumbel")

    def fake_permuted_noise(shape, *, device, seed):
        calls["noise"] = (shape, device, seed)
        return noise

    def fake_canvas_sample(logits, temperature, gumbel_noise):
        calls["sample"] = (logits, temperature, gumbel_noise, gumbel_noise.deallocated)
        return "samples"

    monkeypatch.setattr(TS, "sample_gumbel_noise_with_permuted_vocab", fake_permuted_noise)
    monkeypatch.setattr(TS, "canvas_sample", fake_canvas_sample)

    logits = _FakeLogits()
    out = canvas_sample_from_params(
        logits,
        {"temperature": 0.7, "seed": 47472},
        default_temperature=0.8,
    )

    assert out == "samples"
    assert calls["noise"] == (_FakeLogits.shape, "mesh", 47472)
    assert calls["sample"] == (logits, 0.7, noise, False)
    assert noise.deallocated is True


def test_canvas_sample_from_params_deallocates_generated_gumbel_noise_on_failure(monkeypatch):
    noise = _FakeNoise("permuted-gumbel")

    def fake_permuted_noise(shape, *, device, seed):
        return noise

    def fail_canvas_sample(logits, temperature, gumbel_noise):
        assert gumbel_noise is noise
        assert noise.deallocated is False
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(TS, "sample_gumbel_noise_with_permuted_vocab", fake_permuted_noise)
    monkeypatch.setattr(TS, "canvas_sample", fail_canvas_sample)

    with pytest.raises(RuntimeError, match="sampling failed"):
        canvas_sample_from_params(
            _FakeLogits(),
            {"temperature": 0.7, "seed": 47472},
            default_temperature=0.8,
        )

    assert noise.deallocated is True


def test_canvas_sample_from_params_preserves_injected_gumbel_noise(monkeypatch):
    calls = {}
    noise = _FakeNoise("injected-gumbel")

    def fake_canvas_sample(logits, temperature, gumbel_noise):
        calls["sample"] = (logits, temperature, gumbel_noise)
        return "samples"

    monkeypatch.setattr(TS, "canvas_sample", fake_canvas_sample)

    logits = _FakeLogits()
    out = canvas_sample_from_params(
        logits,
        {"temperature": 0.7},
        default_temperature=0.8,
        gumbel_noise=noise,
    )

    assert out == "samples"
    assert calls["sample"] == (logits, 0.7, noise)
    assert noise.deallocated is False


def test_canvas_sample_from_params_can_use_chunked_rng_without_disabling_default(monkeypatch):
    calls = {}

    def fake_chunked_noise(shape, *, device, seed, vocab_chunk_size):
        calls["noise"] = (shape, device, seed, vocab_chunk_size)
        return "chunked-gumbel"

    def fake_canvas_sample(logits, temperature, gumbel_noise):
        calls["sample"] = (logits, temperature, gumbel_noise)
        return "samples"

    monkeypatch.setattr(TS, "sample_gumbel_noise_by_vocab_chunks", fake_chunked_noise)
    monkeypatch.setattr(TS, "canvas_sample", fake_canvas_sample)

    logits = _FakeLogits()
    out = canvas_sample_from_params(
        logits,
        {"temperature": 0.7, "seed": 47472},
        default_temperature=0.8,
        use_vocab_chunked_noise=True,
        vocab_chunk_size=2,
    )

    assert out == "samples"
    assert calls["noise"] == (_FakeLogits.shape, "mesh", 47472, 2)
    assert calls["sample"] == (logits, 0.7, "chunked-gumbel")


def test_canvas_sample_from_params_can_use_raw_rng_for_diagnostics(monkeypatch):
    calls = {}

    def fake_raw_noise(shape, *, device, seed):
        calls["noise"] = (shape, device, seed)
        return "raw-gumbel"

    def fake_canvas_sample(logits, temperature, gumbel_noise):
        calls["sample"] = (logits, temperature, gumbel_noise)
        return "samples"

    monkeypatch.setattr(TS, "sample_gumbel_noise", fake_raw_noise)
    monkeypatch.setattr(TS, "canvas_sample", fake_canvas_sample)

    logits = _FakeLogits()
    out = canvas_sample_from_params(
        logits,
        {"temperature": 0.7, "seed": 47472},
        default_temperature=0.8,
        use_vocab_permuted_noise=False,
    )

    assert out == "samples"
    assert calls["noise"] == (_FakeLogits.shape, "mesh", 47472)
    assert calls["sample"] == (logits, 0.7, "raw-gumbel")
