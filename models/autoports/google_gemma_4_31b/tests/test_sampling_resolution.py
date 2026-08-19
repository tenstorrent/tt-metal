# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sampling-parameter resolution. Runs without vLLM installed, unlike
test_vllm_adapter_contract.py, because the logic lives on the generator."""

from types import SimpleNamespace

import pytest

from models.autoports.google_gemma_4_31b.tt.generator import Gemma4Generator

GREEDY = (1, 0.0, 1.0)


@pytest.mark.parametrize(
    "params",
    [
        None,
        SimpleNamespace(temperature=[0.0, 0.0], top_k=[262_144, 262_144]),
        SimpleNamespace(temperature=[1.0, 1.0], top_k=[1, 1]),
        SimpleNamespace(temperature=[0.7, 0.7], top_k=[1, 1], top_p=[1.0, 1.0]),
    ],
    ids=["none", "temperature-0", "top-k-1", "top-k-1-with-p-and-t"],
)
def test_every_greedy_form_keeps_the_dedicated_greedy_sampler(params):
    """top-1 is argmax whatever p and T are, so all forms must satisfy _is_semantic_greedy."""
    resolved = Gemma4Generator.resolve_sampling(params)
    assert resolved == GREEDY
    assert Gemma4Generator._is_semantic_greedy(top_k=resolved[0], top_p=resolved[1], temperature=resolved[2])


def test_vllm_default_request_is_clamped_to_the_device_top_k():
    """top_k < 1 means unrestricted; the shared formatter clamps it to the sampler's 32."""
    params = SimpleNamespace(temperature=[1.0, 1.0], top_k=[-1, -1], top_p=[1.0, 1.0])
    assert Gemma4Generator.resolve_sampling(params) == (32, 1.0, 1.0)


@pytest.mark.parametrize("requested, expected", [(0.8, 1.25), (2.0, 0.5)])
def test_temperature_is_inverted_for_the_sampling_kernel(requested, expected):
    """ttnn.sampling multiplies the top-k values by temp, so temp must be 1/T."""
    params = SimpleNamespace(temperature=[requested] * 2, top_k=[5, 5], top_p=[0.9, 0.9])
    top_k, top_p, temperature = Gemma4Generator.resolve_sampling(params)
    assert (top_k, top_p) == (5, 0.9)
    assert temperature == pytest.approx(expected)


def test_top_k_above_the_device_maximum_is_clamped_not_rejected():
    params = SimpleNamespace(temperature=[0.5, 0.5], top_k=[64, 64], top_p=[1.0, 1.0])
    assert Gemma4Generator.resolve_sampling(params)[0] == 32


def test_per_slot_sampling_parameters_are_rejected_explicitly(expect_error):
    params = SimpleNamespace(temperature=[0.8, 0.5], top_k=[5, 7], top_p=[0.9, 0.9])
    with expect_error(ValueError, "one shared"):
        Gemma4Generator.resolve_sampling(params)
