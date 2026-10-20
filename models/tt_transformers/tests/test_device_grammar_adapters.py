# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device grammar opt-in for the tt_transformers multimodal adapters.

Mistral3 and Gemma3 declare ``supports_device_grammar`` and inherit the device
grammar contract from ``Generator``. Their ``decode_forward`` wrappers must hand
the plugin's deferred decode to ``Generator.decode_forward`` unchanged, so the
deferred payload reaches ``sample_deferred_decode``.
"""

import pytest
import torch

from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.generator_vllm import Gemma3ForConditionalGeneration, Mistral3ForConditionalGeneration

ADAPTERS = [Mistral3ForConditionalGeneration, Gemma3ForConditionalGeneration]


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda cls: cls.__name__)
def test_device_grammar_capability_is_declared(adapter):
    capabilities = adapter.model_capabilities

    assert capabilities["supports_sample_on_device"] is True
    assert capabilities["supports_device_grammar"] is True
    assert capabilities["max_device_top_k"] == 32
    assert capabilities.get("output_tokens_per_step", 1) == 1


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda cls: cls.__name__)
def test_deferred_decode_reaches_generator_unchanged(adapter, monkeypatch):
    forwarded = []
    payload = object()

    def fake_decode(self, *args, **kwargs):
        forwarded.append((args, kwargs))
        return payload

    monkeypatch.setattr(Generator, "decode_forward", fake_decode)
    wrapper = adapter.__new__(adapter)
    wrapper.data_parallel = 1  # read by Generator.__del__
    tokens = torch.zeros(32, 1, dtype=torch.int32)
    start_pos = torch.zeros(32, dtype=torch.int32)

    result = adapter.decode_forward(
        wrapper,
        tokens,
        start_pos,
        read_from_device=False,
        defer_device_sampling=True,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )

    assert result is payload
    args, kwargs = forwarded[0]
    assert args[0] is tokens and args[1] is start_pos
    assert kwargs["defer_device_sampling"] is True
    assert kwargs["read_from_device"] is False
    assert "sampling_params" not in kwargs
